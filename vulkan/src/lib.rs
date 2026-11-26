// This file is part of MIDNIGHT-ZK.
// Copyright (C) Midnight Foundation
// SPDX-License-Identifier: Apache-2.0
// Licensed under the Apache License, Version 2.0 (the "License");
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Vulkan backend for the accelerator hooks of `midnight-proofs` (feature
//! `gpu`): BLS12-381 G1 MSM, scalar-field NTT and the quotient on any Vulkan
//! 1.3 GPU with `bufferDeviceAddress`, or an Android Vulkan 1.1+ GPU with
//! the equivalent feature. It has the entry points of
//! `midnight-metal` (the hook fields of `midnight_proofs::gpu::GpuHooks`)
//! over the generic host side `midnight-accel`, with [`Vulkan`] as the
//! backend.
//!
//! ```ignore
//! midnight_proofs::gpu::install(midnight_proofs::gpu::GpuHooks {
//!     msm_g1: Some(midnight_vulkan::msm_g1),
//!     fft: Some(midnight_vulkan::fft),
//!     ifft: Some(midnight_vulkan::ifft),
//!     quotient: Some(midnight_vulkan::quotient),
//! });
//! midnight_vulkan::begin_request();
//! let proof = prove(/* ... */);
//! assert!(midnight_vulkan::end_request());
//! ```
//!
//! # Hook contract
//!
//! - Every entry point returns the exact value of the stock CPU code, or
//!   declines (`None` / `false`) and the caller runs the CPU code.
//! - A decline leaves the caller's buffer unchanged.
//! - No entry point draws randomness or touches a transcript. Any pseudo-random
//!   value (the MSM mask) is a fixed function of the shape.
//! - Every failure, including a panic caught at the entry point, is a counted
//!   decline ([`declines`]), never a panic into the prover. A lost device
//!   (`VK_ERROR_DEVICE_LOST`) declines every later call `no-device`.
//!
//! Per-request device buffers can hold witness values: [`end_request`], or
//! the end of a hook call outside a request, zeroes them.
//!
//! A call below its class's threshold ([`thresholds`]) declines `below-min-k`.
//! By default only MSMs on a registered SRS table ([`register_srs`]) go to the
//! device; an explicit threshold ([`set_min_k`]) also opens MSMs on other
//! bases, which need a per-call G1 check.
//!
//! # Device and shaders
//!
//! The Vulkan loader is loaded at run time, so the crate builds on every
//! target; without a loader or a usable device [`present`] is `None` and
//! every entry point declines `no-device`. A discrete GPU is preferred to an
//! integrated one; CPU implementations are skipped.
//!
//! `build.rs` compiles `shaders/*.comp` to SPIR-V and embeds it; a build
//! without the shader tools has no SPIR-V and finds no device
//! ([`vulkan_built`]). Every kernel exists in two arithmetic [`Variant`]s:
//! 64-bit accumulators (`shaderInt64`) or 32-bit words only.
//!
//! On an integrated GPU (including mobile GPUs such as Qualcomm Adreno) every
//! buffer is mapped, as on Metal. On a discrete GPU buffers are device-local:
//! the host reaches them through staging buffers, and they are zeroed and read
//! back on the GPU.
//!
//! On Qualcomm Adreno (`vendor_id == 0x5143`), `msm_accumulate`,
//! `msm_integrate`, and `msm_ones` are capped to 128-thread workgroups,
//! `msm_ones` reduces through shared memory (`ONES_LANES = 1`), and an
//! Adreno-specific MSM tuning profile is selected automatically.
//!
//! # Environment
//!
//! - `MIDNIGHT_VULKAN=off`: use no device.
//! - `MIDNIGHT_VULKAN_MIN_K`: one threshold for every class ([`set_min_k`]).
//! - `MIDNIGHT_VULKAN_MSM_CPU_SHARE=off`: run every MSM on the device only.
//! - `MIDNIGHT_VULKAN_POOL_MIB`: pool size limit (default a quarter of the
//!   device-local memory budget).
//! - `MIDNIGHT_VULKAN_QUOTIENT_THREADS`: see [`quotient_threads`].
//! - `MIDNIGHT_VULKAN_KEEP_KEYS=1`: see [`keep_resident`].
//! - `MIDNIGHT_VULKAN_LOADER`: the Vulkan loader library to load instead of the
//!   system's.
//! - `MIDNIGHT_VULKAN_PIPELINE_CACHE`: optional file path to persist the
//!   driver's `VkPipelineCache` across process restarts (avoids repeated cold
//!   shader compilation, notably on mobile drivers such as Qualcomm Adreno).
//! - `MIDNIGHT_VULKAN_ARITH=64|32`: the arithmetic variant ([`variant`]).
//! - `MIDNIGHT_VULKAN_DEVICE`: the device, by its index in the Vulkan
//!   enumeration order or by text its name contains (ASCII case ignored);
//!   default: the first discrete GPU, else an integrated one.
//! - `MIDNIGHT_VULKAN_ALLOW_CPU=1` (test only): also take a CPU implementation
//!   (lavapipe).
//! - `MIDNIGHT_VULKAN_DISCRETE=1` (test only): treat the device as discrete, to
//!   run that path on an integrated GPU.
//!
//! Build time: `MIDNIGHT_VULKAN_GLSL`, `MIDNIGHT_VULKAN_SPIRV_OPT` and
//! `MIDNIGHT_VULKAN_REQUIRE_SPIRV` (see `build.rs`). On Android, set
//! `MIDNIGHT_VULKAN_TRACE_PIPELINES=1` at build time to log pipeline compiler
//! inputs, results and timing to logcat under the `MidnightVk` tag.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod ctx;
pub mod device;
mod tuning;

/// The Vulkan backend of `midnight-accel`, for code generic over the backend.
pub use ctx::Vulkan;
pub use device::{Variant, VulkanInfo};
pub use midnight_accel::HostTimes;

midnight_accel::export_backend!(Vulkan, "MIDNIGHT_VULKAN");

/// Default per-class break-evens against the stock CPU code: the Metal
/// backend's values. Qualcomm Adreno GPUs select their own defaults
/// ([`thresholds`]). The one-shot MSM never breaks even because of its G1
/// check.
pub const DEFAULT_THRESHOLDS: Thresholds = tuning::TUNING.thresholds;

/// Whether the SPIR-V of the kernels is embedded (the shader tools were found
/// at build time). Without it no device is usable.
pub const fn vulkan_built() -> bool {
    cfg!(midnight_vulkan_spirv)
}

/// The Vulkan device, without creating any pipeline. `None` without SPIR-V,
/// without a Vulkan loader or a usable device, or with `MIDNIGHT_VULKAN=off`.
pub fn present() -> Option<&'static VulkanInfo> {
    midnight_accel::present::<Vulkan>()?;
    ctx::probe()
}

/// The arithmetic variant of the kernels: `MIDNIGHT_VULKAN_ARITH`, else
/// [`Variant::Int64`] where the device has `shaderInt64`. `None` before the
/// context is made or when it failed.
pub fn variant() -> Option<Variant> {
    ctx::context().ok().map(|c| c.variant())
}

/// The pipelines created so far: `(kernel, workgroup width, driver creation
/// time)`, in no particular order.
pub fn pipeline_times() -> Vec<(&'static str, u32, std::time::Duration)> {
    let Ok(c) = ctx::context() else {
        return Vec::new();
    };
    c.device()
        .pipelines()
        .iter()
        .map(|p| (p.name(), p.workgroup(), p.creation_time()))
        .collect()
}

/// The host time around the device work since the last [`reset_stats`]:
/// copies, zeroing and command buffers.
pub fn host_times() -> HostTimes {
    midnight_accel::host_times::<Vulkan>()
}
