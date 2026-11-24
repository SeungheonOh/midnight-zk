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
//! 1.3 GPU with `bufferDeviceAddress`. It has the same entry points as the
//! CUDA crate `midnight-gpu` and the Metal crate `midnight-metal`, and does
//! not depend on `midnight-proofs`: its functions match the hook fields of
//! `midnight_proofs::gpu::GpuHooks`.
//!
//! The host side is `midnight-accel`, generic over a GPU backend; this crate
//! is its Vulkan backend ([`ctx::Vulkan`]), on the device of [`device`], and
//! the entry points for it.
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
//! - A decline leaves the caller's buffer unchanged: a result is written only
//!   after the last command buffer completed without error.
//! - No entry point draws randomness or touches a transcript. Any pseudo-random
//!   value (the MSM mask) is a fixed function of the shape.
//! - Every failure, including a panic caught at the entry point, is a counted
//!   decline ([`declines`]), never a panic into the prover. A lost device
//!   (`VK_ERROR_DEVICE_LOST`) declines every later call `no-device`.
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
//! 64-bit accumulators (`shaderInt64`, the default where the device has it)
//! or 32-bit words only.
//!
//! On an integrated GPU device memory is host memory and every buffer is
//! mapped, as on Metal. On a discrete GPU buffers are device-local: the host
//! reaches them through staging buffers, and they are zeroed and read back on
//! the GPU.
//!
//! # Requests
//!
//! Per-request device buffers can hold witness values. Between
//! [`begin_request`] and [`end_request`] they are reused across hook calls;
//! [`end_request`] zeroes them. Without an open request every hook call zeroes
//! them before it returns. Large buffers come from a pool of zeroed buffers
//! because a new buffer stalls its first command buffer while it is mapped
//! for the GPU.
//!
//! # Thresholds
//!
//! A call below its class's threshold ([`thresholds`]) declines `below-min-k`.
//! The defaults ([`DEFAULT_THRESHOLDS`]) are the Metal backend's until they
//! are measured on Vulkan devices. By default only MSMs on a registered SRS
//! table ([`register_srs`]) go to the device: other bases need a per-call G1
//! check that costs more than the CPU MSM. An explicit threshold
//! ([`set_min_k`]) replaces every default and opens that one-shot path.
//!
//! # Environment
//!
//! - `MIDNIGHT_VULKAN=off`: use no device.
//! - `MIDNIGHT_VULKAN_MIN_K`: one threshold for every class.
//! - `MIDNIGHT_VULKAN_MSM_CPU_SHARE=off`: run every MSM on the device only.
//! - `MIDNIGHT_VULKAN_POOL_MIB`: pool size limit (default a quarter of the
//!   device-local memory budget).
//! - `MIDNIGHT_VULKAN_QUOTIENT_THREADS`: see [`quotient_threads`].
//! - `MIDNIGHT_VULKAN_KEEP_KEYS=1`: see [`keep_resident`].
//! - `MIDNIGHT_VULKAN_LOADER`: the Vulkan loader library to load instead of the
//!   system's.
//! - `MIDNIGHT_VULKAN_ARITH=64|32`: the arithmetic variant ([`variant`]).
//! - `MIDNIGHT_VULKAN_DEVICE`: the device, by its index in the Vulkan
//!   enumeration order or by text its name contains (ASCII case ignored);
//!   default: the first discrete GPU, else an integrated one.
//!
//! Test only, never in production:
//!
//! - `MIDNIGHT_VULKAN_ALLOW_CPU=1`: also take a CPU implementation (lavapipe),
//!   for CI runners without a GPU.
//! - `MIDNIGHT_VULKAN_DISCRETE=1`: treat the device as discrete (no host
//!   mapping of device buffers), to run that path on an integrated GPU.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

use std::ffi::c_void;

use midnight_accel as accel;
pub use midnight_accel::{
    ClassStats, G1AffineLimbs, G1JacobianLimbs, HostTimes, MAX_LOG_N, MAX_MSM_LOG, Scalar, Stats,
    Thresholds, WipeStats, field,
};

pub mod ctx;
pub mod device;
mod tuning;

/// The Vulkan backend of `midnight-accel`, for code generic over the backend.
pub use ctx::Vulkan;
pub use device::{Variant, VulkanInfo};
/// MSM path and plan of one call (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use midnight_accel::MsmStages;
/// The placement counts of the last quotient served (feature
/// `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub use midnight_accel::QuotientStages;
/// The quotient job layout, for integration tests.
#[doc(hidden)]
pub use midnight_accel::quotient::abi as quotient_abi;

/// Per-class break-evens against the stock CPU code: the Metal backend's
/// until measured on Vulkan devices. The one-shot MSM never breaks even
/// because of its G1 check.
pub const DEFAULT_THRESHOLDS: Thresholds = tuning::TUNING.thresholds;

// ---------------------------------------------------------------------------
// Device

/// Whether the SPIR-V of the kernels is embedded (the shader tools were found
/// at build time). Without it no device is usable.
pub const fn vulkan_built() -> bool {
    cfg!(midnight_vulkan_spirv)
}

/// The Vulkan device, without creating any pipeline. `None` without SPIR-V,
/// without a Vulkan loader or a usable device, or with `MIDNIGHT_VULKAN=off`.
pub fn present() -> Option<&'static VulkanInfo> {
    accel::present::<Vulkan>()?;
    ctx::probe()
}

/// Whether device calls can run: [`present`], and the context is made (the
/// first call makes it) and its device was not lost.
pub fn device() -> bool {
    accel::device::<Vulkan>()
}

/// The arithmetic variant of the kernels: `MIDNIGHT_VULKAN_ARITH`, else
/// variant A where the device has `shaderInt64`. `None` before the context is
/// made or when it failed.
pub fn variant() -> Option<Variant> {
    ctx::context().ok().map(|c| c.variant())
}

/// `(free, total)` device bytes. `total` is the budget of the device-local
/// heap (`VK_EXT_memory_budget`, else three quarters of the heap); `free` is
/// `total` minus the heap's usage plus the idle pool buffers, which are
/// released on demand. This is a budget hint, not a reservation.
pub fn mem_info() -> Option<(usize, usize)> {
    accel::mem_info::<Vulkan>()
}

/// The pipelines created so far: `(kernel, workgroup width, driver creation
/// time)`, in no particular order. Empty before the context is made.
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

// ---------------------------------------------------------------------------
// Thresholds and counters

/// The effective threshold of each class.
pub fn thresholds() -> Thresholds {
    accel::thresholds::<Vulkan>()
}

/// The explicit threshold, else the smallest per-class default.
pub fn min_k() -> u32 {
    accel::min_k::<Vulkan>()
}

/// Sets one threshold for every class and opens the one-shot MSM. Returns
/// `false` and keeps the old value when `k` is above [`MAX_LOG_N`].
pub fn set_min_k(k: u32) -> bool {
    accel::set_min_k::<Vulkan>(k)
}

/// Undoes [`set_min_k`].
pub fn reset_min_k() {
    accel::reset_min_k::<Vulkan>()
}

/// The service counters since the last [`reset_stats`].
pub fn stats() -> Stats {
    accel::stats::<Vulkan>()
}

/// The host time around the device work since the last [`reset_stats`]:
/// copies, zeroing and command buffers.
pub fn host_times() -> HostTimes {
    accel::host_times::<Vulkan>()
}

/// Every decline since the last [`reset_stats`]: `(class, reason, count)`.
/// The reasons are those of [`midnight_accel::declines`]; `no-device` also
/// covers a build without SPIR-V and `MIDNIGHT_VULKAN=off`.
pub fn declines() -> Vec<(&'static str, &'static str, u64)> {
    accel::declines::<Vulkan>()
}

/// Resets [`stats`], [`declines`] and [`host_times`].
pub fn reset_stats() {
    accel::reset_stats::<Vulkan>()
}

// ---------------------------------------------------------------------------
// MSM

/// The length admission of an MSM: `shape` for an empty input, unequal
/// lengths or a length above `2^max_log`; `below-min-k` when the size class
/// is below `msm_table`. [`msm_g1`] may still decline on the bases.
pub fn msm_admission(scalars: usize, bases: usize, max_log: u32) -> Result<(), &'static str> {
    accel::msm_admission::<Vulkan>(scalars, bases, max_log)
}

/// Whether [`msm_g1`] takes an MSM of `len` scalars over `bases`, judged on
/// the length and the bases only. It reads no scalar, so the answer does not
/// depend on the witness; [`msm_g1`] may still decline on the scalars. It does
/// not check the device.
pub fn msm_sized(len: usize, bases: &[G1AffineLimbs]) -> bool {
    accel::msm_sized::<Vulkan>(len, bases)
}

/// G1 MSM `sum scalars[i] * bases[i]`: the hook `GpuHooks::msm_g1`. `None`:
/// declined.
///
/// Every check on public facts (lengths, device, registered table, one-shot
/// admission) runs before any scalar is read. Then come the effective size
/// and reduction of the scalars and, on the one-shot path only, the G1 check
/// of every base.
pub fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    accel::msm_g1::<Vulkan>(scalars, bases)
}

/// [`msm_g1`] and [`msm_sized`] of the same call from one registry lookup, so
/// the bases are compared once per MSM. `sized` is decided before any scalar
/// is read and stands even if the MSM declines or panics.
pub fn msm_g1_sized(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> (bool, Option<G1JacobianLimbs>) {
    accel::msm_g1_sized::<Vulkan>(scalars, bases)
}

/// [`msm_g1`] without the thresholds, reporting the path and plan. Not
/// counted in [`stats`] or [`declines`]; not a stable interface.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn msm_g1_stages(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> Option<(G1JacobianLimbs, MsmStages)> {
    accel::msm_g1_stages::<Vulkan>(scalars, bases)
}

/// Per-call G1 checks started on the one-shot path since process start (not
/// a stable interface).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn one_shot_g1_checks() -> u64 {
    accel::one_shot_g1_checks::<Vulkan>()
}

/// The registration of an SRS base slice. While it lives, the registrant
/// keeps the slice alive and unchanged. Dropping it frees the table.
pub type SrsHandle = accel::SrsHandle<Vulkan>;

/// Registers a host copy of an SRS base slice; the first MSM on it uploads
/// it. Registering the same slice again replaces the table. The KZG prover
/// commits from four `ParamsKZG` base vectors, so register all four.
///
/// Every point is checked to be in G1 once here. A table with a point outside
/// G1 is still registered, and MSMs on a prefix that includes it decline.
pub fn register_srs(bases: &[G1AffineLimbs]) -> SrsHandle {
    accel::register_srs::<Vulkan>(bases)
}

/// Live registered tables.
pub fn srs_handles() -> usize {
    accel::srs_handles::<Vulkan>()
}

/// Tables dropped at an MSM because the caller's slice no longer matched.
pub fn srs_stale() -> u64 {
    accel::srs_stale::<Vulkan>()
}

/// (MSMs served from a registered table, one-shot MSMs, table uploads) since
/// process start.
pub fn msm_counts() -> (u64, u64, u64) {
    accel::msm_counts::<Vulkan>()
}

// ---------------------------------------------------------------------------
// Transforms and the quotient

/// The FFT of `a` (length `2^log_n`) at `omega`, natural order, in place:
/// the hook `GpuHooks::fft`. `false`: declined, `a` unchanged.
pub fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    accel::fft::<Vulkan>(a, omega, log_n)
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place: the hook `GpuHooks::ifft`. `false`: declined, `a` unchanged.
pub fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    accel::ifft::<Vulkan>(a, omega_inv, log_n, divisor)
}

/// The quotient `h(X)` of a job (`midnight_proofs::gpu::abi::QuotientJobAbi`):
/// the hook `GpuHooks::quotient`, writing the first `out.len()` coefficients.
/// `false`: declined, `out` unchanged.
///
/// # Safety
/// `job` is null or points to a valid `QuotientJobAbi` whose pointers live
/// for the call.
pub unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
    // SAFETY: the caller's contract, passed on.
    unsafe { accel::quotient::<Vulkan>(job, out) }
}

/// The registration of proving-key polynomials. While it lives the registrant
/// keeps them alive and, within a request, unchanged. Dropping it frees the
/// device copies.
pub type KeyHandle = accel::KeyHandle<Vulkan>;

/// Registers proving-key polynomials by their digests. A polynomial
/// registered again moves to the new handle. Each length has its own device
/// copy, and a quotient serves from the copy of its extended length only.
pub fn register_key(polys: &[&[Scalar]]) -> KeyHandle {
    accel::register_key::<Vulkan>(polys)
}

/// Frees every device copy of proving-key polynomials (registrations stay).
/// Returns the bytes freed; a copy lent to a running quotient is freed when
/// returned and not counted.
pub fn evict_key_copies() -> u64 {
    accel::evict_key_copies::<Vulkan>()
}

/// Whether the device copies of registered proving-key polynomials stay after
/// a request (default `false`, or `MIDNIGHT_VULKAN_KEEP_KEYS=1`).
pub fn keep_resident() -> bool {
    accel::keep_resident::<Vulkan>()
}

/// Sets [`keep_resident`].
pub fn set_keep_resident(on: bool) {
    accel::set_keep_resident::<Vulkan>(on)
}

/// Registered polynomials now.
pub fn keys_registered() -> usize {
    accel::keys_registered::<Vulkan>()
}

/// Registered polynomials dropped at a quotient because their content
/// changed, since process start.
pub fn key_stale() -> u64 {
    accel::key_stale::<Vulkan>()
}

/// An upper bound of the device bytes one quotient job allocates. The
/// arguments are the fields of `midnight_proofs::gpu::QuotientShape`.
pub fn need_bytes(ext_k: u32, proof_polys: usize, key_polys: usize, nslots: u32) -> usize {
    accel::need_bytes::<Vulkan>(ext_k, proof_polys, key_polys, nslots)
}

/// (quotients computed on the device, quotient jobs declined) since process
/// start.
pub fn quotient_counts() -> (u64, u64) {
    accel::quotient_counts::<Vulkan>()
}

/// The threads of one numerator dispatch, which is also the scratch slot
/// stride: `MIDNIGHT_VULKAN_QUOTIENT_THREADS`, else 2^16; a multiple of 256 in
/// `[256, 2^24]`.
pub fn quotient_threads() -> u32 {
    accel::quotient_threads::<Vulkan>()
}

/// NTT building blocks (not a stable interface).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod ntt_test_support {
    use midnight_accel::ntt::test_support as t;

    use crate::{Scalar, Vulkan};

    /// `coeff_to_extended` on the device.
    pub fn coeff_to_extended(
        coeffs: &[Scalar],
        k: u32,
        ext_k: u32,
        zeta: [Scalar; 2],
    ) -> Result<Vec<Scalar>, String> {
        t::coeff_to_extended::<Vulkan>(coeffs, k, ext_k, zeta)
    }

    /// `extended_to_coeff` on the device.
    pub fn extended_to_coeff(
        values: &[Scalar],
        ext_k: u32,
        divisor: Scalar,
        zeta_out: [Scalar; 2],
    ) -> Result<Vec<Scalar>, String> {
        t::extended_to_coeff::<Vulkan>(values, ext_k, divisor, zeta_out)
    }

    /// The NTT of a copy of `values`.
    pub fn ntt(
        values: &[Scalar],
        log_n: u32,
        inverse: bool,
        scale: Option<Scalar>,
    ) -> Result<Vec<Scalar>, String> {
        t::ntt::<Vulkan>(values, log_n, inverse, scale)
    }

    /// The elementwise blocks on a copy of `values`.
    pub fn elementwise(
        values: &[Scalar],
        zeta: [Scalar; 2],
        c: Scalar,
        table: &[Scalar],
    ) -> Result<Vec<Scalar>, String> {
        t::elementwise::<Vulkan>(values, zeta, c, table)
    }

    pub use t::{passes, root};
}

/// Quotient test entry points (not a stable interface).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod quotient_test_support {
    use std::ffi::c_void;

    use midnight_accel::quotient::test_support as t;

    use crate::{QuotientStages, Scalar, Vulkan};

    /// The exact memory need of `job` now.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn need_bytes_job(job: *const c_void) -> Result<usize, &'static str> {
        // SAFETY: the caller's contract.
        unsafe { t::need_bytes_job::<Vulkan>(job) }
    }

    /// The placement counts of the last quotient served.
    pub fn last_stages() -> Option<QuotientStages> {
        t::last_stages::<Vulkan>()
    }

    /// A device-memory limit for the admission.
    pub fn set_memory_limit(limit: Option<u64>) {
        t::set_memory_limit::<Vulkan>(limit)
    }

    /// The numerator's target GPU time per command buffer.
    pub fn set_target_nanos(ns: Option<u64>) {
        t::set_target_nanos::<Vulkan>(ns)
    }

    /// The dispatch threads of the numerator.
    pub fn set_threads(threads: Option<u32>) {
        t::set_threads::<Vulkan>(threads)
    }

    /// The numerator of a job, read back.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn numerator(job: *const c_void) -> Result<Vec<Scalar>, &'static str> {
        // SAFETY: the caller's contract.
        unsafe { t::numerator::<Vulkan>(job) }
    }
}

// ---------------------------------------------------------------------------
// Test switches

/// Makes the hook `class` panic inside its backend (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_hook_panic(class: &str, on: bool) {
    accel::set_hook_panic::<Vulkan>(class, on)
}

/// Hook calls whose backend panicked since process start.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn hook_panics() -> u64 {
    accel::hook_panics::<Vulkan>()
}

// ---------------------------------------------------------------------------
// Requests and memory

/// Starts a request: per-request buffers are kept for reuse until
/// [`end_request`]. Also resets the peaks of [`device_bytes`] and
/// [`request_peak_bytes`].
pub fn begin_request() {
    accel::begin_request::<Vulkan>()
}

/// Ends a request: waits for device work and background zeroing, zeroes the
/// per-request buffers and returns them to the pool, then frees the
/// proving-key copies unless [`keep_resident`]. With `check`, every zeroed
/// region and every idle pool buffer is read back.
///
/// `Err` when a region read back non-zero or unreadable, or a zeroing
/// panicked. Such a region is zeroed again and released, never pooled.
pub fn end_request_report(check: bool) -> Result<WipeStats, WipeStats> {
    accel::end_request_report::<Vulkan>(check)
}

/// [`end_request_report`] without the read-back; `true` on success.
pub fn end_request() -> bool {
    accel::end_request::<Vulkan>()
}

/// Releases the pool of zeroed buffers once background zeroing has ended.
/// Returns the bytes released.
pub fn release_pool() -> u64 {
    accel::release_pool::<Vulkan>()
}

/// Bytes of idle, zeroed buffers in the pool.
pub fn pool_bytes() -> u64 {
    accel::pool_bytes::<Vulkan>()
}

/// `(now, peak)` bytes held in this crate's Vulkan buffers, idle pool buffers
/// included. The peak is since process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    accel::device_bytes::<Vulkan>()
}

/// Peak bytes in use since the last [`begin_request`]: buffers handed out,
/// resident SRS tables and key copies, not idle pool buffers. It starts from
/// the bytes in use at [`begin_request`], so it includes what earlier requests
/// left resident.
pub fn request_peak_bytes() -> u64 {
    accel::request_peak_bytes::<Vulkan>()
}

/// Helpers of the crate's tests (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_support {
    use std::time::Duration;

    use crate::device::{Context, Grid, Params, PipelineOptions, Result, Variant};

    /// The backend's device; panics when there is none.
    pub fn context() -> &'static Context {
        match crate::ctx::context() {
            Ok(c) => c.device(),
            Err(e) => panic!("no usable Vulkan device: {e}"),
        }
    }

    /// Runs `kernel` once over `grid` with `options`.
    pub fn run_with(
        kernel: &str,
        variant: Variant,
        options: PipelineOptions,
        params: &Params<'_>,
        grid: Grid,
    ) -> Result<Duration> {
        let c = context();
        let p = c.pipeline(kernel, variant, options)?;
        let mut cmds = c.commands()?;
        cmds.dispatch(&p, params, grid)?;
        cmds.submit()
    }

    /// Runs `kernel` once over `threads` invocations with the default
    /// pipeline options.
    pub fn run(
        kernel: &str,
        variant: Variant,
        params: &Params<'_>,
        threads: u64,
    ) -> Result<Duration> {
        run_with(
            kernel,
            variant,
            PipelineOptions::default(),
            params,
            Grid::Threads(threads),
        )
    }
}

#[cfg(test)]
mod tests;
