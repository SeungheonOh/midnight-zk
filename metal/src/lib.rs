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

//! # midnight_metal
//!
//! Metal kernels for the accelerator hooks of `midnight-proofs` (feature
//! `gpu`) on Apple GPUs: BLS12-381 G1 MSM, scalar-field NTT, and the
//! quotient (coset transforms, numerator, division by the vanishing
//! polynomial, extended inverse transform). It is the counterpart of the CUDA
//! crate `midnight-gpu`, with the same entry points.
//!
//! Every entry point returns the exact value of the stock CPU code or
//! declines (`None` / `false`); on a decline the caller's buffer is unchanged
//! and the caller runs the CPU code. The device is never a correctness
//! dependency.
//!
//! This crate does not depend on midnight-proofs. Its functions have the
//! signatures of the hook fields of `midnight_proofs::gpu::GpuHooks`, so a
//! process installs them as they are, and brackets each proof with a request:
//!
//! ```ignore
//! midnight_proofs::gpu::install(midnight_proofs::gpu::GpuHooks {
//!     msm_g1: Some(midnight_metal::msm_g1),
//!     fft: Some(midnight_metal::fft),
//!     ifft: Some(midnight_metal::ifft),
//!     quotient: Some(midnight_metal::quotient),
//! });
//! midnight_metal::begin_request();
//! let proof = prove(/* ... */);
//! assert!(midnight_metal::end_request()); // or end_request_report(true)
//! ```
//!
//! On macOS and iOS the crate depends on midnight-curves for the CPU share of
//! a G1 MSM, which runs the stock `msm_pippenger` (see the `msm` module, "The
//! CPU share"). On other targets the crate builds without device code and
//! every entry point declines `no-device`.
//!
//! ## The hook contract
//!
//! 1. Every entry point returns the exact value of the stock CPU code on the
//!    same input, or declines.
//! 2. A decline leaves the caller's buffer unchanged: a kernel module writes
//!    the caller's slice only with the complete device result, after the last
//!    command buffer has completed without error.
//! 3. No entry point draws a random value or touches a transcript. Any
//!    pseudo-random value a kernel needs (for example the MSM's load-balancing
//!    mask) is a fixed public function of the input shape.
//! 4. Every failure (no device, an allocation, a pipeline, a command buffer in
//!    the error state) is a counted decline, never a panic. A panic inside a
//!    kernel module is caught at the entry point and becomes the decline
//!    `panic` (with the caller's buffer unchanged, per 2), so it never unwinds
//!    into the prover (unless the process is built with `panic = "abort"`).
//!
//! ## Data layout
//!
//! The in-memory form of the `midnight-curves` types (through the zero-copy
//! adapter `midnight_proofs::gpu::adapter`), the same bytes on the device:
//! - [`Scalar`]: `blst_fr`, four little-endian `u64` limbs of the Montgomery
//!   form (`R = 2^256`), reduced modulo `r` ([`field::FR_MODULUS`]). Device
//!   type `Fr` (`shaders/include/fr.h`), eight 32-bit limbs.
//! - [`G1AffineLimbs`]: `blst_p1_affine`, `x` then `y`, six limbs each,
//!   Montgomery (`R = 2^384`); the point at infinity is all zero. Device type
//!   `Fp` (`shaders/include/fp.h`), twelve 32-bit limbs per coordinate.
//! - [`G1JacobianLimbs`]: `blst_p1`, Jacobian `X`, `Y`, `Z`.
//!
//! ## Requests and wiping
//!
//! Call [`begin_request`] and [`end_request`] (or [`end_request_report`])
//! around each proof. Between them the per-request buffers (device scratch
//! that can hold witness values: the MSM's scalars, digits, histograms,
//! temporaries and buckets, the transforms' data and scratch, the quotient's
//! transform scratch) are kept and reused by the next hook call.
//! [`end_request`] zeroes them; [`end_request_report`]`(true)` also reads
//! each one back from the host mapping, reads back every idle buffer of the
//! pool, and counts the regions that are not zero.
//!
//! Without a request (a process that only installs the hooks), every hook
//! call zeroes the per-request buffers before it returns, on success,
//! decline, error or a caught panic, with no read-back: nothing stays in
//! device-visible memory between calls, at the cost of making and zeroing
//! the scratch at every call.
//!
//! Scoped buffers (owned by one call) are zeroed when dropped, at the end of
//! that call or on a background thread that [`end_request`] waits for. The
//! large buffers, per-request and scoped, come from a pool of zeroed buffers
//! and go back to it after their zeroing instead of being released: a new
//! buffer makes the first command buffer that uses it wait while its pages
//! are mapped for the GPU (about 20 ms for 448 MiB on an M4 Max), a pooled
//! one does not. Every idle buffer of the pool reads zero. The pool keeps at
//! most `MIDNIGHT_METAL_POOL_MIB` (default a quarter of
//! `recommendedMaxWorkingSetSize`); [`release_pool`] releases it and
//! [`pool_bytes`] says what it holds.
//!
//! A region that fails the read-back is zeroed again, read back again and
//! released (never pooled, never kept as scratch), and the report is `Err`.
//! There is no quarantine and no retry, unlike the wipe layer of the CUDA
//! crate.
//!
//! ## Size thresholds
//!
//! A call below its class's threshold declines `below-min-k` before any
//! device work, and the stock CPU code runs. Each class has its own default
//! ([`DEFAULT_THRESHOLDS`]), the measured break-even of the device against
//! the stock CPU code on Apple M4 Max (40-core GPU), release builds. One
//! explicit threshold ([`set_min_k`] or `MIDNIGHT_METAL_MIN_K`) replaces
//! every default; [`reset_min_k`] undoes [`set_min_k`]. [`thresholds`] gives
//! the effective values. Other machines may break even elsewhere: set one
//! threshold explicitly there.
//!
//! | Class | Default (log2) | Measured |
//! |---|---|---|
//! | `fft`, `ifft` | 15 (`log_n`) | the device wins from `2^15`, loses at `2^14` |
//! | `msm` on a registered SRS table | 13 (size class of the length) | wins from `2^13` in a proof |
//! | `msm` without a table (one-shot) | never (`None`) | loses at every size with its per-call G1 check |
//! | `quotient` | 14 (`ext_k`) | wins from `ext_k = 14` in a proof |
//!
//! The size class of an MSM length is its log2 rounded to the nearest
//! integer ([`msm_admission`]): the multi-open commits `n - 1` coefficients,
//! which are in the class of `n`.
//!
//! The table MSM and quotient thresholds come from whole proofs, not from
//! isolated kernels: between the device calls of a proof the GPU idles while
//! the CPU works, and an idle GPU runs the next command buffer at a lower
//! clock (a `2^14` MSM after 100 ms of idle takes 10.1 to 10.8 ms of GPU
//! time for its first one or two calls and 3.4 to 3.9 ms after). A proof that
//! sends the device only its quotient runs it at the low clock; one that also
//! sends its dense commitments keeps the GPU at the high clock. Proof times of
//! the rc1 chain circuit (`gpu-prover/bench/rc1`) at `k = 13`, warm, 8 to 12
//! proofs per process, two to four processes per setting, range of the
//! minimum and of the median per process in ms:
//!
//! | stock CPU build | `msm_table` 14, `quotient` 14 | 14, 15 | 13, 15 | 13, 14 |
//! |-----------------|------------|------------|------------|------------|
//! | 41.3 to 42.1, 42.6 to 43.8 | 44.0, 45.0 to 47.3 | 39.9 to 40.1, 41.5 to 41.9 | 36.8 to 37.0, 37.9 to 40.1 | 33.6 to 34.4, 34.1 to 36.3 |
//!
//! At `k = 14` these settings send the same calls (47.8 to 48.2, 49.4 to
//! 49.5; the stock build 76.9 to 77.1, 77.7 to 79.1). `fft` and `ifft` at 13,
//! 14 or 15 give the same proof times within 1 ms at `k = 13` and `k = 14`.
//!
//! ### MSMs without a registered table
//!
//! With the default thresholds the device takes only an MSM whose bases a
//! registered table serves ([`register_srs`]: the registered slice or a
//! prefix of it, every point equal to the table's, and no point of the
//! prefix outside G1). Any other MSM declines `unregistered-bases` before
//! any scalar is read and before any check of its points, and the stock CPU
//! MSM runs it: bases that were never registered, and a slice whose table
//! the identity comparison finds stale (the table is then dropped). A table
//! with a point outside G1 in the prefix declines `bases-not-in-subgroup`.
//! The reason is the per-call G1 check (on the curve and in the prime-order
//! subgroup) that the device needs on bases it has not checked before (see
//! the `msm` module, "Bases outside G1"): it alone takes 187 ms for `2^16`
//! points and 593 ms for `2^18` on Apple M4 Max, against about 25 ms and
//! 78 ms for the whole stock CPU MSM. The stock CPU MSM is exact for any
//! point on the curve and needs no such check. The decline depends only on
//! public facts (the registry and the points), so its count and timing do
//! not depend on the witness.
//!
//! The one-shot device path, with its per-call G1 check of every base, is
//! reached only with an explicit threshold, which sets `msm_one_shot` of
//! [`thresholds`] to that value.
//!
//! ### The effective size of an MSM
//!
//! An MSM is compared twice: by its length against its threshold, and by its
//! effective size, the count of its non-zero scalars (1 when all are zero),
//! against `2^12 + n/5` on a registered table with the default thresholds
//! (`n` the length). A mostly-zero MSM (a permutation product or quotient
//! limb with a handful of non-zero coefficients) costs the masked device MSM
//! as much as a dense one of its length, while the stock CPU MSM, whose time
//! follows the sparsity, finishes in about a millisecond; `n/5` is the
//! measured crossover on Apple M4 Max with the CPU share of the split active,
//! and `2^12` the part of it that does not grow with the length (see the
//! `msm` module). Below its need the MSM declines `below-min-k` before any
//! device work. The count is not kept, logged, counted or returned: the
//! ordinary `below-min-k` decline is the only observable. An explicit
//! threshold replaces the need entirely (`2^min_k`, no fraction of `n`, on
//! the table and the one-shot path), so `set_min_k(0)` admits every MSM.
//!
//! ## Registered tables and keys
//!
//! - SRS base tables ([`register_srs`], handle [`SrsHandle`]): a host copy of a
//!   base slice, checked to be in G1 once, uploaded at the first MSM on it and
//!   kept on the device while the handle lives. [`srs_handles`], [`srs_stale`]
//!   and [`msm_counts`] count them and their use.
//! - Proving-key polynomials ([`register_key`], handle [`KeyHandle`]): the
//!   quotient serves the job's fixed and permutation polynomials from resident
//!   device copies of their extended values. [`end_request`] frees the copies
//!   unless [`keep_resident`] (`MIDNIGHT_METAL_KEEP_KEYS=1`,
//!   [`set_keep_resident`]); [`evict_key_copies`] frees them at any time.
//!   [`need_bytes`] bounds the device memory of one quotient.
//!
//! ## Environment
//!
//! Run time:
//! - `MIDNIGHT_METAL=off`: use no device (every call declines `no-device`).
//! - `MIDNIGHT_METAL_MIN_K`: one log2 size threshold for every class, replacing
//!   the per-class defaults and opening the one-shot MSM (see "Size
//!   thresholds").
//! - `MIDNIGHT_METAL_MSM_CPU_SHARE=off`: run every G1 MSM on the device only.
//!   By default a measured share of the points goes to the stock CPU MSM, which
//!   runs at the same time as the device: from `2^17` on a registered table,
//!   from `2^13` on the one-shot path (see the `msm` module, "The CPU share").
//! - `MIDNIGHT_METAL_POOL_MIB`: the most bytes the pool of zeroed buffers keeps
//!   idle (default a quarter of `recommendedMaxWorkingSetSize`).
//! - `MIDNIGHT_METAL_QUOTIENT_THREADS`: the threads of one numerator dispatch
//!   ([`quotient_threads`]).
//! - `MIDNIGHT_METAL_KEEP_KEYS=1`: keep the proving-key copies between requests
//!   ([`keep_resident`]).
//! - `MIDNIGHT_METAL_SHADER_SOURCE=1`: compile the embedded shader source
//!   instead of loading the embedded metallib ([`library_source`]).
//!
//! Build time (see `build.rs`): `MIDNIGHT_METAL_SHADER_SOURCE=1` skips the
//! Metal toolchain and embeds the source only;
//! `MIDNIGHT_METAL_REQUIRE_METALLIB=1` makes a missing toolchain or a failed
//! shader compile a build error.
//!
//! Detection ([`present`]) loads no shader library; the first device call
//! ([`device`]) makes the context.
//!
//! ## Declines and counters
//!
//! [`declines`] lists every decline by class and reason, [`stats`] the calls
//! the device served and their GPU time; [`reset_stats`] clears both. The
//! reasons are listed at [`declines`].
//!
//! ## Limits
//!
//! Transforms up to `2^`[`MAX_LOG_N`], MSMs up to `2^`[`MAX_MSM_LOG`] terms,
//! each buffer at most `maxBufferLength` ([`MetalInfo`]). The shaders use the
//! Metal Shading Language 3.2 (macOS 15+, every Apple Silicon Mac runs it;
//! iOS 18+).
//!
//! ## Source layout
//!
//! - `src/lib.rs`: this API, admission, thresholds, the hook boundary.
//! - `src/account.rs`: decline and service accounting. A kernel module ends
//!   every call with exactly one of `account::served` or `account::decline`.
//! - `src/ctx.rs` ([`ctx`]): the device context: device, queue, library,
//!   pipeline cache, buffers (scoped, pooled, per-request), command buffers
//!   with error reporting, byte accounting, the erasure report.
//! - `src/field.rs` ([`field`]): moduli and the reduction check.
//! - `src/msm.rs`, `src/ntt.rs`, `src/quotient.rs`: one kernel family each. The
//!   entry point calls the module after admission.
//! - `build.rs`: the shader build (a metallib with `xcrun metal`, else the
//!   embedded source compiled at first use).
//! - `shaders/include/mont.h`: Montgomery arithmetic over N 32-bit limbs;
//!   `fr.h`, `fp.h`: the two fields on it (`fr_mul`, `fp_add`, ...).
//! - `shaders/*.metal`: kernel files. Each file is its own translation unit
//!   with the Metal toolchain, and all of them form one translation unit on the
//!   source path, so a kernel file includes the headers it needs (header guards
//!   make repeated includes harmless), prefixes its kernel names with its file
//!   stem (`ntt_*`, `msm_*`, `quotient_*`) and keeps its helpers in a namespace
//!   of that name. `build.rs` picks up a new file without any other change.
//!   `field_test.metal`, `msm_test.metal` and `diag.metal` hold test kernels,
//!   compiled only with the feature `test-support`.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
// Other targets keep the host-side code they do not call.
#![cfg_attr(
    not(any(target_os = "macos", target_os = "ios")),
    allow(dead_code, unused_imports)
)]

use std::{
    ffi::c_void,
    sync::{
        OnceLock,
        atomic::{AtomicU32, Ordering::Relaxed},
    },
};

mod account;
pub mod field;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub mod ctx;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod msm;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod ntt;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod quotient;

/// The MSM's test entry, which reports the path and the plan an MSM took
/// (public facts only), and the count of the one-shot path's per-call G1
/// checks (feature `test-support`; not a stable interface, not for secret
/// scalars).
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    any(test, feature = "test-support")
))]
#[doc(hidden)]
pub use msm::{MsmStages, msm_g1_stages, one_shot_g1_checks};
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use msm::{SrsHandle, msm_counts, register_srs, srs_handles, srs_stale};
/// Host-to-host wrappers of the NTT device building blocks (feature
/// `test-support`; not a stable interface).
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    any(test, feature = "test-support")
))]
#[doc(hidden)]
pub use ntt::test_support as ntt_test_support;
/// The stage timings of the last quotient served (feature `test-support`).
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    any(test, feature = "test-support")
))]
#[doc(hidden)]
pub use quotient::QuotientStages;
/// The quotient job layout, for this crate's integration tests (not a stable
/// interface).
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[doc(hidden)]
pub use quotient::abi as quotient_abi;
/// The quotient's test and measurement entry points (feature
/// `test-support`).
#[cfg(all(
    any(target_os = "macos", target_os = "ios"),
    any(test, feature = "test-support")
))]
#[doc(hidden)]
pub use quotient::test_support as quotient_test_support;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use quotient::{
    KeyHandle, evict_key_copies, keep_resident, key_stale, keys_registered, need_bytes,
    quotient_counts, register_key, set_keep_resident, threads as quotient_threads,
};

#[cfg(all(test, any(target_os = "macos", target_os = "ios")))]
mod tests;

use account::{Class, Reason};

/// A scalar: `blst_fr` limbs, Montgomery form (see the crate text).
pub type Scalar = [u64; 4];
/// An affine G1 point: `blst_p1_affine` limbs.
pub type G1AffineLimbs = [u64; 12];
/// A Jacobian G1 point: `blst_p1` limbs.
pub type G1JacobianLimbs = [u64; 18];

/// The largest log2 transform size taken.
pub const MAX_LOG_N: u32 = 28;
/// The largest log2 MSM length taken; a longer one declines (`shape`).
pub const MAX_MSM_LOG: u32 = 28;

/// log2 rounded to the nearest integer on the log scale: the floor of
/// `log2(n sqrt 2)`. The size class of an MSM length: its admission (see
/// "Size thresholds"), its window plan and its CPU share (the `msm` module)
/// all go by it, so that a length just below a power of two (the `n - 1`
/// coefficients of a multi-open quotient) is treated as that power.
pub(crate) const fn lg2_nearest(n: usize) -> u32 {
    // 181 / 128 = 1.4140625, just below sqrt 2
    let mut m = ((n as u128 * 181) >> 7) as usize;
    let mut r = 0;
    while m > 1 {
        m >>= 1;
        r += 1;
    }
    r
}

// ---------------------------------------------------------------------------
// Device

/// The Metal device as its properties describe it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalInfo {
    /// Device name (`MTLDevice.name`).
    pub name: String,
    /// `MTLDevice.registryID`.
    pub registry_id: u64,
    /// `hasUnifiedMemory`: the GPU shares the system memory (all Apple
    /// Silicon).
    pub unified_memory: bool,
    /// `recommendedMaxWorkingSetSize`: the bytes the GPU can use without
    /// hurting performance, Metal's guidance for the total of this process.
    pub recommended_max_working_set: u64,
    /// `maxBufferLength`: the largest single buffer in bytes.
    pub max_buffer_length: u64,
    /// `maxThreadsPerThreadgroup.width`.
    pub max_threads_per_threadgroup: u64,
    /// `maxThreadgroupMemoryLength`: the threadgroup memory of one
    /// threadgroup in bytes. A dispatch that needs more declines
    /// (`device-error`).
    pub max_threadgroup_memory_length: u64,
    /// The highest `MTLGPUFamilyAppleN` supported (0: none).
    pub apple_family: u32,
    /// Supports `MTLGPUFamilyMetal3`.
    pub metal3: bool,
    /// The Metal Shading Language version of the shaders.
    pub msl_version: &'static str,
}

/// Whether the Metal code is compiled in (a macOS or iOS target).
pub const fn metal_built() -> bool {
    cfg!(any(target_os = "macos", target_os = "ios"))
}

/// Whether `MIDNIGHT_METAL=off` (read once).
fn disabled() -> bool {
    static OFF: OnceLock<bool> = OnceLock::new();
    *OFF.get_or_init(|| std::env::var("MIDNIGHT_METAL").as_deref() == Ok("off"))
}

/// The Metal device, without making the context (no shader library is
/// loaded): `None` on other targets, with no device, or with
/// `MIDNIGHT_METAL=off`.
pub fn present() -> Option<&'static MetalInfo> {
    if disabled() {
        return None;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::probe()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    None
}

/// Whether device calls can run: [`present`], and the context (queue, shader
/// library) is made; the first call makes it.
pub fn device() -> bool {
    if present().is_none() {
        return false;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::context().is_ok()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    false
}

/// Where the shader library came from: `metallib` (compiled at build time),
/// or `source ...` (compiled from the embedded source at first use, with the
/// reason). `None` before the context is made or when it failed.
pub fn library_source() -> Option<&'static str> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::context().ok().map(|c| c.library_source())
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    None
}

/// `(free, total)` bytes for the device, when a device is usable.
///
/// `total` is `MTLDevice.recommendedMaxWorkingSetSize`: Metal's estimate of
/// the memory the GPU can use without hurting performance. `free` is `total`
/// minus `MTLDevice.currentAllocatedSize` (everything this process has
/// allocated on the device: buffers, the library, pipelines, driver
/// structures), plus the idle buffers of the pool ([`pool_bytes`]: a new
/// allocation that would not fit releases idle buffers first, so they are
/// free for any allocation), saturating at zero and at most `total`.
///
/// On unified memory these are neither physical figures nor reservations:
/// the GPU allocates from the same system memory as the CPU, other processes
/// and this process's own heap draw on it too, and nothing is set aside. A
/// `free` above zero does not promise that an allocation succeeds, and an
/// allocation beyond `total` can succeed (with paging). Use it as a budget
/// hint, not as a guarantee.
pub fn mem_info() -> Option<(usize, usize)> {
    if !device() {
        return None;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::mem_info()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    None
}

// ---------------------------------------------------------------------------
// Thresholds

/// The log2 size from which each kernel class is sent to the device (see
/// [`thresholds`] and the crate text, "Size thresholds").
///
/// `msm_one_shot` is an `Option`: `None` means the class never goes to the
/// device (the default, see "MSMs without a registered table" in the crate
/// text).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thresholds {
    /// Forward FFTs: the transform size `log_n`.
    pub fft: u32,
    /// Inverse FFTs: the transform size `log_n`.
    pub ifft: u32,
    /// MSMs served from a registered SRS table: the size class of the length.
    pub msm_table: u32,
    /// MSMs with no registered table that serves their bases (one-shot): the
    /// size class of the length; `None`: never, every such MSM declines
    /// `unregistered-bases` (the default). Only an explicit threshold sets
    /// it.
    pub msm_one_shot: Option<u32>,
    /// Quotients: the extended domain `ext_k`.
    pub quotient: u32,
    /// Whether one value was set explicitly ([`set_min_k`] or
    /// `MIDNIGHT_METAL_MIN_K`) and applies to every class; `false`: the
    /// measured per-class defaults ([`DEFAULT_THRESHOLDS`]).
    pub explicit: bool,
}

impl Thresholds {
    /// One value for every class (an explicit `min_k`).
    pub const fn all(k: u32) -> Thresholds {
        Thresholds {
            fft: k,
            ifft: k,
            msm_table: k,
            msm_one_shot: Some(k),
            quotient: k,
            explicit: true,
        }
    }

    /// `(class, log2 size)` pairs, for reports: `fft`, `ifft`, `msm-table`,
    /// `msm-one-shot`, `quotient-ext`; `None`: the class never goes to the
    /// device (`msm-one-shot` with the default thresholds).
    pub fn named(&self) -> [(&'static str, Option<u32>); 5] {
        [
            ("fft", Some(self.fft)),
            ("ifft", Some(self.ifft)),
            ("msm-table", Some(self.msm_table)),
            ("msm-one-shot", self.msm_one_shot),
            ("quotient-ext", Some(self.quotient)),
        ]
    }
}

/// The per-class break-evens measured on Apple M4 Max (40-core GPU), release
/// build: the size from which the device call is faster than the stock CPU
/// code (see the crate text, "Size thresholds"). The one-shot MSM never is,
/// with its per-call G1 check (`None`).
pub const DEFAULT_THRESHOLDS: Thresholds = Thresholds {
    fft: 15,
    ifft: 15,
    msm_table: 13,
    msm_one_shot: None,
    quotient: 14,
    explicit: false,
};

/// [`MIN_K`]: not read yet (the environment is read at the next use).
const MIN_K_UNREAD: u32 = u32::MAX;
/// [`MIN_K`]: no explicit value (the per-class defaults apply).
const MIN_K_DEFAULTS: u32 = u32::MAX - 1;
/// The explicit threshold, or one of the two markers above.
static MIN_K: AtomicU32 = AtomicU32::new(MIN_K_UNREAD);

/// The explicit threshold: [`set_min_k`], else `MIDNIGHT_METAL_MIN_K`
/// (ignored above [`MAX_LOG_N`]); `None`: none was set.
fn explicit_min_k() -> Option<u32> {
    let mut k = MIN_K.load(Relaxed);
    if k == MIN_K_UNREAD {
        let env = std::env::var("MIDNIGHT_METAL_MIN_K")
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|k| *k <= MAX_LOG_N)
            .unwrap_or(MIN_K_DEFAULTS);
        let _ = MIN_K.compare_exchange(MIN_K_UNREAD, env, Relaxed, Relaxed);
        k = MIN_K.load(Relaxed);
    }
    (k <= MAX_LOG_N).then_some(k)
}

/// The effective size threshold of each class: the explicit value
/// ([`set_min_k`], else `MIDNIGHT_METAL_MIN_K`) for every class when one was
/// set, else [`DEFAULT_THRESHOLDS`].
pub fn thresholds() -> Thresholds {
    explicit_min_k().map_or(DEFAULT_THRESHOLDS, Thresholds::all)
}

/// The explicit threshold when one was set ([`set_min_k`], else
/// `MIDNIGHT_METAL_MIN_K`), else the smallest per-class default (13). The
/// threshold of each class is [`thresholds`].
pub fn min_k() -> u32 {
    let t = thresholds();
    let m = t.fft.min(t.ifft).min(t.msm_table).min(t.quotient);
    t.msm_one_shot.map_or(m, |k| m.min(k))
}

/// Sets one threshold for every class (log2 size; for a quotient, the
/// extended domain), the one-shot MSM included, which this opens (see "MSMs
/// without a registered table" in the crate text). Returns `false` and
/// keeps the old value when `k` is above [`MAX_LOG_N`] (no size would be
/// admitted).
pub fn set_min_k(k: u32) -> bool {
    if k > MAX_LOG_N {
        return false;
    }
    MIN_K.store(k, Relaxed);
    true
}

/// Undoes [`set_min_k`]: `MIDNIGHT_METAL_MIN_K` is read again at the next
/// use, and without it the per-class defaults apply.
pub fn reset_min_k() {
    MIN_K.store(MIN_K_UNREAD, Relaxed);
}

// ---------------------------------------------------------------------------
// Counters

/// Calls served by the device in one hook class since the last
/// [`reset_stats`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassStats {
    /// Calls the device computed.
    pub served: u64,
    /// GPU time of those calls, nanoseconds (command-buffer GPU start to end).
    pub device_nanos: u64,
}

/// Per-class service counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// MSMs.
    pub msm: ClassStats,
    /// Forward FFTs.
    pub fft: ClassStats,
    /// Inverse FFTs.
    pub ifft: ClassStats,
    /// Quotients.
    pub quotient: ClassStats,
}

/// The service counters since the last [`reset_stats`].
pub fn stats() -> Stats {
    Stats {
        msm: account::class_stats(Class::Msm),
        fft: account::class_stats(Class::Fft),
        ifft: account::class_stats(Class::Ifft),
        quotient: account::class_stats(Class::Quotient),
    }
}

/// Every decline since the last [`reset_stats`]: `(class, reason, count)`.
///
/// Classes: `msm`, `fft`, `ifft`, `quotient`. Reasons:
/// - `no-device`: another target, no usable device, `MIDNIGHT_METAL=off`;
/// - `below-min-k`: below the class's threshold ([`thresholds`]), or an MSM
///   below its effective-size need;
/// - `shape`: a length, root, job layout or version the device does not take;
/// - `unreduced`: a scalar not below the modulus;
/// - `memory`: a buffer above `maxBufferLength`, a failed allocation or a
///   budget;
/// - `device-error`: a pipeline, encoder or command buffer failed;
/// - `panic`: the kernel module panicked and the entry point caught it;
/// - `bases-not-in-subgroup` (`msm` only): a base that is not a point of G1,
///   off the curve or outside the prime-order subgroup;
/// - `unregistered-bases` (`msm` only): no registered table serves the bases
///   (never registered, or found stale) and the one-shot path is off, as it is
///   with the default thresholds; decided before any scalar is read and before
///   any G1 check (see "MSMs without a registered table" in the crate text).
pub fn declines() -> Vec<(&'static str, &'static str, u64)> {
    account::declines()
}

/// Resets [`stats`] and [`declines`].
pub fn reset_stats() {
    account::reset();
}

// ---------------------------------------------------------------------------
// MSM

/// The admission of an MSM on its lengths: an empty input, unequal lengths
/// or a length above `2^max_log` is `shape`; a length whose size class (its
/// log2 rounded to the nearest integer, so `2^k - 1` is in class `k`) is
/// below `msm_table` of [`thresholds`] (the lower of the two MSM thresholds)
/// is `below-min-k`. When no registered table serves the bases, [`msm_g1`]
/// also declines: `unregistered-bases` when `msm_one_shot` is `None` (the
/// default), `below-min-k` below its class otherwise.
pub fn msm_admission(scalars: usize, bases: usize, max_log: u32) -> Result<(), &'static str> {
    length_admission(scalars, bases, max_log).map_err(Reason::name)
}

/// [`msm_admission`] with the reason as a [`Reason`].
fn length_admission(scalars: usize, bases: usize, max_log: u32) -> Result<(), Reason> {
    if scalars == 0 || scalars != bases || scalars > (1usize << max_log.min(62)) {
        return Err(Reason::Shape);
    }
    if lg2_nearest(scalars) < thresholds().msm_table {
        return Err(Reason::BelowMinK);
    }
    Ok(())
}

/// The one-shot admission of an MSM of `len` scalars whose bases no
/// registered table serves: `unregistered-bases` when `msm_one_shot` is
/// `None`, `below-min-k` below its class; else the threshold.
fn one_shot_admission(len: usize, t: &Thresholds) -> Result<u32, Reason> {
    match t.msm_one_shot {
        None => Err(Reason::UnregisteredBases),
        Some(k) if lg2_nearest(len) < k => Err(Reason::BelowMinK),
        Some(k) => Ok(k),
    }
}

/// Whether [`msm_g1`] takes an MSM of `len` scalars over `bases` by its
/// length and bases: `len` passes [`msm_admission`] (with [`MAX_MSM_LOG`]),
/// and either a registered table serves `bases` (the same lookup and
/// identity comparison as [`msm_g1`], which drops a stale table, and no
/// point of the prefix outside G1), or none does and the one-shot path is
/// open with `len` in the class `msm_one_shot` of [`thresholds`] (with the
/// default thresholds it is not: `false`).
///
/// It reads no scalar: the checks that do (the effective size of a sparse
/// MSM, the reduction) come after it in [`msm_g1`], which may still
/// decline, and so may the per-call G1 check of the one-shot path, which
/// this does not run. A figure that does not depend on the scalar values,
/// for a caller that must not report one that does. It does not look at the
/// device ([`device`]).
///
/// A caller that also runs the MSM calls [`msm_g1_sized`] instead, which
/// gives this figure from the lookup and comparison of the MSM itself.
pub fn msm_sized(len: usize, bases: &[G1AffineLimbs]) -> bool {
    length_admission(len, bases.len(), MAX_MSM_LOG).is_ok() && sized_by_bases(len, bases)
}

/// [`msm_sized`] of a length that passed [`length_admission`].
fn sized_by_bases(len: usize, bases: &[G1AffineLimbs]) -> bool {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        sized_by_lookup(&msm::lookup(bases, len), len, &thresholds())
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = bases;
        one_shot_admission(len, &thresholds()).is_ok()
    }
}

/// [`msm_sized`] of a length that passed [`length_admission`], from the
/// lookup of its bases.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn sized_by_lookup(lookup: &msm::Lookup, len: usize, t: &Thresholds) -> bool {
    match lookup {
        msm::Lookup::Table(_) => true,
        msm::Lookup::NotInG1 => false,
        msm::Lookup::Unregistered => one_shot_admission(len, t).is_ok(),
    }
}

/// G1 MSM `sum scalars[i] * bases[i]` (equal lengths): the hook
/// `GpuHooks::msm_g1`. `None`: declined.
///
/// An MSM on a registered SRS table ([`register_srs`]) is taken from the
/// size class `msm_table` ([`thresholds`], [`msm_admission`]). An MSM whose
/// bases no registered table serves declines `unregistered-bases` with the
/// default thresholds (the stock CPU MSM then runs it; see "MSMs without a
/// registered table" in the crate text); with an explicit threshold it takes
/// the one-shot path from the class `msm_one_shot`. The count of non-zero
/// scalars (the effective size, 1 when all are zero) is held, with the
/// default thresholds, to `2^12 + n/5` (table), and with an explicit
/// threshold to `2^min_k` (either path), with the decline `below-min-k`.
///
/// The order of the checks: the lengths ([`msm_admission`]), the device,
/// the registered table (lookup, identity comparison of every point, the
/// table's G1 check) and the one-shot admission, all on public facts; then
/// the effective size and the reduction of the scalars; then, on the
/// one-shot path only, the G1 check of every base.
pub fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    guarded(Class::Msm, None, || msm_g1_body(scalars, bases, None))
}

/// [`msm_g1`], together with [`msm_sized`] of the same call: `(sized,
/// result)`. Both come from one registry lookup and one identity comparison
/// of every point, made within this call, so the bases are compared once
/// per MSM instead of once for each entry. `sized` is decided from the
/// length and the bases before any scalar is read, as [`msm_sized`] decides
/// it, whether the MSM is then served or declined (a panic in the MSM
/// leaves it as decided).
pub fn msm_g1_sized(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> (bool, Option<G1JacobianLimbs>) {
    let mut sized = false;
    let r = guarded(Class::Msm, None, || {
        msm_g1_body(scalars, bases, Some(&mut sized))
    });
    (sized, r)
}

/// The body of [`msm_g1`]; with `sized`, also stores there [`msm_sized`] of
/// the call ([`msm_g1_sized`]).
fn msm_g1_body(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    sized: Option<&mut bool>,
) -> Option<G1JacobianLimbs> {
    if let Err(r) = length_admission(scalars.len(), bases.len(), MAX_MSM_LOG) {
        account::decline(Class::Msm, r);
        return None;
    }
    if !device() {
        // `msm_sized` does not look at the device
        if let Some(s) = sized {
            *s = sized_by_bases(scalars.len(), bases);
        }
        account::decline(Class::Msm, Reason::NoDevice);
        return None;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let t = thresholds();
        let n = scalars.len();
        // the path, from public facts only (the registry and the points),
        // before any scalar is read: a table that serves the bases, or the
        // one-shot path when an explicit threshold opens it
        let lookup = msm::lookup(bases, n);
        if let Some(s) = sized {
            *s = sized_by_lookup(&lookup, n, &t);
        }
        let (table, need) = match lookup {
            msm::Lookup::Table(tb) => (
                Some(tb),
                msm::effective_need(n, t.msm_table, true, t.explicit),
            ),
            msm::Lookup::NotInG1 => {
                account::decline(Class::Msm, msm::BASES_REASON);
                return None;
            }
            msm::Lookup::Unregistered => match one_shot_admission(n, &t) {
                Ok(k) => (None, msm::effective_need(n, k, false, t.explicit)),
                Err(r) => {
                    account::decline(Class::Msm, r);
                    return None;
                }
            },
        };
        // the effective size (the non-zero scalars) against the need of the
        // path; only the decline is observable
        if !msm::effective_size_reaches(scalars, need) {
            account::decline(Class::Msm, Reason::BelowMinK);
            return None;
        }
        if !field::all_reduced(scalars) {
            account::decline(Class::Msm, Reason::Unreduced);
            return None;
        }
        let r = msm::msm_g1(scalars, bases, table);
        // the test panic comes after the backend used its scratch
        test_switch::hook_panic(Class::Msm);
        r
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    None
}

// ---------------------------------------------------------------------------
// Transforms

/// The admission shared by [`fft`] and [`ifft`]: the shape, the threshold,
/// the device, then the reduction of the root, the divisor and the data.
fn transform_admission(
    class: Class,
    a: &[Scalar],
    omega: &Scalar,
    log_n: u32,
    divisor: Option<&Scalar>,
) -> bool {
    if log_n == 0 || log_n > MAX_LOG_N || a.len() != 1usize << log_n {
        return account::decline(class, Reason::Shape);
    }
    let t = thresholds();
    let min = if class == Class::Ifft { t.ifft } else { t.fft };
    if log_n < min {
        return account::decline(class, Reason::BelowMinK);
    }
    if !device() {
        return account::decline(class, Reason::NoDevice);
    }
    if !field::is_reduced(omega)
        || divisor.is_some_and(|d| !field::is_reduced(d))
        || !field::all_reduced(a)
    {
        return account::decline(class, Reason::Unreduced);
    }
    true
}

/// The FFT of `a` (length `2^log_n`) at `omega`, natural order, in place:
/// the hook `GpuHooks::fft`. `false`: declined, `a` unchanged.
pub fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    guarded(Class::Fft, false, || fft_body(a, omega, log_n))
}

fn fft_body(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    if !transform_admission(Class::Fft, a, &omega, log_n, None) {
        return false;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ntt::fft(a, omega, log_n)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    false
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place: the hook `GpuHooks::ifft`. `false`: declined, `a` unchanged.
pub fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    guarded(Class::Ifft, false, || {
        ifft_body(a, omega_inv, log_n, divisor)
    })
}

fn ifft_body(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    if !transform_admission(Class::Ifft, a, &omega_inv, log_n, Some(&divisor)) {
        return false;
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ntt::ifft(a, omega_inv, log_n, divisor)
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    false
}

// ---------------------------------------------------------------------------
// Quotient

/// The quotient `h(X)` of a job (`midnight_proofs::gpu::abi::QuotientJobAbi`):
/// the hook `GpuHooks::quotient`, writing the first `out.len()` coefficients.
/// `false`: declined, `out` unchanged.
///
/// # Safety
/// `job` is null or points to a valid `QuotientJobAbi` whose pointers live
/// for the call (the prover passes such a job).
pub unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
    // SAFETY: the caller's contract, passed on.
    guarded(Class::Quotient, false, || unsafe {
        quotient_body(job, out)
    })
}

/// # Safety
/// As [`quotient`].
unsafe fn quotient_body(job: *const c_void, out: &mut [Scalar]) -> bool {
    if job.is_null() {
        return account::decline(Class::Quotient, Reason::Shape);
    }
    if !device() {
        return account::decline(Class::Quotient, Reason::NoDevice);
    }
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        // the test panic comes before the backend: `out` is not written yet
        test_switch::hook_panic(Class::Quotient);
        // SAFETY: the caller's contract.
        unsafe { quotient::quotient(job, out) }
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    {
        let _ = out;
        false
    }
}

// ---------------------------------------------------------------------------
// The hook boundary

/// The scope of one hook call: when no request is open ([`begin_request`])
/// at its start or at its end, its drop zeroes every per-request buffer (the
/// device scratch and the idle lendable buffers, which can hold witness
/// values) before the hook returns, on success, decline, error or a caught
/// panic. The wipe takes the device, so it never zeroes a buffer another
/// call is using (see `ctx::wipe_unbracketed`). Inside a request it does
/// nothing: the scratch is reused by the next call and zeroed by
/// [`end_request`].
struct HookScope {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    open: bool,
}

impl HookScope {
    fn enter() -> Self {
        HookScope {
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            open: ctx::request_open(),
        }
    }
}

impl Drop for HookScope {
    fn drop(&mut self) {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        if !self.open || !ctx::request_open() {
            ctx::wipe_unbracketed();
        }
    }
}

/// Runs the body of a hook: a panic inside it becomes a counted decline
/// `panic` of `class` with the value `declined` (the caller's buffer is
/// unchanged: no body writes it before its result is complete), and the
/// call's scope ([`HookScope`]) wipes the per-request buffers outside a
/// request, on every path.
fn guarded<R>(class: Class, declined: R, body: impl FnOnce() -> R) -> R {
    let _scope = HookScope::enter();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(r) => r,
        Err(_) => {
            test_switch::count_panic();
            account::decline(class, Reason::Panic);
            declined
        }
    }
}

/// Test switches of the hook entries (feature `test-support`, or this crate's
/// unit tests).
#[cfg(any(test, feature = "test-support"))]
mod test_switch {
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};

    use crate::account::Class;

    /// One bit per class ([`bit`]) whose backend panics.
    static HOOK_PANIC: AtomicU32 = AtomicU32::new(0);
    /// Hook calls whose backend panicked, since process start.
    static HOOK_PANICS: AtomicU64 = AtomicU64::new(0);

    fn bit(class: Class) -> u32 {
        match class {
            Class::Msm => 1,
            Class::Fft => 2,
            Class::Ifft => 4,
            Class::Quotient => 8,
        }
    }

    pub(super) fn set_hook_panic(class: &str, on: bool) {
        let b = [Class::Msm, Class::Fft, Class::Ifft, Class::Quotient]
            .into_iter()
            .find(|c| c.name() == class)
            .map_or(0, bit);
        if on {
            HOOK_PANIC.fetch_or(b, Relaxed);
        } else {
            HOOK_PANIC.fetch_and(!b, Relaxed);
        }
    }

    /// Panics when [`set_hook_panic`] is on for `class`.
    pub(crate) fn hook_panic(class: Class) {
        if HOOK_PANIC.load(Relaxed) & bit(class) != 0 {
            panic!(
                "midnight-metal test switch: a panic in the {} backend",
                class.name()
            );
        }
    }

    pub(super) fn count_panic() {
        HOOK_PANICS.fetch_add(1, Relaxed);
    }

    pub(super) fn hook_panics() -> u64 {
        HOOK_PANICS.load(Relaxed)
    }
}

/// The test switches of a normal build: no-ops.
#[cfg(not(any(test, feature = "test-support")))]
mod test_switch {
    use crate::account::Class;

    #[inline(always)]
    pub(crate) fn hook_panic(_: Class) {}

    #[inline(always)]
    pub(super) fn count_panic() {}
}

/// Makes every later call of the hook `class` (`msm`, `fft`, `ifft`,
/// `quotient`) panic inside its backend (`true`), or stops it: a test
/// switch of the `panic` decline (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_hook_panic(class: &str, on: bool) {
    test_switch::set_hook_panic(class, on);
}

/// Hook calls whose backend panicked since process start (each one also a
/// decline `panic` of its class; feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn hook_panics() -> u64 {
    test_switch::hook_panics()
}

// ---------------------------------------------------------------------------
// Requests and memory

/// Starts a request: the per-request buffers (device scratch that can hold
/// witness values) are then kept for reuse across the hook calls of the
/// request until [`end_request`] / [`end_request_report`] zeroes them. Also
/// resets the peaks of [`device_bytes`] and [`request_peak_bytes`]. Without
/// a request every hook call zeroes them before it returns (see "Requests
/// and wiping" in the crate text).
pub fn begin_request() {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::begin_request();
        quotient::begin_request();
    }
}

/// What [`end_request_report`] erased: the per-request buffers (regions) of
/// the request, zeroed before their release and, with `check`, read back,
/// together with (check mode) the idle buffers of the pool read back.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WipeStats {
    /// Regions zeroed (then pooled, or released when one failed its check):
    /// the per-request buffers, and in check mode the idle buffers of the
    /// pool read back (`pool_regions`).
    pub regions: u64,
    /// Their bytes.
    pub bytes: u64,
    /// Whether every region was read back after its zeroing.
    pub checked: bool,
    /// Regions read back (check mode).
    pub checked_regions: u64,
    /// Regions that read back non-zero after the zeroing (check mode).
    pub nonzero_regions: u64,
    /// Regions that could not be read back from the host (not in shared
    /// storage), so no result came back (check mode).
    pub check_failures: u64,
    /// Of `regions` (and `bytes`): the idle buffers of the pool of zeroed
    /// buffers read back (check mode). Scoped buffers that held witness
    /// values go back there after their zeroing.
    pub pool_regions: u64,
    /// Their bytes.
    pub pool_bytes: u64,
    /// Zeroings that panicked since the last report (on a background
    /// release thread or in a buffer's drop). Each one was completed on the
    /// calling thread and its buffer released, not pooled; the report is
    /// `Err` all the same.
    pub erasure_failures: u64,
}

/// Ends a request (call it between proofs): waits for any device work in
/// progress and for the zeroing still running in the background; with
/// `check` reads back every idle buffer of the pool of zeroed buffers; then
/// zeroes every per-request buffer, with `check` reads each one back from
/// the host and compares it with zero, and puts it back into the pool; then
/// frees the proving-key copies unless [`keep_resident`].
///
/// `Err`: a region read back non-zero or could not be read, or a zeroing
/// panicked (`erasure_failures`). A region (a per-request buffer, or an idle
/// buffer of the pool) that read back non-zero or could not be read back is
/// zeroed again, read back again and released: it never goes back into the
/// pool and is never kept as scratch. On other targets, or before the
/// context exists, there is no region.
pub fn end_request_report(check: bool) -> Result<WipeStats, WipeStats> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        let r = ctx::end_request_report(check);
        quotient::end_request();
        r
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    Ok(WipeStats {
        checked: check,
        ..WipeStats::default()
    })
}

/// [`end_request_report`] without the read-back: `true` when every
/// per-request buffer was zeroed (and pooled, or released above the pool's
/// limit) and no zeroing panicked.
pub fn end_request() -> bool {
    end_request_report(false).is_ok()
}

/// Releases the pool of zeroed buffers (see "Requests and wiping" in the
/// crate text), after the zeroing still running in the background has
/// ended. Returns the bytes released.
pub fn release_pool() -> u64 {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::release_pool()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    0
}

/// Bytes of the idle, zeroed buffers in the pool (kept for the next
/// request; part of [`device_bytes`]).
pub fn pool_bytes() -> u64 {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::pool_bytes()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    0
}

/// `(now, peak)` bytes this process holds in Metal buffers of this crate,
/// the idle buffers of the pool included ([`pool_bytes`]); the peak is since
/// process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::device_bytes()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    (0, 0)
}

/// The peak, since the last [`begin_request`], of the bytes in use: buffers
/// handed out (new or taken from the pool of zeroed buffers), the resident
/// SRS tables and the resident proving-key copies, without the idle buffers
/// of the pool. A buffer taken from the pool counts while it is in use.
/// [`begin_request`] resets it to the bytes in use then, so it includes what
/// earlier requests left resident, as the peak of the CUDA crate's
/// `device_bytes` does. Lock-free.
pub fn request_peak_bytes() -> u64 {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        ctx::request_peak_bytes()
    }
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    0
}
