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

//! Metal backend for the accelerator hooks of `midnight-proofs` (feature
//! `gpu`) on Apple GPUs: BLS12-381 G1 MSM, scalar-field NTT and the quotient.
//! It has the same entry points as the CUDA crate `midnight-gpu` and does not
//! depend on `midnight-proofs`: its functions match the hook fields of
//! `midnight_proofs::gpu::GpuHooks`.
//!
//! The host side is `midnight-accel`, generic over a GPU backend; this crate
//! is its Metal backend ([`ctx::Metal`]) and the entry points for it.
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
//! assert!(midnight_metal::end_request());
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
//!   decline ([`declines`]), never a panic into the prover.
//!
//! # Requests
//!
//! Per-request device buffers can hold witness values. Between
//! [`begin_request`] and [`end_request`] they are reused across hook calls;
//! [`end_request`] zeroes them. Without an open request every hook call zeroes
//! them before it returns. Large buffers come from a pool of zeroed buffers
//! because a new buffer stalls its first command buffer while its pages are
//! mapped for the GPU.
//!
//! # Thresholds
//!
//! A call below its class's threshold ([`thresholds`]) declines `below-min-k`.
//! The defaults ([`DEFAULT_THRESHOLDS`]) were measured on whole proofs, not
//! isolated kernels, because the GPU clocks down while the CPU works between
//! device calls. By default only MSMs on a registered SRS table
//! ([`register_srs`]) go to the device: other bases need a per-call G1 check
//! that costs more than the CPU MSM. An explicit threshold ([`set_min_k`])
//! replaces every default and opens that one-shot path.
//!
//! # Environment
//!
//! - `MIDNIGHT_METAL=off`: use no device.
//! - `MIDNIGHT_METAL_MIN_K`: one threshold for every class.
//! - `MIDNIGHT_METAL_MSM_CPU_SHARE=off`: run every MSM on the device only.
//! - `MIDNIGHT_METAL_POOL_MIB`: pool size limit (default a quarter of
//!   `recommendedMaxWorkingSetSize`).
//! - `MIDNIGHT_METAL_QUOTIENT_THREADS`: see [`quotient_threads`].
//! - `MIDNIGHT_METAL_KEEP_KEYS=1`: see [`keep_resident`].
//! - `MIDNIGHT_METAL_SHADER_SOURCE=1`: compile the embedded shader source
//!   instead of the metallib ([`library_source`]); also read by `build.rs`.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

use std::ffi::c_void;

use midnight_accel as accel;
pub use midnight_accel::{
    ClassStats, G1AffineLimbs, G1JacobianLimbs, MAX_LOG_N, MAX_MSM_LOG, Scalar, Stats, Thresholds,
    WipeStats, field,
};

#[cfg(target_os = "macos")]
pub mod ctx;
/// The Metal backend of `midnight-accel`, for code generic over the backend.
#[cfg(target_os = "macos")]
pub use ctx::Metal;
#[cfg(not(target_os = "macos"))]
mod stub;
/// The Metal backend of `midnight-accel`, for code generic over the backend:
/// a type without values on this target.
#[cfg(not(target_os = "macos"))]
pub use stub::Metal;
mod tuning;

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

/// Per-class break-evens against the stock CPU code, measured on Apple
/// silicon. The one-shot MSM never breaks even because of its G1 check.
pub const DEFAULT_THRESHOLDS: Thresholds = tuning::TUNING.thresholds;

// ---------------------------------------------------------------------------
// Device

/// The Metal device as its properties describe it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalInfo {
    /// Device name (`MTLDevice.name`).
    pub name: String,
    /// `MTLDevice.registryID`.
    pub registry_id: u64,
    /// `hasUnifiedMemory`.
    pub unified_memory: bool,
    /// `recommendedMaxWorkingSetSize` in bytes.
    pub recommended_max_working_set: u64,
    /// `maxBufferLength`: the largest single buffer in bytes.
    pub max_buffer_length: u64,
    /// `maxThreadsPerThreadgroup.width`.
    pub max_threads_per_threadgroup: u64,
    /// The highest `MTLGPUFamilyAppleN` supported (0: none).
    pub apple_family: u32,
    /// Supports `MTLGPUFamilyMetal3`.
    pub metal3: bool,
    /// The Metal Shading Language version of the shaders.
    pub msl_version: &'static str,
}

/// Whether the Metal code is compiled in (a macOS target).
pub const fn metal_built() -> bool {
    cfg!(target_os = "macos")
}

/// The Metal device, without loading the shader library. `None` on other
/// targets, with no device, or with `MIDNIGHT_METAL=off`.
pub fn present() -> Option<&'static MetalInfo> {
    accel::present::<Metal>()?;
    #[cfg(target_os = "macos")]
    {
        ctx::probe()
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// Whether device calls can run: [`present`], and the context (queue, shader
/// library) is made; the first call makes it.
pub fn device() -> bool {
    accel::device::<Metal>()
}

/// Where the shader library came from: `metallib`, or `source ...` with the
/// reason. `None` before the context is made or when it failed.
pub fn library_source() -> Option<&'static str> {
    #[cfg(target_os = "macos")]
    {
        ctx::context().ok().map(|c| c.library_source())
    }
    #[cfg(not(target_os = "macos"))]
    None
}

/// `(free, total)` device bytes. `total` is `recommendedMaxWorkingSetSize`;
/// `free` is `total` minus `currentAllocatedSize` plus the idle pool buffers,
/// which are released on demand. On unified memory this is a budget hint, not
/// a reservation.
pub fn mem_info() -> Option<(usize, usize)> {
    accel::mem_info::<Metal>()
}

// ---------------------------------------------------------------------------
// Thresholds and counters

/// The effective threshold of each class.
pub fn thresholds() -> Thresholds {
    accel::thresholds::<Metal>()
}

/// The explicit threshold, else the smallest per-class default.
pub fn min_k() -> u32 {
    accel::min_k::<Metal>()
}

/// Sets one threshold for every class and opens the one-shot MSM. Returns
/// `false` and keeps the old value when `k` is above [`MAX_LOG_N`].
pub fn set_min_k(k: u32) -> bool {
    accel::set_min_k::<Metal>(k)
}

/// Undoes [`set_min_k`].
pub fn reset_min_k() {
    accel::reset_min_k::<Metal>()
}

/// The service counters since the last [`reset_stats`].
pub fn stats() -> Stats {
    accel::stats::<Metal>()
}

/// Every decline since the last [`reset_stats`]: `(class, reason, count)`.
/// The reasons are those of [`midnight_accel::declines`]; `no-device` also
/// covers another target and `MIDNIGHT_METAL=off`.
pub fn declines() -> Vec<(&'static str, &'static str, u64)> {
    accel::declines::<Metal>()
}

/// Resets [`stats`] and [`declines`].
pub fn reset_stats() {
    accel::reset_stats::<Metal>()
}

// ---------------------------------------------------------------------------
// MSM

/// The length admission of an MSM: `shape` for an empty input, unequal
/// lengths or a length above `2^max_log`; `below-min-k` when the size class
/// is below `msm_table`. [`msm_g1`] may still decline on the bases.
pub fn msm_admission(scalars: usize, bases: usize, max_log: u32) -> Result<(), &'static str> {
    accel::msm_admission::<Metal>(scalars, bases, max_log)
}

/// Whether [`msm_g1`] takes an MSM of `len` scalars over `bases`, judged on
/// the length and the bases only. It reads no scalar, so the answer does not
/// depend on the witness; [`msm_g1`] may still decline on the scalars. It does
/// not check the device.
pub fn msm_sized(len: usize, bases: &[G1AffineLimbs]) -> bool {
    accel::msm_sized::<Metal>(len, bases)
}

/// G1 MSM `sum scalars[i] * bases[i]`: the hook `GpuHooks::msm_g1`. `None`:
/// declined.
///
/// Every check on public facts (lengths, device, registered table, one-shot
/// admission) runs before any scalar is read. Then come the effective size
/// and reduction of the scalars and, on the one-shot path only, the G1 check
/// of every base.
pub fn msm_g1(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    accel::msm_g1::<Metal>(scalars, bases)
}

/// [`msm_g1`] and [`msm_sized`] of the same call from one registry lookup, so
/// the bases are compared once per MSM. `sized` is decided before any scalar
/// is read and stands even if the MSM declines or panics.
pub fn msm_g1_sized(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> (bool, Option<G1JacobianLimbs>) {
    accel::msm_g1_sized::<Metal>(scalars, bases)
}

/// [`msm_g1`] without the thresholds, reporting the path and plan. Not
/// counted in [`stats`] or [`declines`]; not a stable interface.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn msm_g1_stages(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> Option<(G1JacobianLimbs, MsmStages)> {
    accel::msm_g1_stages::<Metal>(scalars, bases)
}

/// Per-call G1 checks started on the one-shot path since process start (not
/// a stable interface).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn one_shot_g1_checks() -> u64 {
    accel::one_shot_g1_checks::<Metal>()
}

/// The registration of an SRS base slice. While it lives, the registrant
/// keeps the slice alive and unchanged. Dropping it frees the table.
pub type SrsHandle = accel::SrsHandle<Metal>;

/// Registers a host copy of an SRS base slice; the first MSM on it uploads
/// it. Registering the same slice again replaces the table. The KZG prover
/// commits from four `ParamsKZG` base vectors, so register all four.
///
/// Every point is checked to be in G1 once here. A table with a point outside
/// G1 is still registered, and MSMs on a prefix that includes it decline.
pub fn register_srs(bases: &[G1AffineLimbs]) -> SrsHandle {
    accel::register_srs::<Metal>(bases)
}

/// Live registered tables.
pub fn srs_handles() -> usize {
    accel::srs_handles::<Metal>()
}

/// Tables dropped at an MSM because the caller's slice no longer matched.
pub fn srs_stale() -> u64 {
    accel::srs_stale::<Metal>()
}

/// (MSMs served from a registered table, one-shot MSMs, table uploads) since
/// process start.
pub fn msm_counts() -> (u64, u64, u64) {
    accel::msm_counts::<Metal>()
}

// ---------------------------------------------------------------------------
// Transforms and the quotient

/// The FFT of `a` (length `2^log_n`) at `omega`, natural order, in place:
/// the hook `GpuHooks::fft`. `false`: declined, `a` unchanged.
pub fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    accel::fft::<Metal>(a, omega, log_n)
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place: the hook `GpuHooks::ifft`. `false`: declined, `a` unchanged.
pub fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    accel::ifft::<Metal>(a, omega_inv, log_n, divisor)
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
    unsafe { accel::quotient::<Metal>(job, out) }
}

/// The registration of proving-key polynomials. While it lives the registrant
/// keeps them alive and, within a request, unchanged. Dropping it frees the
/// device copies.
pub type KeyHandle = accel::KeyHandle<Metal>;

/// Registers proving-key polynomials by their digests. A polynomial
/// registered again moves to the new handle. Each length has its own device
/// copy, and a quotient serves from the copy of its extended length only.
pub fn register_key(polys: &[&[Scalar]]) -> KeyHandle {
    accel::register_key::<Metal>(polys)
}

/// Frees every device copy of proving-key polynomials (registrations stay).
/// Returns the bytes freed; a copy lent to a running quotient is freed when
/// returned and not counted.
pub fn evict_key_copies() -> u64 {
    accel::evict_key_copies::<Metal>()
}

/// Whether the device copies of registered proving-key polynomials stay after
/// a request (default `false`, or `MIDNIGHT_METAL_KEEP_KEYS=1`).
pub fn keep_resident() -> bool {
    accel::keep_resident::<Metal>()
}

/// Sets [`keep_resident`].
pub fn set_keep_resident(on: bool) {
    accel::set_keep_resident::<Metal>(on)
}

/// Registered polynomials now.
pub fn keys_registered() -> usize {
    accel::keys_registered::<Metal>()
}

/// Registered polynomials dropped at a quotient because their content
/// changed, since process start.
pub fn key_stale() -> u64 {
    accel::key_stale::<Metal>()
}

/// An upper bound of the device bytes one quotient job allocates. The
/// arguments are the fields of `midnight_proofs::gpu::QuotientShape`.
pub fn need_bytes(ext_k: u32, proof_polys: usize, key_polys: usize, nslots: u32) -> usize {
    accel::need_bytes::<Metal>(ext_k, proof_polys, key_polys, nslots)
}

/// (quotients computed on the device, quotient jobs declined) since process
/// start.
pub fn quotient_counts() -> (u64, u64) {
    accel::quotient_counts::<Metal>()
}

/// The threads of one numerator dispatch, which is also the scratch slot
/// stride: `MIDNIGHT_METAL_QUOTIENT_THREADS`, else 2^16; a multiple of 256 in
/// `[256, 2^24]`.
pub fn quotient_threads() -> u32 {
    accel::quotient_threads::<Metal>()
}

/// NTT building blocks (not a stable interface).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod ntt_test_support {
    use midnight_accel::ntt::test_support as t;

    use crate::{Metal, Scalar};

    /// `coeff_to_extended` on the device.
    pub fn coeff_to_extended(
        coeffs: &[Scalar],
        k: u32,
        ext_k: u32,
        zeta: [Scalar; 2],
    ) -> Result<Vec<Scalar>, String> {
        t::coeff_to_extended::<Metal>(coeffs, k, ext_k, zeta)
    }

    /// `extended_to_coeff` on the device.
    pub fn extended_to_coeff(
        values: &[Scalar],
        ext_k: u32,
        divisor: Scalar,
        zeta_out: [Scalar; 2],
    ) -> Result<Vec<Scalar>, String> {
        t::extended_to_coeff::<Metal>(values, ext_k, divisor, zeta_out)
    }

    /// The NTT of a copy of `values`.
    pub fn ntt(
        values: &[Scalar],
        log_n: u32,
        inverse: bool,
        scale: Option<Scalar>,
    ) -> Result<Vec<Scalar>, String> {
        t::ntt::<Metal>(values, log_n, inverse, scale)
    }

    /// The elementwise blocks on a copy of `values`.
    pub fn elementwise(
        values: &[Scalar],
        zeta: [Scalar; 2],
        c: Scalar,
        table: &[Scalar],
    ) -> Result<Vec<Scalar>, String> {
        t::elementwise::<Metal>(values, zeta, c, table)
    }

    pub use t::{passes, root};
}

/// Quotient test entry points (not a stable interface).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod quotient_test_support {
    use std::ffi::c_void;

    use midnight_accel::quotient::test_support as t;

    use crate::{Metal, QuotientStages, Scalar};

    /// The exact memory need of `job` now.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn need_bytes_job(job: *const c_void) -> Result<usize, &'static str> {
        // SAFETY: the caller's contract.
        unsafe { t::need_bytes_job::<Metal>(job) }
    }

    /// The placement counts of the last quotient served.
    pub fn last_stages() -> Option<QuotientStages> {
        t::last_stages::<Metal>()
    }

    /// A device-memory limit for the admission.
    pub fn set_memory_limit(limit: Option<u64>) {
        t::set_memory_limit::<Metal>(limit)
    }

    /// The numerator's target GPU time per command buffer.
    pub fn set_target_nanos(ns: Option<u64>) {
        t::set_target_nanos::<Metal>(ns)
    }

    /// The dispatch threads of the numerator.
    pub fn set_threads(threads: Option<u32>) {
        t::set_threads::<Metal>(threads)
    }

    /// The numerator of a job, read back.
    ///
    /// # Safety
    /// As [`crate::quotient`].
    pub unsafe fn numerator(job: *const c_void) -> Result<Vec<Scalar>, &'static str> {
        // SAFETY: the caller's contract.
        unsafe { t::numerator::<Metal>(job) }
    }
}

// ---------------------------------------------------------------------------
// Test switches

/// Makes the hook `class` panic inside its backend (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_hook_panic(class: &str, on: bool) {
    accel::set_hook_panic::<Metal>(class, on)
}

/// Hook calls whose backend panicked since process start.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn hook_panics() -> u64 {
    accel::hook_panics::<Metal>()
}

// ---------------------------------------------------------------------------
// Requests and memory

/// Starts a request: per-request buffers are kept for reuse until
/// [`end_request`]. Also resets the peaks of [`device_bytes`] and
/// [`request_peak_bytes`].
pub fn begin_request() {
    accel::begin_request::<Metal>()
}

/// Ends a request: waits for device work and background zeroing, zeroes the
/// per-request buffers and returns them to the pool, then frees the
/// proving-key copies unless [`keep_resident`]. With `check`, every zeroed
/// region and every idle pool buffer is read back.
///
/// `Err` when a region read back non-zero or unreadable, or a zeroing
/// panicked. Such a region is zeroed again and released, never pooled.
pub fn end_request_report(check: bool) -> Result<WipeStats, WipeStats> {
    accel::end_request_report::<Metal>(check)
}

/// [`end_request_report`] without the read-back; `true` on success.
pub fn end_request() -> bool {
    accel::end_request::<Metal>()
}

/// Releases the pool of zeroed buffers once background zeroing has ended.
/// Returns the bytes released.
pub fn release_pool() -> u64 {
    accel::release_pool::<Metal>()
}

/// Bytes of idle, zeroed buffers in the pool.
pub fn pool_bytes() -> u64 {
    accel::pool_bytes::<Metal>()
}

/// `(now, peak)` bytes held in this crate's Metal buffers, idle pool buffers
/// included. The peak is since process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    accel::device_bytes::<Metal>()
}

/// Peak bytes in use since the last [`begin_request`]: buffers handed out,
/// resident SRS tables and key copies, not idle pool buffers. It starts from
/// the bytes in use at [`begin_request`], so it includes what earlier requests
/// left resident.
pub fn request_peak_bytes() -> u64 {
    accel::request_peak_bytes::<Metal>()
}
