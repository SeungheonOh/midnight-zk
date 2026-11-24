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

//! The entry points of the accelerator hooks, generic over the backend: the
//! admission, the hook boundary and the request bracket. A backend crate
//! exports them as plain functions (`fn` pointers for `GpuHooks`).
//!
//! # Hook contract
//!
//! - Every entry point returns the exact value of the stock CPU code, or
//!   declines (`None` / `false`) and the caller runs the CPU code.
//! - A decline leaves the caller's buffer unchanged: a result is written only
//!   after the last command buffer completed without error.
//! - No entry point draws randomness or touches a transcript. Any
//!   pseudo-random value (the MSM mask) is a fixed function of the shape.
//! - Every failure, including a panic caught at the entry point, is a counted
//!   decline ([`declines`]), never a panic into the prover.

use std::{
    ffi::c_void,
    marker::PhantomData,
    sync::atomic::{AtomicU32, Ordering::Relaxed},
};

use crate::{
    Backend, DeviceInfo, G1AffineLimbs, G1JacobianLimbs, MAX_LOG_N, MAX_MSM_LOG, Scalar, Stats,
    Thresholds, WipeStats,
    account::{self, Class, Reason},
    ctx, field, lg2_nearest, msm, ntt,
    quotient::{self as quot},
};

// ---------------------------------------------------------------------------
// Device

/// Whether `{ENV_PREFIX}=off` (read once).
fn disabled<B: Backend>() -> bool {
    *B::host()
        .off
        .get_or_init(|| std::env::var(B::ENV_PREFIX).as_deref() == Ok("off"))
}

/// The device, without making the context. `None` with no device or with
/// `{ENV_PREFIX}=off`.
pub fn present<B: Backend>() -> Option<&'static DeviceInfo> {
    if disabled::<B>() {
        return None;
    }
    B::probe()
}

/// Whether device calls can run: [`present`], and the context is made; the
/// first call makes it.
pub fn device<B: Backend>() -> bool {
    present::<B>().is_some() && B::context().is_ok()
}

/// `(free, total)` device bytes ([`ctx::mem_info`]); `None` without a device.
pub fn mem_info<B: Backend>() -> Option<(usize, usize)> {
    if !device::<B>() {
        return None;
    }
    ctx::mem_info::<B>()
}

// ---------------------------------------------------------------------------
// Thresholds

/// The explicit threshold: not read yet (the environment is read at the next
/// use).
const MIN_K_UNREAD: u32 = u32::MAX;
/// The explicit threshold: none (the per-class defaults apply).
const MIN_K_DEFAULTS: u32 = u32::MAX - 1;

/// The explicit threshold of a backend, or one of the two markers.
pub(crate) const fn min_k_unread() -> AtomicU32 {
    AtomicU32::new(MIN_K_UNREAD)
}

/// [`set_min_k`], else `{ENV_PREFIX}_MIN_K` (ignored above [`MAX_LOG_N`]).
fn explicit_min_k<B: Backend>() -> Option<u32> {
    let min_k = &B::host().min_k;
    let mut k = min_k.load(Relaxed);
    if k == MIN_K_UNREAD {
        let env = std::env::var(format!("{}_MIN_K", B::ENV_PREFIX))
            .ok()
            .and_then(|s| s.trim().parse().ok())
            .filter(|k| *k <= MAX_LOG_N)
            .unwrap_or(MIN_K_DEFAULTS);
        let _ = min_k.compare_exchange(MIN_K_UNREAD, env, Relaxed, Relaxed);
        k = min_k.load(Relaxed);
    }
    (k <= MAX_LOG_N).then_some(k)
}

/// The effective threshold of each class: the explicit one, else the
/// backend's defaults ([`crate::Tuning::thresholds`]).
pub fn thresholds<B: Backend>() -> Thresholds {
    explicit_min_k::<B>().map_or(B::TUNING.thresholds, Thresholds::all)
}

/// The explicit threshold, else the smallest per-class default.
pub fn min_k<B: Backend>() -> u32 {
    let t = thresholds::<B>();
    let m = t.fft.min(t.ifft).min(t.msm_table).min(t.quotient);
    t.msm_one_shot.map_or(m, |k| m.min(k))
}

/// Sets one threshold for every class and opens the one-shot MSM. Returns
/// `false` and keeps the old value when `k` is above [`MAX_LOG_N`].
pub fn set_min_k<B: Backend>(k: u32) -> bool {
    if k > MAX_LOG_N {
        return false;
    }
    B::host().min_k.store(k, Relaxed);
    true
}

/// Undoes [`set_min_k`].
pub fn reset_min_k<B: Backend>() {
    B::host().min_k.store(MIN_K_UNREAD, Relaxed);
}

// ---------------------------------------------------------------------------
// Counters

/// The service counters since the last [`reset_stats`].
pub fn stats<B: Backend>() -> Stats {
    Stats {
        msm: account::class_stats::<B>(Class::Msm),
        fft: account::class_stats::<B>(Class::Fft),
        ifft: account::class_stats::<B>(Class::Ifft),
        quotient: account::class_stats::<B>(Class::Quotient),
    }
}

/// Every decline since the last [`reset_stats`]: `(class, reason, count)`.
///
/// Reasons:
/// - `no-device`: no usable device, or `{ENV_PREFIX}=off`;
/// - `below-min-k`: below the threshold or the MSM's effective-size need;
/// - `shape`: a length, root, job layout or version the device does not take;
/// - `unreduced`: a scalar not below the modulus;
/// - `memory`: a buffer too large, a failed allocation or a budget;
/// - `device-error`: a pipeline, encoder or command buffer failed;
/// - `panic`: the backend panicked and the entry point caught it;
/// - `bases-not-in-subgroup` (`msm`): a base outside G1;
/// - `unregistered-bases` (`msm`): no registered table serves the bases and
///   the one-shot path is off.
pub fn declines<B: Backend>() -> Vec<(&'static str, &'static str, u64)> {
    account::declines::<B>()
}

/// The host time around the device work since the last [`reset_stats`].
pub fn host_times<B: Backend>() -> crate::HostTimes {
    account::host_times::<B>()
}

/// Resets [`stats`], [`declines`] and [`host_times`].
pub fn reset_stats<B: Backend>() {
    account::reset::<B>();
}

// ---------------------------------------------------------------------------
// MSM

/// The length admission of an MSM: `shape` for an empty input, unequal
/// lengths or a length above `2^max_log`; `below-min-k` when the size class
/// is below `msm_table`. [`msm_g1`] may still decline on the bases.
pub fn msm_admission<B: Backend>(
    scalars: usize,
    bases: usize,
    max_log: u32,
) -> Result<(), &'static str> {
    length_admission::<B>(scalars, bases, max_log).map_err(Reason::name)
}

/// [`msm_admission`] with the reason as a [`Reason`].
fn length_admission<B: Backend>(scalars: usize, bases: usize, max_log: u32) -> Result<(), Reason> {
    if scalars == 0 || scalars != bases || scalars > (1usize << max_log.min(62)) {
        return Err(Reason::Shape);
    }
    if lg2_nearest(scalars) < thresholds::<B>().msm_table {
        return Err(Reason::BelowMinK);
    }
    Ok(())
}

/// Admission of an MSM no registered table serves; `Ok` is the threshold.
fn one_shot_admission(len: usize, t: &Thresholds) -> Result<u32, Reason> {
    match t.msm_one_shot {
        None => Err(Reason::UnregisteredBases),
        Some(k) if lg2_nearest(len) < k => Err(Reason::BelowMinK),
        Some(k) => Ok(k),
    }
}

/// Whether [`msm_g1`] takes an MSM of `len` scalars over `bases`, judged on
/// the length and the bases only. It reads no scalar, so the answer does not
/// depend on the witness; [`msm_g1`] may still decline on the scalars. It does
/// not check the device.
pub fn msm_sized<B: Backend>(len: usize, bases: &[G1AffineLimbs]) -> bool {
    length_admission::<B>(len, bases.len(), MAX_MSM_LOG).is_ok() && sized_by_bases::<B>(len, bases)
}

/// [`msm_sized`] of a length that passed [`length_admission`].
fn sized_by_bases<B: Backend>(len: usize, bases: &[G1AffineLimbs]) -> bool {
    sized_by_lookup(&msm::lookup::<B>(bases, len), len, &thresholds::<B>())
}

/// [`msm_sized`] from the lookup of the bases.
fn sized_by_lookup<B: Backend>(lookup: &msm::Lookup<B>, len: usize, t: &Thresholds) -> bool {
    match lookup {
        msm::Lookup::Table(_) => true,
        msm::Lookup::NotInG1 => false,
        msm::Lookup::Unregistered => one_shot_admission(len, t).is_ok(),
    }
}

/// G1 MSM `sum scalars[i] * bases[i]`: the hook `GpuHooks::msm_g1`. `None`:
/// declined.
///
/// Every check on public facts (lengths, device, registered table, one-shot
/// admission) runs before any scalar is read. Then come the effective size
/// and reduction of the scalars and, on the one-shot path only, the G1 check
/// of every base.
pub fn msm_g1<B: Backend>(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    guarded::<B, _>(Class::Msm, None, || msm_g1_body::<B>(scalars, bases, None))
}

/// [`msm_g1`] and [`msm_sized`] of the same call from one registry lookup, so
/// the bases are compared once per MSM. `sized` is decided before any scalar
/// is read and stands even if the MSM declines or panics.
pub fn msm_g1_sized<B: Backend>(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
) -> (bool, Option<G1JacobianLimbs>) {
    let mut sized = false;
    let r = guarded::<B, _>(Class::Msm, None, || {
        msm_g1_body::<B>(scalars, bases, Some(&mut sized))
    });
    (sized, r)
}

/// The body of [`msm_g1`]; with `sized`, also stores [`msm_sized`].
fn msm_g1_body<B: Backend>(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    sized: Option<&mut bool>,
) -> Option<G1JacobianLimbs> {
    if let Err(r) = length_admission::<B>(scalars.len(), bases.len(), MAX_MSM_LOG) {
        account::decline::<B>(Class::Msm, r);
        return None;
    }
    if !device::<B>() {
        // `msm_sized` does not look at the device
        if let Some(s) = sized {
            *s = sized_by_bases::<B>(scalars.len(), bases);
        }
        account::decline::<B>(Class::Msm, Reason::NoDevice);
        return None;
    }
    let t = thresholds::<B>();
    let n = scalars.len();
    // pick the path from public facts only, before any scalar is read
    let lookup = msm::lookup::<B>(bases, n);
    if let Some(s) = sized {
        *s = sized_by_lookup(&lookup, n, &t);
    }
    let tuning = &B::TUNING;
    let (table, need) = match lookup {
        msm::Lookup::Table(tb) => (
            Some(tb),
            msm::effective_need(n, t.msm_table, true, t.explicit, tuning),
        ),
        msm::Lookup::NotInG1 => {
            account::decline::<B>(Class::Msm, msm::BASES_REASON);
            return None;
        }
        msm::Lookup::Unregistered => match one_shot_admission(n, &t) {
            Ok(k) => (None, msm::effective_need(n, k, false, t.explicit, tuning)),
            Err(r) => {
                account::decline::<B>(Class::Msm, r);
                return None;
            }
        },
    };
    // only the decline is observable, never the count
    if !msm::effective_size_reaches(scalars, need) {
        account::decline::<B>(Class::Msm, Reason::BelowMinK);
        return None;
    }
    if !field::all_reduced(scalars) {
        account::decline::<B>(Class::Msm, Reason::Unreduced);
        return None;
    }
    let r = msm::msm_g1::<B>(scalars, bases, table);
    // the test panic comes after the backend used its scratch
    test_switch::hook_panic::<B>(Class::Msm);
    r
}

// ---------------------------------------------------------------------------
// Transforms

/// The admission shared by [`fft`] and [`ifft`]: the shape, the threshold,
/// the device, then the reduction of the root, the divisor and the data.
fn transform_admission<B: Backend>(
    class: Class,
    a: &[Scalar],
    omega: &Scalar,
    log_n: u32,
    divisor: Option<&Scalar>,
) -> bool {
    if log_n == 0 || log_n > MAX_LOG_N || a.len() != 1usize << log_n {
        return account::decline::<B>(class, Reason::Shape);
    }
    let t = thresholds::<B>();
    let min = if class == Class::Ifft { t.ifft } else { t.fft };
    if log_n < min {
        return account::decline::<B>(class, Reason::BelowMinK);
    }
    if !device::<B>() {
        return account::decline::<B>(class, Reason::NoDevice);
    }
    if !field::is_reduced(omega)
        || divisor.is_some_and(|d| !field::is_reduced(d))
        || !field::all_reduced(a)
    {
        return account::decline::<B>(class, Reason::Unreduced);
    }
    true
}

/// The FFT of `a` (length `2^log_n`) at `omega`, natural order, in place:
/// the hook `GpuHooks::fft`. `false`: declined, `a` unchanged.
pub fn fft<B: Backend>(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    guarded::<B, _>(Class::Fft, false, || {
        transform_admission::<B>(Class::Fft, a, &omega, log_n, None)
            && ntt::fft::<B>(a, omega, log_n)
    })
}

/// The FFT of `a` at `omega_inv`, then each element times `divisor`, in
/// place: the hook `GpuHooks::ifft`. `false`: declined, `a` unchanged.
pub fn ifft<B: Backend>(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    guarded::<B, _>(Class::Ifft, false, || {
        transform_admission::<B>(Class::Ifft, a, &omega_inv, log_n, Some(&divisor))
            && ntt::ifft::<B>(a, omega_inv, log_n, divisor)
    })
}

// ---------------------------------------------------------------------------
// Quotient

/// The quotient `h(X)` of a job (`midnight_proofs::gpu::abi::QuotientJobAbi`):
/// the hook `GpuHooks::quotient`, writing the first `out.len()` coefficients.
/// `false`: declined, `out` unchanged.
///
/// # Safety
/// `job` is null or points to a valid `QuotientJobAbi` whose pointers live
/// for the call.
pub unsafe fn quotient<B: Backend>(job: *const c_void, out: &mut [Scalar]) -> bool {
    // SAFETY: the caller's contract, passed on.
    guarded::<B, _>(Class::Quotient, false, || unsafe {
        quotient_body::<B>(job, out)
    })
}

/// # Safety
/// As [`quotient`].
unsafe fn quotient_body<B: Backend>(job: *const c_void, out: &mut [Scalar]) -> bool {
    if job.is_null() {
        return account::decline::<B>(Class::Quotient, Reason::Shape);
    }
    if !device::<B>() {
        return account::decline::<B>(Class::Quotient, Reason::NoDevice);
    }
    // the test panic comes before the backend: `out` is not written yet
    test_switch::hook_panic::<B>(Class::Quotient);
    // SAFETY: the caller's contract.
    unsafe { quot::quotient::<B>(job, out) }
}

// ---------------------------------------------------------------------------
// The hook boundary

/// Outside a request, zeroes the per-request buffers when the hook call ends,
/// on every path. The wipe takes the device, so it never zeroes a buffer
/// another call is using.
struct HookScope<B: Backend> {
    open: bool,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> HookScope<B> {
    fn enter() -> Self {
        HookScope {
            open: ctx::request_open::<B>(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Drop for HookScope<B> {
    fn drop(&mut self) {
        if !self.open || !ctx::request_open::<B>() {
            ctx::wipe_unbracketed::<B>();
        }
    }
}

/// Runs a hook body; a panic becomes the decline `panic` with the value
/// `declined`. The caller's buffer is intact because no body writes it before
/// its result is complete.
fn guarded<B: Backend, R>(class: Class, declined: R, body: impl FnOnce() -> R) -> R {
    let _scope = HookScope::<B>::enter();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(r) => r,
        Err(_) => {
            test_switch::count_panic::<B>();
            account::decline::<B>(class, Reason::Panic);
            declined
        }
    }
}

/// The hook test switches of one backend.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug)]
pub(crate) struct HookSwitches {
    /// One bit per class ([`test_switch::bit`]) whose backend panics.
    hook_panic: AtomicU32,
    /// Hook calls whose backend panicked, since process start.
    hook_panics: std::sync::atomic::AtomicU64,
}

#[cfg(any(test, feature = "test-support"))]
impl HookSwitches {
    /// Every switch off.
    pub(crate) const fn new() -> Self {
        HookSwitches {
            hook_panic: AtomicU32::new(0),
            hook_panics: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// Test switches of the hook entries.
#[cfg(any(test, feature = "test-support"))]
pub(crate) mod test_switch {
    use std::sync::atomic::Ordering::Relaxed;

    use crate::{Backend, account::Class};

    fn bit(class: Class) -> u32 {
        match class {
            Class::Msm => 1,
            Class::Fft => 2,
            Class::Ifft => 4,
            Class::Quotient => 8,
        }
    }

    pub(super) fn set_hook_panic<B: Backend>(class: &str, on: bool) {
        let b = [Class::Msm, Class::Fft, Class::Ifft, Class::Quotient]
            .into_iter()
            .find(|c| c.name() == class)
            .map_or(0, bit);
        let s = &B::host().hooks.hook_panic;
        if on {
            s.fetch_or(b, Relaxed);
        } else {
            s.fetch_and(!b, Relaxed);
        }
    }

    /// Panics when [`set_hook_panic`] is on for `class`.
    pub(crate) fn hook_panic<B: Backend>(class: Class) {
        if B::host().hooks.hook_panic.load(Relaxed) & bit(class) != 0 {
            panic!(
                "midnight-{} test switch: a panic in the {} backend",
                B::NAME,
                class.name()
            );
        }
    }

    pub(super) fn count_panic<B: Backend>() {
        B::host().hooks.hook_panics.fetch_add(1, Relaxed);
    }

    pub(super) fn hook_panics<B: Backend>() -> u64 {
        B::host().hooks.hook_panics.load(Relaxed)
    }
}

/// The test switches of a normal build: no-ops.
#[cfg(not(any(test, feature = "test-support")))]
pub(crate) mod test_switch {
    use crate::{Backend, account::Class};

    #[inline(always)]
    pub(crate) fn hook_panic<B: Backend>(_: Class) {}

    #[inline(always)]
    pub(super) fn count_panic<B: Backend>() {}
}

/// Makes the hook `class` panic inside its backend (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_hook_panic<B: Backend>(class: &str, on: bool) {
    test_switch::set_hook_panic::<B>(class, on);
}

/// Hook calls whose backend panicked since process start.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn hook_panics<B: Backend>() -> u64 {
    test_switch::hook_panics::<B>()
}

// ---------------------------------------------------------------------------
// Requests and memory

/// Starts a request: per-request buffers are kept for reuse until
/// [`end_request`]. Also resets the peaks of [`device_bytes`] and
/// [`request_peak_bytes`].
pub fn begin_request<B: Backend>() {
    ctx::begin_request::<B>();
    quot::begin_request::<B>();
}

/// Ends a request: waits for device work and background zeroing, zeroes the
/// per-request buffers and returns them to the pool, then frees the
/// proving-key copies unless [`crate::keep_resident`]. With `check`, every
/// zeroed region and every idle pool buffer is read back.
///
/// `Err` when a region read back non-zero or unreadable, or a zeroing
/// panicked. Such a region is zeroed again and released, never pooled.
pub fn end_request_report<B: Backend>(check: bool) -> Result<WipeStats, WipeStats> {
    let r = ctx::end_request_report::<B>(check);
    quot::end_request::<B>();
    r
}

/// [`end_request_report`] without the read-back; `true` on success.
pub fn end_request<B: Backend>() -> bool {
    end_request_report::<B>(false).is_ok()
}

/// Releases the pool of zeroed buffers once background zeroing has ended.
/// Returns the bytes released.
pub fn release_pool<B: Backend>() -> u64 {
    ctx::release_pool::<B>()
}

/// Bytes of idle, zeroed buffers in the pool.
pub fn pool_bytes<B: Backend>() -> u64 {
    ctx::pool_bytes::<B>()
}

/// `(now, peak)` bytes held in the backend's buffers, idle pool buffers
/// included. The peak is since process start or the last [`begin_request`].
pub fn device_bytes<B: Backend>() -> (u64, u64) {
    ctx::device_bytes::<B>()
}

/// Peak bytes in use since the last [`begin_request`]: buffers handed out,
/// resident SRS tables and key copies, not idle pool buffers. It starts from
/// the bytes in use at [`begin_request`], so it includes what earlier requests
/// left resident.
pub fn request_peak_bytes<B: Backend>() -> u64 {
    ctx::request_peak_bytes::<B>()
}
