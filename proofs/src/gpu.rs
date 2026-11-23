//! Accelerator hooks of the prover (cargo feature `gpu`).
//!
//! A process can install one set of [`GpuHooks`] with [`install`]. The prover
//! offers these operations to the hooks before it runs its own code:
//!
//! - every G1 MSM of [`msm_specific`](crate::poly::kzg::msm::msm_specific)
//!   over BLS12-381 ([`GpuHooks::msm`]). The bases are projective. The
//!   commitments of the prover pass a prefix of the parameters' own vector
//!   (`ParamsKZG::g()` or `ParamsKZG::g_lagrange()`), so a hook can recognise
//!   them by pointer; the verifier's MSMs pass temporary vectors;
//! - the forward and inverse FFTs of
//!   [`EvaluationDomain`](crate::poly::EvaluationDomain) over the BLS12-381
//!   scalar field, on the base and the extended domain ([`GpuHooks::fft`],
//!   [`GpuHooks::ifft`]).
//!
//! Each hook either returns the exact value of the stock code or declines.
//! When a hook declines, the stock code runs on the unchanged input. With no
//! hooks installed the prover runs only the stock code. The hooks add, remove
//! or move no random draw and no transcript operation.
//!
//! Values cross the boundary as the plain limb arrays of `blst`, which the
//! `midnight-curves` types wrap transparently:
//!
//! - [`Scalar`]: `blst_fr`, four little-endian `u64` limbs in Montgomery form
//!   (`R = 2^256`);
//! - [`G1JacobianLimbs`]: `blst_p1`, Jacobian `X`, `Y`, `Z`, six little-endian
//!   `u64` limbs each in Montgomery form (`R = 2^384`); `Z = 0` is the point
//!   at infinity.
#![allow(unsafe_code)]

use std::{
    any::TypeId,
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, OnceLock,
    },
};

use midnight_curves::{Fq, G1Affine, G1Projective};

/// A scalar: `blst_fr` limbs, Montgomery form.
pub type Scalar = [u64; 4];
/// A Jacobian G1 point: `blst_p1` limbs.
pub type G1JacobianLimbs = [u64; 18];

const _: () = {
    assert!(std::mem::size_of::<Fq>() == std::mem::size_of::<Scalar>());
    assert!(std::mem::align_of::<Fq>() == std::mem::align_of::<Scalar>());
    assert!(std::mem::size_of::<G1Projective>() == std::mem::size_of::<G1JacobianLimbs>());
    assert!(std::mem::align_of::<G1Projective>() == std::mem::align_of::<G1JacobianLimbs>());
};

/// The MSM hook: `(scalars, bases)` with equal lengths, returns
/// `sum scalars[i] * bases[i]`, or `None` to decline.
pub type MsmFn = fn(&[Scalar], &[G1JacobianLimbs]) -> Option<G1JacobianLimbs>;
/// The FFT hook: `(a, omega, log_n)`. Returns `false` to decline.
pub type FftFn = fn(&mut [Scalar], Scalar, u32) -> bool;
/// The inverse FFT hook: `(a, omega_inv, log_n, divisor)`. Returns `false`
/// to decline.
pub type IfftFn = fn(&mut [Scalar], Scalar, u32, Scalar) -> bool;

/// The accelerator functions. Each one is optional, and each one may decline.
#[derive(Default, Clone, Copy, Debug)]
pub struct GpuHooks {
    /// `sum scalars[i] * bases[i]` over projective bases (equal lengths, any
    /// length, scalars may be zero). `None`: declined.
    pub msm: Option<MsmFn>,
    /// The FFT of `a` (length `2^log_n`) at the root `omega`, natural order
    /// in and out, in place. `false`: declined, and `a` must be unchanged.
    pub fft: Option<FftFn>,
    /// The FFT of `a` (length `2^log_n`) at `omega_inv`, then each element
    /// times `divisor`, in place. `false`: declined, and `a` must be
    /// unchanged.
    pub ifft: Option<IfftFn>,
}

static HOOKS: OnceLock<GpuHooks> = OnceLock::new();

/// Installs the hooks of this process. Returns `false`, and changes nothing,
/// when hooks are installed already.
pub fn install(hooks: GpuHooks) -> bool {
    HOOKS.set(hooks).is_ok()
}

/// The installed hooks, if any.
pub fn hooks() -> Option<&'static GpuHooks> {
    HOOKS.get()
}

// ------------------------------------------------------------------ limbs

/// The limbs of scalars.
pub fn scalar_limbs(a: &[Fq]) -> &[Scalar] {
    // SAFETY: `Fq` is a transparent `blst_fr`, a `repr(C)` `[u64; 4]`; size
    // and alignment are checked above.
    unsafe { &*(a as *const [Fq] as *const [Scalar]) }
}

/// The limbs of scalars, mutable. A write must leave reduced Montgomery
/// limbs.
pub fn scalar_limbs_mut(a: &mut [Fq]) -> &mut [Scalar] {
    // SAFETY: as `scalar_limbs`.
    unsafe { &mut *(a as *mut [Fq] as *mut [Scalar]) }
}

/// The scalars of limbs (reduced Montgomery form).
pub fn scalars_of_limbs(a: &[Scalar]) -> &[Fq] {
    // SAFETY: as `scalar_limbs`.
    unsafe { &*(a as *const [Scalar] as *const [Fq]) }
}

/// The scalars of limbs, mutable.
pub fn scalars_of_limbs_mut(a: &mut [Scalar]) -> &mut [Fq] {
    // SAFETY: as `scalar_limbs`.
    unsafe { &mut *(a as *mut [Scalar] as *mut [Fq]) }
}

/// The limbs of projective points.
pub fn point_limbs(a: &[G1Projective]) -> &[G1JacobianLimbs] {
    // SAFETY: `G1Projective` is a transparent `blst_p1`, `repr(C)` of three
    // `[u64; 6]`; size and alignment are checked above.
    unsafe { &*(a as *const [G1Projective] as *const [G1JacobianLimbs]) }
}

/// The projective points of limbs (points on the curve).
pub fn points_of_limbs(a: &[G1JacobianLimbs]) -> &[G1Projective] {
    // SAFETY: as `point_limbs`.
    unsafe { &*(a as *const [G1JacobianLimbs] as *const [G1Projective]) }
}

// ------------------------------------------------------------- dispatchers

#[inline]
fn same<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

/// The MSM hook for `msm_specific::<C>`: `S` its scalar, `P` its projective
/// curve. `None`: not taken, the stock code runs.
pub(crate) fn msm<C: 'static, S: 'static, P: 'static>(scalars: &[S], bases: &[P]) -> Option<P> {
    let f = hooks()?.msm?;
    if !same::<C, G1Affine>() || !same::<S, Fq>() || !same::<P, G1Projective>() {
        return declined("msm", "field-type");
    }
    if scalars.len() != bases.len() {
        return declined("msm", "length-mismatch");
    }
    // SAFETY: `S` is `Fq` and `P` is `G1Projective` (checked above).
    let s = unsafe { &*(scalars as *const [S] as *const [Fq]) };
    let b = unsafe { &*(bases as *const [P] as *const [G1Projective]) };
    let Some(r) = f(scalar_limbs(s), point_limbs(b)) else {
        return declined("msm", "hook");
    };
    served("msm", scalars.len());
    let r = points_of_limbs(std::slice::from_ref(&r))[0];
    // SAFETY: `P` is `G1Projective` (checked above).
    Some(unsafe { std::mem::transmute_copy::<G1Projective, P>(&r) })
}

/// The FFT hook for the domain's transforms. `true`: `a` holds the result.
pub(crate) fn fft<F: 'static + Copy>(a: &mut [F], omega: F, log_n: u32) -> bool {
    let Some(f) = hooks().and_then(|h| h.fft) else {
        return false;
    };
    if !same::<F, Fq>() {
        return declined_bool("fft", "field-type");
    }
    if a.len() as u64 != 1u64 << log_n {
        return declined_bool("fft", "length-mismatch");
    }
    // SAFETY: `F` is `Fq` (checked above).
    let a = unsafe { &mut *(a as *mut [F] as *mut [Fq]) };
    let omega = unsafe { std::mem::transmute_copy::<F, Fq>(&omega) };
    if !f(scalar_limbs_mut(a), scalar_limbs(&[omega])[0], log_n) {
        return declined_bool("fft", "hook");
    }
    served("fft", a.len());
    true
}

/// The inverse FFT hook (transform at the inverse root, then scale).
/// `true`: `a` holds the result.
pub(crate) fn ifft<F: 'static + Copy>(a: &mut [F], omega_inv: F, log_n: u32, divisor: F) -> bool {
    let Some(f) = hooks().and_then(|h| h.ifft) else {
        return false;
    };
    if !same::<F, Fq>() {
        return declined_bool("ifft", "field-type");
    }
    if a.len() as u64 != 1u64 << log_n {
        return declined_bool("ifft", "length-mismatch");
    }
    // SAFETY: `F` is `Fq` (checked above).
    let a = unsafe { &mut *(a as *mut [F] as *mut [Fq]) };
    let omega_inv = unsafe { std::mem::transmute_copy::<F, Fq>(&omega_inv) };
    let divisor = unsafe { std::mem::transmute_copy::<F, Fq>(&divisor) };
    let consts = [omega_inv, divisor];
    let consts = scalar_limbs(&consts);
    if !f(scalar_limbs_mut(a), consts[0], log_n, consts[1]) {
        return declined_bool("ifft", "hook");
    }
    served("ifft", a.len());
    true
}

// ---------------------------------------------------------------- counters

static SERVED_TOTAL: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static DECLINED_TOTAL: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static SERVED: Mutex<Vec<(&str, usize, u64)>> = Mutex::new(Vec::new());
static DECLINES: Mutex<Vec<(&str, &str, u64)>> = Mutex::new(Vec::new());

fn class_index(class: &str) -> usize {
    match class {
        "msm" => 0,
        "fft" => 1,
        _ => 2,
    }
}

fn served(class: &'static str, len: usize) {
    SERVED_TOTAL[class_index(class)].fetch_add(1, Ordering::Relaxed);
    let mut v = SERVED.lock().unwrap_or_else(|e| e.into_inner());
    match v.iter_mut().find(|e| e.0 == class && e.1 == len) {
        Some(e) => e.2 += 1,
        None => v.push((class, len, 1)),
    }
}

/// Counts a decline of an installed hook and returns `None`.
fn declined<T>(class: &'static str, reason: &'static str) -> Option<T> {
    DECLINED_TOTAL[class_index(class)].fetch_add(1, Ordering::Relaxed);
    let mut v = DECLINES.lock().unwrap_or_else(|e| e.into_inner());
    match v.iter_mut().find(|e| e.0 == class && e.1 == reason) {
        Some(e) => e.2 += 1,
        None => v.push((class, reason, 1)),
    }
    None
}

fn declined_bool(class: &'static str, reason: &'static str) -> bool {
    declined::<()>(class, reason);
    false
}

/// Totals of the hook calls since process start or the last [`reset_stats`].
/// A call made while no hook of its class is installed is not counted.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct HookStats {
    /// MSMs the hook returned a value for.
    pub msm_served: u64,
    /// MSMs declined (by this layer or by the hook).
    pub msm_declined: u64,
    /// FFTs the hook performed.
    pub fft_served: u64,
    /// FFTs declined.
    pub fft_declined: u64,
    /// Inverse FFTs the hook performed.
    pub ifft_served: u64,
    /// Inverse FFTs declined.
    pub ifft_declined: u64,
}

/// The totals of the hook calls.
pub fn stats() -> HookStats {
    let s = |i: usize| SERVED_TOTAL[i].load(Ordering::Relaxed);
    let d = |i: usize| DECLINED_TOTAL[i].load(Ordering::Relaxed);
    HookStats {
        msm_served: s(0),
        msm_declined: d(0),
        fft_served: s(1),
        fft_declined: d(1),
        ifft_served: s(2),
        ifft_declined: d(2),
    }
}

/// Served calls as `(class, length, count)`. Classes: `msm`, `fft`, `ifft`;
/// the length is the number of terms of an MSM or the length of a
/// transform.
pub fn served_sizes() -> Vec<(&'static str, usize, u64)> {
    SERVED.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Declines as `(class, reason, count)`. Reasons: `field-type` (not the
/// BLS12-381 types), `length-mismatch` (scalars and bases, or a transform
/// length and its `log_n`), `hook` (the hook itself declined).
pub fn hook_declines() -> Vec<(&'static str, &'static str, u64)> {
    DECLINES.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Resets [`stats`], [`served_sizes`] and [`hook_declines`].
pub fn reset_stats() {
    for a in SERVED_TOTAL.iter().chain(DECLINED_TOTAL.iter()) {
        a.store(0, Ordering::Relaxed);
    }
    SERVED.lock().unwrap_or_else(|e| e.into_inner()).clear();
    DECLINES.lock().unwrap_or_else(|e| e.into_inner()).clear();
}
