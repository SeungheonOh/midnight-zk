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

//! The binding to a backend of `midnight-accel` (`midnight-metal`,
//! `midnight-vulkan`) and `midnight_proofs::gpu`, generic over the backend.
//! It follows the CUDA binding (module `cuda` of `device.rs`): the hooks are
//! installed once per process and each declines when its phase is not
//! admitted for the current request.
//!
//! The backends decline a sparse MSM, so the number of MSMs the device serves
//! depends on the witness. The counters therefore report one MSM total
//! (every hook invocation, served or declined) and, for the label,
//! `msm_sized`: the invocations the device takes by length and bases before
//! reading any scalar. There is no `msm` entry in `calls` and no MSM row in
//! `declines` (see `docs/gpu-prover/security.md`, section 8).

use std::{
    marker::PhantomData,
    sync::{
        Mutex, OnceLock, RwLock,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

use midnight_accel::{
    self as accel, G1AffineLimbs, G1JacobianLimbs, KeyHandle, Scalar, SrsHandle, WipeStats,
};
use midnight_proofs::{
    gpu::{self as hooks, GpuHooks, adapter},
    poly::{Coeff, LagrangeCoeff, LagrangeDeltaCoeff, LagrangeDoubleDeltaCoeff},
};

use super::*;

/// A backend of `midnight-accel` the service can run on: what differs
/// between the backend crates. Everything else is the generic API of
/// `midnight-accel`, which the backend crates wrap.
pub(super) trait Accel: accel::Backend {
    /// The binding's process-wide state for this backend.
    fn binding() -> &'static Binding;
    /// Whether the backend crate finds a device (making no pipeline).
    fn present() -> bool;
    /// The device's facts for the start record.
    fn info() -> Option<DeviceInfo>;
}

/// The process-wide state of the binding to one backend.
pub(super) struct Binding {
    /// The phases admitted for the current request (bits of [`Phases`]).
    admitted: AtomicU32,
    /// Nanoseconds of the calls the device served, by class. For `msm`
    /// (class 0): of every hook invocation.
    served_ns: [AtomicU64; 4],
    /// Hook calls declined as not admitted, by class.
    not_admitted: [AtomicU64; 4],
    /// MSM hook invocations, served or declined.
    msm_calls: AtomicU64,
    /// MSM hook invocations the device takes by length and bases
    /// (`msm_g1_sized`), whatever the scalars.
    msm_sized: AtomicU64,
    /// The request's cancel check ([`Device::set_gate`]).
    gate: RwLock<Option<crate::service::Cancel>>,
    /// The gate withdrew the device phases.
    withdrawn: AtomicBool,
    /// This binding made the device context (the first device operation).
    context: AtomicBool,
    /// Whether this binding installed the hooks.
    installed: OnceLock<bool>,
}

impl Binding {
    /// The state of a process that has not used the backend.
    pub(super) const fn new() -> Binding {
        Binding {
            admitted: AtomicU32::new(0),
            served_ns: [const { AtomicU64::new(0) }; 4],
            not_admitted: [const { AtomicU64::new(0) }; 4],
            msm_calls: AtomicU64::new(0),
            msm_sized: AtomicU64::new(0),
            gate: RwLock::new(None),
            withdrawn: AtomicBool::new(false),
            context: AtomicBool::new(false),
            installed: OnceLock::new(),
        }
    }
}

/// `Err` unless the erasure is confirmed with no non-zero region, failed
/// check or failed erasure.
fn wipe_result(r: Result<WipeStats, WipeStats>) -> Result<WipeReport, String> {
    let (w, confirmed) = match r {
        Ok(w) => (w, true),
        Err(w) => (w, false),
    };
    let report = WipeReport {
        regions: w.regions,
        bytes: w.bytes,
        checked: w.checked,
        nonzero_regions: w.nonzero_regions,
        check_failures: w.check_failures,
        erasure_failures: w.erasure_failures,
    };
    if confirmed
        && report.nonzero_regions == 0
        && report.check_failures == 0
        && report.erasure_failures == 0
    {
        return Ok(report);
    }
    Err(format!(
        "the device erasure was not confirmed ({} non-zero regions, {} failed checks, {} \
         failed erasures, of {} regions, {} of them idle pool buffers)",
        report.nonzero_regions,
        report.check_failures,
        report.erasure_failures,
        report.regions,
        w.pool_regions
    ))
}

/// Whether `p` is admitted, counting a `not-admitted` decline otherwise. The
/// request's gate runs before each device phase; a failed gate withdraws
/// every device phase.
fn admitted<B: Accel>(p: Phases, class: usize) -> bool {
    let b = B::binding();
    let mut ok = Phases::from_bits(b.admitted.load(Relaxed)).contains(p);
    if ok && b.gate.read().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|g| g()) {
        b.admitted.store(0, Relaxed);
        b.withdrawn.store(true, Relaxed);
        ok = false;
    }
    if !ok {
        b.not_admitted[class].fetch_add(1, Relaxed);
    }
    ok
}

fn timed<B: Accel>(class: usize, t: Instant, counted: bool) {
    if counted {
        B::binding().served_ns[class].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
    }
}

fn msm<B: Accel>(scalars: &[Scalar], bases: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
    B::binding().msm_calls.fetch_add(1, Relaxed);
    if !admitted::<B>(Phases::MSM, 0) {
        return None;
    }
    let t = Instant::now();
    let (sized, r) = accel::msm_g1_sized::<B>(scalars, bases);
    if sized {
        B::binding().msm_sized.fetch_add(1, Relaxed);
    }
    // timed whether served or declined, so the time does not tell them apart
    timed::<B>(0, t, true);
    r
}

fn fft<B: Accel>(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    if !admitted::<B>(Phases::FFT, 1) {
        return false;
    }
    let t = Instant::now();
    let ok = accel::fft::<B>(a, omega, log_n);
    timed::<B>(1, t, ok);
    ok
}

fn ifft<B: Accel>(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    if !admitted::<B>(Phases::IFFT, 2) {
        return false;
    }
    let t = Instant::now();
    let ok = accel::ifft::<B>(a, omega_inv, log_n, divisor);
    timed::<B>(2, t, ok);
    ok
}

/// # Safety
/// As [`midnight_accel::quotient`]: `job` is the job the prover passes.
unsafe fn quotient<B: Accel>(job: *const std::ffi::c_void, out: &mut [Scalar]) -> bool {
    if !admitted::<B>(Phases::QUOTIENT, 3) {
        return false;
    }
    let t = Instant::now();
    // SAFETY: the prover's job pointer, passed through unchanged
    let ok = unsafe { accel::quotient::<B>(job, out) };
    timed::<B>(3, t, ok);
    ok
}

/// `false` when another component installed hooks first.
fn install<B: Accel>() -> bool {
    *B::binding().installed.get_or_init(|| {
        hooks::install(GpuHooks {
            msm_g1: Some(msm::<B>),
            fft: Some(fft::<B>),
            ifft: Some(ifft::<B>),
            quotient: Some(quotient::<B>),
        })
    })
}

/// Decline reasons that mean the device was asked and the CPU code ran
/// instead (a fallback). `below-min-k` and `no-device` are by design,
/// `not-admitted` is the admission's decision.
fn is_fallback(reason: &str) -> bool {
    matches!(reason, "memory" | "device-error" | "shape" | "unreduced")
}

/// The declines by phase and reason, without MSM rows, and the MSM calls the
/// hook layer declined before the hook (added to the MSM total).
fn decline_rows<B: Accel>() -> (Vec<Fallback>, u64) {
    let row = |class: &str, reason: &str, count: u64| Fallback {
        phase: class.to_string(),
        reason: reason.to_string(),
        count,
    };
    let mut rows: Vec<Fallback> = accel::declines::<B>()
        .into_iter()
        .filter(|(class, _, _)| *class != "msm")
        .map(|(class, reason, count)| row(class, reason, count))
        .collect();
    let mut msm_unhooked = 0;
    // `hook` mirrors a decline counted above
    for (class, reason, count) in hooks::hook_declines() {
        if reason == "hook" {
            continue;
        }
        if class == "msm" {
            msm_unhooked += count;
        } else {
            rows.push(row(class, reason, count));
        }
    }
    for (i, class) in CLASSES.iter().enumerate().skip(1) {
        let n = B::binding().not_admitted[i].load(Relaxed);
        if n > 0 {
            rows.push(row(class, "not-admitted", n));
        }
    }
    (rows, msm_unhooked)
}

struct State<B: Accel> {
    srs: Vec<SrsHandle<B>>,
    key: Option<KeyHandle<B>>,
    admitted: Phases,
    /// The prepared key reaches the quotient's size threshold.
    sized: bool,
}

impl<B: Accel> Default for State<B> {
    fn default() -> Self {
        State {
            srs: Vec::new(),
            key: None,
            admitted: Phases::NONE,
            sized: false,
        }
    }
}

impl<B: Accel> State<B> {
    fn table_bytes(&self) -> u64 {
        self.srs.iter().map(SrsHandle::device_bytes).sum()
    }
}

/// The device of backend `B` in this process.
pub struct AccelDevice<B: Accel> {
    state: Mutex<State<B>>,
    _backend: PhantomData<fn() -> B>,
}

impl<B: Accel> AccelDevice<B> {
    /// A device over the binding's state, without detection.
    fn new() -> AccelDevice<B> {
        AccelDevice {
            state: Mutex::new(State::default()),
            _backend: PhantomData,
        }
    }

    /// The device, when one is present and the hooks are installed. Makes no
    /// device context.
    pub fn detect() -> Option<AccelDevice<B>> {
        if !B::present() || !install::<B>() {
            return None;
        }
        B::binding().admitted.store(0, Relaxed);
        Some(AccelDevice::new())
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State<B>> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<B: Accel> Device for AccelDevice<B> {
    fn info(&self) -> Option<DeviceInfo> {
        B::info()
    }

    fn mem(&self) -> Option<MemInfo> {
        // reading the memory info would make the context: before this binding
        // made it, none is reported free
        if !B::binding().context.load(Relaxed) {
            return Some(MemInfo {
                free: 0,
                total: B::info()?.total_bytes,
            });
        }
        accel::mem_info::<B>().map(|(free, total)| MemInfo {
            free: free as u64,
            total: total as u64,
        })
    }

    fn held(&self) -> Held {
        let s = self.state();
        Held {
            resident: s.key.as_ref().map_or(0, KeyHandle::resident_bytes),
            tables: s.table_bytes(),
        }
    }

    fn shape(&self, pk: &Pk) -> Option<Shape> {
        let q = hooks::quotient_shape(pk)?;
        Some(Shape {
            k: q.k,
            ext_k: q.ext_k,
            proof_cosets: q.proof_polys as u32,
            resident_polys: q.key_polys as u32,
            nslots: q.nslots,
            grid_threads: u64::from(accel::quotient_threads::<B>()).min(1u64 << q.ext_k),
            quotient_bytes: accel::need_bytes::<B>(q.ext_k, q.proof_polys, q.key_polys, q.nslots)
                as u64,
        })
    }

    fn set_keep(&self, keep: bool) {
        accel::set_keep_resident::<B>(keep);
    }

    fn prepare(&self, params: &ParamsKZG<Bls12>, pk: &Pk, _check: bool) {
        let mut s = self.state();
        s.sized = pk.get_vk().get_domain().extended_k() >= accel::thresholds::<B>().quotient;
        if s.srs.is_empty() {
            s.srs = [
                params.bases::<Coeff>(),
                params.bases::<LagrangeCoeff>(),
                params.bases::<LagrangeDeltaCoeff>(),
                params.bases::<LagrangeDoubleDeltaCoeff>(),
            ]
            .into_iter()
            .map(|b| accel::register_srs::<B>(adapter::affine(b)))
            .collect();
        }
        if s.key.is_none()
            && let Some(polys) = pk.gpu_resident_polys()
        {
            let limbs: Vec<&[Scalar]> = polys.iter().map(|p| adapter::scalars(p)).collect();
            s.key = Some(accel::register_key::<B>(&limbs));
        }
    }

    fn begin_request(&self, phases: Phases) -> Phases {
        if phases.is_empty() || !accel::device::<B>() {
            return Phases::NONE;
        }
        B::binding().context.store(true, Relaxed);
        accel::begin_request::<B>();
        self.state().admitted = phases;
        B::binding().admitted.store(phases.bits(), Relaxed);
        phases
    }

    fn revoke(&self) -> bool {
        B::binding().admitted.swap(0, Relaxed) != 0
    }

    fn set_gate(&self, gate: Option<crate::service::Cancel>) {
        *B::binding().gate.write().unwrap_or_else(|e| e.into_inner()) = gate;
    }

    fn withdrawn(&self) -> bool {
        B::binding().withdrawn.swap(false, Relaxed)
    }

    fn activate(&self) {
        if accel::device::<B>() {
            B::binding().context.store(true, Relaxed);
        }
    }

    fn evict(&self, tables: bool) -> u64 {
        // the key copies return to the pool, so release the pool after them;
        // the key registration holds no device memory and stays
        let mut s = self.state();
        let before = accel::device_bytes::<B>().0;
        accel::evict_key_copies::<B>();
        accel::release_pool::<B>();
        let mut freed = before.saturating_sub(accel::device_bytes::<B>().0);
        if tables {
            freed += s.table_bytes();
            s.srs.clear();
        }
        freed
    }

    fn end_request(&self, check: bool) -> Result<WipeReport, String> {
        B::binding().admitted.store(0, Relaxed);
        wipe_result(accel::end_request_report::<B>(check))
    }

    fn counters(&self) -> DeviceCounters {
        let s = self.state();
        let st = accel::stats::<B>();
        let b = B::binding();
        // no `msm` entry: see the module docs
        let calls = [
            ("fft", st.fft.served),
            ("ifft", st.ifft.served),
            ("quotient", st.quotient.served),
        ]
        .into_iter()
        .filter(|(_, n)| *n > 0)
        .map(|(k, n)| (k.to_string(), n))
        .collect();
        let (declines, msm_unhooked) = decline_rows::<B>();
        let seconds = |i: usize| b.served_ns[i].load(Relaxed) as f64 * 1e-9;
        let phase_seconds = CLASSES
            .iter()
            .enumerate()
            .filter(|(i, _)| *i == 0 || b.served_ns[*i].load(Relaxed) > 0)
            .map(|(i, c)| (c.to_string(), seconds(i)))
            .collect();
        DeviceCounters {
            msm_count: b.msm_calls.load(Relaxed) + msm_unhooked,
            msm_seconds: seconds(0),
            msm_sized: b.msm_sized.load(Relaxed),
            calls,
            fallbacks: fallback_rows(&declines, s.admitted, s.sized, is_fallback),
            declines,
            // reset only when a request starts on the device
            peak_bytes: if s.admitted.is_empty() {
                0
            } else {
                accel::request_peak_bytes::<B>()
            },
            phase_seconds,
        }
    }

    fn reset_counters(&self) {
        let b = B::binding();
        accel::reset_stats::<B>();
        hooks::reset_hook_declines();
        for c in b
            .not_admitted
            .iter()
            .chain(b.served_ns.iter())
            .chain([&b.msm_calls, &b.msm_sized])
        {
            c.store(0, Relaxed);
        }
        self.state().admitted = Phases::NONE;
    }
}

/// The `#[test]` functions of the binding's tests for one backend.
#[cfg(test)]
macro_rules! binding_tests {
    ($b:ty) => {
        #[cfg(test)]
        mod tests {
            use crate::device::accel::tests as t;

            #[test]
            fn a_declined_msm_is_counted_in_the_total_and_other_declines_by_reason() {
                t::a_declined_msm_is_counted_in_the_total_and_other_declines_by_reason::<$b>();
            }

            #[test]
            fn a_failed_gate_withdraws_before_the_launch() {
                t::a_failed_gate_withdraws_before_the_launch::<$b>();
            }
        }
    };
}
#[cfg(test)]
pub(super) use binding_tests;

/// The binding's tests, run by each backend's module ([`binding_tests`]).
#[cfg(test)]
pub(super) mod tests {
    use std::sync::Mutex;

    use super::*;

    /// The hook layer's counters are process-wide: one test at a time, over
    /// every backend.
    static SERIAL: Mutex<()> = Mutex::new(());

    /// A declined MSM counts in the total only; other declines count by
    /// phase and reason.
    pub(in crate::device) fn a_declined_msm_is_counted_in_the_total_and_other_declines_by_reason<
        B: Accel,
    >() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = AccelDevice::<B>::new();
        d.reset_counters();
        d.state().admitted = Phases::MSM;
        B::binding().admitted.store(Phases::MSM.bits(), Relaxed);
        assert!(msm::<B>(&[[0u64; 4]], &[[0u64; 12]]).is_none());
        assert!(!fft::<B>(&mut [[0u64; 4]; 2], [0u64; 4], 1));
        assert!(!ifft::<B>(&mut [[0u64; 4]; 2], [0u64; 4], 1, [0u64; 4]));
        B::binding().admitted.store(0, Relaxed);
        let c = d.counters();
        let row = |p: &str, r: &str| {
            c.declines
                .iter()
                .filter(|f| f.phase == p && f.reason == r)
                .map(|f| f.count)
                .sum::<u64>()
        };
        assert_eq!(c.msm_count, 1);
        assert!(
            c.declines.iter().all(|f| f.phase != "msm"),
            "{:?}",
            c.declines
        );
        assert!(!c.calls.contains_key("msm"));
        assert_eq!(row("fft", "not-admitted"), 1, "{:?}", c.declines);
        assert_eq!(row("ifft", "not-admitted"), 1, "{:?}", c.declines);
        assert_eq!(c.fallback_count(), 0);
    }

    /// A failed gate withdraws every device phase before the launch.
    pub(in crate::device) fn a_failed_gate_withdraws_before_the_launch<B: Accel>() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let d = AccelDevice::<B>::new();
        d.reset_counters();
        d.withdrawn();
        B::binding().admitted.store(Phases::FFT.union(Phases::IFFT).bits(), Relaxed);
        d.set_gate(Some(std::sync::Arc::new(|| true)));
        assert!(!fft::<B>(&mut [[0u64; 4]; 2], [0u64; 4], 1));
        assert_eq!(B::binding().admitted.load(Relaxed), 0);
        assert!(d.withdrawn());
        d.set_gate(None);
        d.reset_counters();
    }

    /// Any non-zero region, failed check or failed erasure fails the wipe.
    #[test]
    fn a_failed_check_is_counted_and_fails_the_wipe() {
        let w = WipeStats {
            regions: 3,
            bytes: 300,
            checked: true,
            checked_regions: 3,
            ..Default::default()
        };
        let r = wipe_result(Ok(w)).unwrap();
        assert_eq!((r.regions, r.bytes, r.checked), (3, 300, true));
        assert_eq!((r.nonzero_regions, r.check_failures), (0, 0));
        let e = wipe_result(Err(WipeStats {
            nonzero_regions: 1,
            ..w
        }))
        .unwrap_err();
        assert!(e.contains("1 non-zero regions, 0 failed checks"), "{e}");
        let e = wipe_result(Err(WipeStats {
            check_failures: 2,
            ..w
        }))
        .unwrap_err();
        assert!(e.contains("2 failed checks"), "{e}");
        // a zeroing that panicked is an erasure failure
        let e = wipe_result(Err(WipeStats {
            erasure_failures: 1,
            ..w
        }))
        .unwrap_err();
        assert!(
            e.contains("0 non-zero regions, 0 failed checks, 1 failed erasures"),
            "{e}"
        );
        // also when the crate's result is `Ok`
        let e = wipe_result(Ok(WipeStats {
            erasure_failures: 1,
            ..w
        }))
        .unwrap_err();
        assert!(e.contains("1 failed erasures"), "{e}");
        let r = wipe_result(Ok(WipeStats {
            pool_regions: 2,
            pool_bytes: 200,
            ..w
        }))
        .unwrap();
        assert_eq!(r.erasure_failures, 0);
    }
}
