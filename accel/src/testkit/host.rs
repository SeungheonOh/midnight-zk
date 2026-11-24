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

//! Tests of the field arithmetic against blst, of the device context and of
//! the hook boundary.

use ff::Field;
use midnight_curves::{Fp, Fq};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use crate::{
    Backend, RawBuffer,
    ctx::{self, Arg, ReadBack},
    field::{FP_MODULUS, FR_MODULUS, fp_is_reduced, is_reduced},
};

// Operation codes of shaders/field_test.metal.
const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_SQR: u32 = 3;
const OP_NEG: u32 = 4;
const OP_DBL: u32 = 5;
const OP_IS_REDUCED: u32 = 6;
const OP_ONE: u32 = 7;
const OP_EQ: u32 = 8;

fn require_device<B: Backend>() {
    assert!(
        crate::device::<B>(),
        "no usable {} device: {:?}",
        B::NAME,
        ctx::context::<B>().err()
    );
}

/// Runs `kernel` (`field_test_fr` / `field_test_fp`) on `a`, `b`.
fn run_op<B: Backend, T: ctx::Pod + Default>(kernel: &str, op: u32, a: &[T], b: &[T]) -> Vec<T> {
    let s = ctx::lock::<B>().unwrap();
    let c = s.ctx();
    let (ba, bb) = (c.upload(a).unwrap(), c.upload(b).unwrap());
    let out = c.alloc(size_of_val(a)).unwrap();
    let n = a.len() as u32;
    c.run(|cmd| {
        let args = [Arg::Buffer(&ba), Arg::Buffer(&bb), Arg::Buffer(&out)];
        cmd.dispatch(kernel, a.len(), &args, &[op, n])
    })
    .unwrap();
    let mut r = vec![T::default(); a.len()];
    out.read(&mut r);
    r
}

/// `op` by blst, through the midnight-curves field `F` whose element is a
/// transparent blst value of `N` limbs.
fn blst_op<F: Field, const N: usize>(op: u32, a: &[u64; N], b: &[u64; N]) -> [u64; N] {
    assert_eq!(size_of::<F>(), size_of::<[u64; N]>());
    if op == OP_EQ {
        let mut flag = [0u64; N];
        flag[0] = (a == b) as u64;
        return flag;
    }
    // SAFETY: `F` is a transparent `[u64; N]` of Montgomery limbs (sizes
    // checked above), and every operand is reduced.
    let (fa, fb) = unsafe {
        (
            std::mem::transmute_copy::<_, F>(a),
            std::mem::transmute_copy::<_, F>(b),
        )
    };
    let r = match op {
        OP_ADD => fa + fb,
        OP_SUB => fa - fb,
        OP_MUL => fa * fb,
        OP_SQR => fa.square(),
        OP_NEG => -fa,
        OP_DBL => fa.double(),
        OP_ONE => F::ONE,
        _ => unreachable!(),
    };
    // SAFETY: as above.
    unsafe { std::mem::transmute_copy(&r) }
}

fn sub_small<const N: usize>(m: &[u64; N], k: u64) -> [u64; N] {
    let mut r = *m;
    let mut borrow = k;
    for l in r.iter_mut() {
        let (v, o) = l.overflowing_sub(borrow);
        *l = v;
        borrow = o as u64;
    }
    r
}

/// Every pair of edge values (0, 1, 2, the Montgomery one, p-1, p-2,
/// p-2^32, (p-1)/2, (p+1)/2, a single high bit, all-ones limbs below p, a
/// 32-bit word), then random pairs, some sharing an operand, some near p.
fn pairs<const N: usize>(m: &[u64; N], mont_one: [u64; N], seed: u64) -> Vec<([u64; N], [u64; N])> {
    let low = |v: u64| {
        let mut r = [0u64; N];
        r[0] = v;
        r
    };
    let mut half = sub_small(m, 1);
    for i in 0..N {
        half[i] = (half[i] >> 1) | half.get(i + 1).map_or(0, |h| h << 63);
    }
    let mut half_up = half;
    half_up[0] += 1;
    let mut top = [0u64; N];
    top[N - 1] = 1 << (63 - m[N - 1].leading_zeros());
    let mut lowfull = [u64::MAX; N];
    lowfull[N - 1] = 0;
    let e = [
        low(0),
        low(1),
        low(2),
        mont_one,
        sub_small(m, 1),
        sub_small(m, 2),
        sub_small(m, 1 << 32),
        half,
        half_up,
        top,
        lowfull,
        low(0xffff_ffff),
    ];
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let mut random = || loop {
        let mut a: [u64; N] = std::array::from_fn(|_| rng.next_u64());
        a[N - 1] &= u64::MAX >> m[N - 1].leading_zeros();
        if a.iter().rev().lt(m.iter().rev()) {
            return a;
        }
    };
    let mut v: Vec<_> = e.iter().flat_map(|x| e.iter().map(|y| (*x, *y))).collect();
    for i in 0..1 << 16 {
        let x = random();
        let y = match i % 8 {
            0 => x,
            1 => sub_small(m, 1 + (random()[0] & 0xffff)),
            2 => e[i % e.len()],
            _ => random(),
        };
        v.push((x, y));
    }
    v
}

/// Values at, just above and far above the modulus, the largest limbs, and
/// random values with a top limb below four times the modulus'.
fn unreduced<const N: usize>(m: &[u64; N]) -> Vec<[u64; N]> {
    let plus = |i: usize, d: i64| {
        let mut r = *m;
        r[i] = r[i].wrapping_add_signed(d);
        r
    };
    let mut top = [0u64; N];
    top[N - 1] = 1 << 63;
    let mut v = vec![
        *m,
        sub_small(m, 1),
        [u64::MAX; N],
        [0; N],
        plus(0, 1),
        plus(N - 1, 1),
        plus(N - 1, -1),
        top,
    ];
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    for _ in 0..1000 {
        let mut a: [u64; N] = std::array::from_fn(|_| rng.next_u64());
        a[N - 1] %= m[N - 1].saturating_mul(4);
        v.push(a);
    }
    v
}

/// Every operation of `kernel` against blst on edge and random pairs, and
/// `is_reduced` against the host on reduced and unreduced values.
fn field_matches_blst<B: Backend, F: Field, const N: usize>(
    kernel: &str,
    m: &[u64; N],
    seed: u64,
    reduced: fn(&[u64; N]) -> bool,
) where
    [u64; N]: Default,
{
    require_device::<B>();
    let one = blst_op::<F, N>(OP_ONE, &[0; N], &[0; N]);
    for op in [
        OP_ADD, OP_SUB, OP_MUL, OP_SQR, OP_NEG, OP_DBL, OP_ONE, OP_EQ,
    ] {
        let (a, b): (Vec<_>, Vec<_>) = pairs(m, one, seed ^ op as u64).into_iter().unzip();
        let got = run_op::<B, _>(kernel, op, &a, &b);
        for i in 0..a.len() {
            let want = blst_op::<F, N>(op, &a[i], &b[i]);
            assert_eq!(
                got[i], want,
                "{kernel} op {op}: a = {:x?}, b = {:x?}",
                a[i], b[i]
            );
        }
    }
    let a = unreduced(m);
    let got = run_op::<B, _>(kernel, OP_IS_REDUCED, &a, &a);
    let mut both = [0usize; 2];
    for (x, g) in a.iter().zip(&got) {
        assert_eq!(g[0], reduced(x) as u64, "{kernel} is_reduced {x:x?}");
        both[g[0] as usize] += 1;
    }
    assert!(both[0] > 0 && both[1] > 0);
}

pub fn fr_matches_blst<B: Backend>() {
    field_matches_blst::<B, Fq, 4>("field_test_fr", &FR_MODULUS, 0x5eed, is_reduced);
}

pub fn fp_matches_blst<B: Backend>() {
    field_matches_blst::<B, Fp, 6>("field_test_fp", &FP_MODULUS, 0xf00d, fp_is_reduced);
}

pub fn errors_are_values<B: Backend>() {
    require_device::<B>();
    let s = ctx::lock::<B>().unwrap();
    let c = s.ctx();
    let e = c.run(|cmd| cmd.dispatch("no_such_kernel", 1, &[], &())).unwrap_err();
    assert_eq!(e.reason(), "device-error");
    let e = c.alloc(usize::MAX / 2).unwrap_err();
    assert_eq!(e.reason(), "memory");
    let out = c.alloc(64).unwrap();
    let n = 1u32;
    let e = c
        .run(|cmd| cmd.dispatch_with("diag_iota", 1, 1 << 20, &[Arg::Buffer(&out)], &n, &[]))
        .unwrap_err();
    assert_eq!(e.reason(), "device-error");
    c.run(|cmd| cmd.dispatch("diag_iota", 1, &[Arg::Buffer(&out)], &n)).unwrap();
}

/// The tests that change the thresholds, hold buffers across a session or
/// compare byte counts run one at a time.
fn serial<B: Backend>() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    require_device::<B>();
    g
}

pub fn request_buffers_and_byte_accounting<B: Backend>() {
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    let mut s = ctx::lock::<B>().unwrap();
    let before = ctx::in_use_bytes::<B>();
    {
        let b = s.scratch("test-scratch", 1 << 20).unwrap();
        b.as_mut_slice::<u64>().fill(0xdead_beef);
    }
    // the same key, not longer: the same buffer with its contents
    let b = s.scratch("test-scratch", 1 << 19).unwrap();
    assert!(b.as_slice::<u64>().iter().all(|v| *v == 0xdead_beef));
    assert!(s.scratch_bytes() >= 1 << 20);
    let mid = ctx::in_use_bytes::<B>();
    let (held, peak) = crate::device_bytes::<B>();
    assert!(mid >= before + (1 << 20) && held >= mid && peak >= held);
    {
        let owned = s.ctx().alloc(1 << 16).unwrap();
        assert!(ctx::in_use_bytes::<B>() >= mid + (1 << 16));
        drop(owned);
    }
    assert_eq!(ctx::in_use_bytes::<B>(), mid);
    drop(s);
    assert!(crate::end_request::<B>());
    let s = ctx::lock::<B>().unwrap();
    assert!(s.get_scratch("test-scratch").is_none());
    assert_eq!(s.scratch_bytes(), 0);
    assert_eq!(ctx::in_use_bytes::<B>(), before);
    assert!(crate::pool_bytes::<B>() >= 1 << 20);
    drop(s);
    let (held, _) = crate::device_bytes::<B>();
    let freed = crate::release_pool::<B>();
    assert!(freed >= 1 << 20);
    assert_eq!(crate::pool_bytes::<B>(), 0);
    assert_eq!(crate::device_bytes::<B>().0, held - freed);
}

/// A pooled buffer goes back to the pool only zeroed, and comes out of it
/// zeroed; the pool keeps no more than its limit.
pub fn pooled_buffers_come_back_zeroed<B: Backend>() {
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    let c = ctx::context::<B>().unwrap();
    let len = (3 << 20) + 4096;
    let mut a = c.alloc_pooled(len).unwrap();
    a.as_mut_slice::<u8>().fill(0x5a);
    let at = a.id();
    drop(a);
    assert!(crate::pool_bytes::<B>() >= len as u64);
    // a test running at the same time may take that buffer first
    let b = c.alloc_pooled(len - 4096).unwrap();
    assert_eq!(b.len_bytes(), len - 4096);
    if b.id() == at {
        assert_eq!(b.alloc_bytes(), len);
    }
    // the whole allocation
    assert_eq!(b.raw().read_back_zero(), ReadBack::Zero);
    drop(b);
    let small = c.alloc_pooled(4096).unwrap();
    assert_ne!(small.id(), at);
    drop(small);
    // a buffer zeroed in the background
    let mut d = c.alloc_pooled(len).unwrap();
    d.as_mut_slice::<u8>().fill(0x33);
    ctx::release_later::<B>(vec![d]);
    ctx::settle_releases::<B>();
    let e = c.alloc_pooled(len).unwrap();
    assert!(e.as_slice::<u8>().iter().all(|x| *x == 0));
    drop(e);
    ctx::set_pool_limit::<B>(Some(0));
    assert_eq!(crate::pool_bytes::<B>(), 0);
    let f = c.alloc_pooled(len).unwrap();
    drop(f);
    assert_eq!(
        crate::pool_bytes::<B>(),
        0,
        "nothing is kept above the limit"
    );
    ctx::set_pool_limit::<B>(None);
    assert!(crate::end_request::<B>());
}

mod stock {
    use ff::{Field, PrimeField};
    use group::{Curve, Group};
    use midnight_curves::{Fq, G1Affine, G1Projective};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    /// The standard `2^log_n`-th root of unity.
    pub fn root(log_n: u32) -> Fq {
        let mut w = Fq::ROOT_OF_UNITY;
        for _ in log_n..Fq::S {
            w = w.square();
        }
        w
    }

    pub fn scalars(n: usize, seed: u64) -> Vec<Fq> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..n).map(|_| Fq::random(&mut rng)).collect()
    }

    pub fn points(n: usize, seed: u64) -> Vec<G1Affine> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let step = G1Projective::random(&mut rng);
        let mut p = G1Projective::random(&mut rng);
        let proj: Vec<G1Projective> = (0..n)
            .map(|_| {
                p += step;
                p
            })
            .collect();
        let mut out = vec![G1Affine::default(); n];
        G1Projective::batch_normalize(&proj, &mut out);
        out
    }

    pub fn msm(s: &[Fq], b: &[G1Affine]) -> G1Affine {
        midnight_curves::msm::msm_pippenger(s, b).to_affine()
    }
}

fn decline_count<B: Backend>(class: &str, reason: &str) -> u64 {
    crate::declines::<B>()
        .iter()
        .filter(|d| d.0 == class && d.1 == reason)
        .map(|d| d.2)
        .sum()
}

/// Without an explicit threshold each class takes its default; an explicit
/// one applies to every class; an MSM below the one-shot threshold is served
/// from a registered table and declines without one.
pub fn per_class_thresholds<B: Backend>() {
    use group::Curve;
    use midnight_proofs::gpu::adapter::{affine, from_jacobian, scalar, scalars, scalars_mut};
    let _g = serial::<B>();
    crate::reset_min_k::<B>();
    let min_k_var = format!("{}_MIN_K", B::ENV_PREFIX);
    if std::env::var_os(&min_k_var).is_some() {
        eprintln!("{min_k_var} is set: the defaults are not in force; skipped");
        return;
    }
    let d = B::TUNING.thresholds;
    assert_eq!(crate::thresholds::<B>(), d);
    assert!(!d.explicit);
    assert_eq!(d.msm_one_shot, None, "the one-shot MSM is off by default");
    assert_eq!(
        crate::min_k::<B>(),
        d.fft.min(d.ifft).min(d.msm_table).min(d.quotient)
    );

    // a transform one below its threshold, an MSM class `k` table threshold
    let log_n = d.fft - 1;
    let k = d.msm_table;
    let input = stock::scalars((1 << log_n).max(1 << (k + 1)), 3);
    let mut a = input[..1 << log_n].to_vec();
    let below = decline_count::<B>("fft", "below-min-k");
    assert!(!crate::fft::<B>(
        scalars_mut(&mut a),
        scalar(stock::root(log_n)),
        log_n
    ));
    assert_eq!(a, input[..1 << log_n]);
    assert_eq!(decline_count::<B>("fft", "below-min-k") - below, 1);

    let points = stock::points(3 << k, 4);
    let m = 1 << (k + 1);
    let (cached, one_shot, uploads) = crate::msm_counts::<B>();
    let checks = crate::one_shot_g1_checks::<B>();
    let unregistered = decline_count::<B>("msm", "unregistered-bases");
    assert!(!crate::msm_sized::<B>(m, affine(&points[..m])));
    assert!(crate::msm_g1::<B>(scalars(&input[..m]), affine(&points[..m])).is_none());
    assert_eq!(
        decline_count::<B>("msm", "unregistered-bases") - unregistered,
        1
    );
    assert_eq!(
        crate::msm_counts::<B>().1,
        one_shot,
        "no one-shot device MSM"
    );
    assert_eq!(crate::one_shot_g1_checks::<B>(), checks, "no G1 check");
    let h = crate::register_srs::<B>(affine(&points));
    assert_eq!(h.device_bytes(), 0, "a registration is host memory");
    let r = crate::msm_g1::<B>(scalars(&input[..m]), affine(&points[..m]))
        .expect("served from the table");
    assert_eq!(
        from_jacobian(r).to_affine(),
        stock::msm(&input[..m], &points[..m])
    );
    let (cached2, one_shot2, uploads2) = crate::msm_counts::<B>();
    assert_eq!(
        (cached2 - cached, one_shot2 - one_shot, uploads2 - uploads),
        (1, 0, 1)
    );
    assert_eq!(h.device_bytes(), (points.len() * 96) as u64);
    // the size class of a length is its log2 rounded to the nearest integer
    let below = (1 << k) - 1;
    let r = crate::msm_g1::<B>(scalars(&input[..below]), affine(&points[..below]))
        .expect("class k, served from the table");
    assert_eq!(
        from_jacobian(r).to_affine(),
        stock::msm(&input[..below], &points[..below])
    );
    let quarter = 1 << (k - 1);
    assert!(crate::msm_g1::<B>(scalars(&input[..quarter]), affine(&points[..quarter])).is_none());
    // the first length of class `k` (5794 for class 13)
    let edge = (1..).find(|&n| crate::lg2_nearest(n) >= k).unwrap();
    assert_eq!(
        crate::msm_admission::<B>(edge - 1, edge - 1, crate::MAX_MSM_LOG),
        Err("below-min-k")
    );
    assert_eq!(
        crate::msm_admission::<B>(edge, edge, crate::MAX_MSM_LOG),
        Ok(())
    );
    drop(h);

    assert!(crate::set_min_k::<B>(10));
    assert_eq!(crate::thresholds::<B>(), crate::Thresholds::all(10));
    assert_eq!(crate::thresholds::<B>().msm_one_shot, Some(10));
    assert_eq!(crate::min_k::<B>(), 10);
    assert!(crate::msm_sized::<B>(m, affine(&points[..m])));
    let mut a = input[..1 << log_n].to_vec();
    assert!(crate::fft::<B>(
        scalars_mut(&mut a),
        scalar(stock::root(log_n)),
        log_n
    ));
    assert!(!crate::set_min_k::<B>(crate::MAX_LOG_N + 1));
    crate::reset_min_k::<B>();
    assert_eq!(crate::thresholds::<B>(), d);
    assert!(crate::end_request::<B>());
}

/// A region that fails the zero read-back is released, not returned to the
/// pool.
pub fn erase_report_reads_back_every_region<B: Backend>() {
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    let fill = |n: usize| {
        let mut s = ctx::lock::<B>().unwrap();
        for (i, key) in ["test-erase-a", "test-erase-b"].into_iter().enumerate() {
            s.scratch(key, n << i).unwrap().as_mut_slice::<u8>().fill(0x5c);
        }
    };
    fill(4096 + 3);
    let w = crate::end_request_report::<B>(true).expect("every region reads back zero");
    assert_eq!(w.regions - w.pool_regions, 2);
    assert_eq!(w.checked_regions, w.regions);
    assert_eq!(w.bytes - w.pool_bytes, 3 * (4096 + 3) as u64);
    assert!(w.checked && w.nonzero_regions == 0 && w.check_failures == 0);
    assert_eq!(w.erasure_failures, 0);
    fill(64);
    let w = crate::end_request_report::<B>(false).unwrap();
    assert_eq!((w.regions, w.checked, w.checked_regions), (2, false, 0));
    assert_eq!((w.pool_regions, w.pool_bytes), (0, 0));
    // planted through the raw pointer: the buffer's dirty flag is not set
    fill(64);
    ctx::set_plant_nonzero::<B>(true);
    let r = crate::end_request_report::<B>(true);
    ctx::set_plant_nonzero::<B>(false);
    let w = r.expect_err("a non-zero region fails the report");
    assert_eq!(
        (
            w.regions - w.pool_regions,
            w.nonzero_regions,
            w.check_failures
        ),
        (2, 1, 0)
    );
    assert_eq!(ctx::lock::<B>().unwrap().scratch_bytes(), 0);
    let planted = ctx::last_planted::<B>();
    assert_ne!(planted, 0);
    assert!(
        !ctx::idle_pool_ids::<B>().contains(&planted),
        "the failed region is not in the pool"
    );
    assert!(ctx::idle_pool_reads_zero::<B>());
    assert!(ctx::request_buffers::<B>().is_empty());
    let w = crate::end_request_report::<B>(true).expect("clean afterwards");
    assert_eq!(w.nonzero_regions, 0);
}

/// A non-zero idle buffer of the pool fails the report and leaves the pool.
pub fn erase_report_reads_back_the_idle_pool<B: Backend>() {
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    let c = ctx::context::<B>().unwrap();
    let mut b = c.alloc_pooled(1 << 20).unwrap();
    b.as_mut_slice::<u8>().fill(0x42);
    drop(b);
    let idle = ctx::idle_pool_ids::<B>().len() as u64;
    assert!(idle >= 1);
    let w = crate::end_request_report::<B>(true).expect("the idle pool reads zero");
    assert_eq!(
        (w.regions, w.pool_regions, w.checked_regions),
        (idle, idle, idle)
    );
    assert_eq!(w.bytes, w.pool_bytes);
    assert!(w.pool_bytes >= 1 << 20);
    let planted = ctx::plant_in_idle_pool::<B>().expect("an idle buffer");
    let pooled = crate::pool_bytes::<B>();
    let w = crate::end_request_report::<B>(true).expect_err("a non-zero idle buffer");
    assert_eq!((w.nonzero_regions, w.check_failures), (1, 0));
    assert_eq!(w.pool_regions, idle);
    assert!(!ctx::idle_pool_ids::<B>().contains(&planted));
    assert!(
        crate::pool_bytes::<B>() < pooled,
        "the buffer left the pool"
    );
    assert!(ctx::idle_pool_reads_zero::<B>());
    assert!(crate::end_request_report::<B>(true).is_ok());
}

/// An upload shorter than the 16-byte minimum allocation reads no byte past
/// the caller's data.
pub fn short_uploads_copy_exactly_their_bytes<B: Backend>() {
    require_device::<B>();
    let c = ctx::context::<B>().unwrap();
    for len in [1usize, 4, 15] {
        let data: Vec<u8> = (1..=len as u8).collect();
        let b = c.upload(&data).unwrap();
        assert_eq!(b.len_bytes(), len);
        assert!(b.alloc_bytes() >= 16);
        let mut back = vec![0u8; len];
        b.read(&mut back);
        assert_eq!(back, data, "{len} bytes");
        let tail_zero = b.host_read_alloc(|whole| whole[len..].iter().all(|x| *x == 0)).unwrap();
        assert!(tail_zero, "the tail of {len} bytes");
    }
}

/// `settle_releases` returns, for every concurrent caller, only once every
/// earlier release has completed.
pub fn settle_releases_waits_for_every_caller<B: Backend>() {
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    let c = ctx::context::<B>().unwrap();
    for _ in 0..2 {
        let bufs: Vec<_> = (0..2)
            .map(|_| {
                let mut b = c.alloc_pooled(32 << 20).unwrap();
                b.as_mut_slice::<u8>().fill(0x11);
                b
            })
            .collect();
        ctx::release_later::<B>(bufs);
    }
    let start = std::sync::Barrier::new(2);
    std::thread::scope(|s| {
        let hs: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    start.wait();
                    ctx::settle_releases::<B>();
                    !ctx::releases_pending::<B>()
                })
            })
            .collect();
        for h in hs {
            assert!(
                h.join().unwrap(),
                "a caller returned before the zeroing ended"
            );
        }
    });
    assert!(ctx::idle_pool_reads_zero::<B>());
    assert!(crate::end_request::<B>());
}

/// A zeroing that panics is completed on the calling thread; the buffer is
/// released, not pooled, and the next report fails.
pub fn a_panicking_zeroing_is_completed_and_reported<B: Backend>() {
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    let c = ctx::context::<B>().unwrap();
    let mut b = c.alloc_pooled(1 << 20).unwrap();
    b.as_mut_slice::<u8>().fill(0x66);
    let at = b.id();
    ctx::set_zeroing_panic::<B>(at);
    ctx::release_later::<B>(vec![b]);
    ctx::settle_releases::<B>();
    ctx::set_zeroing_panic::<B>(0);
    assert!(!ctx::idle_pool_ids::<B>().contains(&at), "not pooled");
    assert!(ctx::idle_pool_reads_zero::<B>());
    let w = crate::end_request_report::<B>(false).expect_err("the fault is reported");
    assert_eq!(w.erasure_failures, 1);
    assert!(crate::end_request::<B>());
}

/// `{ENV_PREFIX}=off` declines every call `no-device`. The test reruns itself,
/// `name` (its path in the test binary), in a child process with the switch.
pub fn off_switch_declines_no_device<B: Backend>(name: &str) {
    let child = format!("{}_OFF_CHILD", B::ENV_PREFIX);
    if std::env::var_os(&child).is_some() {
        assert!(crate::present::<B>().is_none());
        assert!(!crate::device::<B>());
        assert!(crate::mem_info::<B>().is_none());
        assert!(crate::set_min_k::<B>(1));
        let mut a = vec![[0u64; 4]; 2];
        assert!(!crate::fft::<B>(&mut a, [0; 4], 1));
        assert!(crate::msm_g1::<B>(&a, &[[0u64; 12]; 2]).is_none());
        let d = crate::declines::<B>();
        assert!(d.contains(&("fft", "no-device", 1)), "{d:?}");
        assert!(d.contains(&("msm", "no-device", 1)), "{d:?}");
        return;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name])
        .env(&child, "1")
        .env(B::ENV_PREFIX, "off")
        .stdout(std::process::Stdio::null())
        .status()
        .unwrap();
    assert!(status.success());
}

/// No per-request buffer is left and every idle buffer of the pool reads
/// zero.
fn assert_wiped<B: Backend>(after: &str) {
    let left = ctx::request_buffers::<B>();
    assert!(left.is_empty(), "after {after}: {left:?}");
    assert!(
        ctx::idle_pool_reads_zero::<B>(),
        "after {after}: an idle buffer is not zero"
    );
}

/// Without a request open every hook call zeroes its per-request buffers
/// before it returns; inside one they are kept until `end_request`.
pub fn hooks_outside_a_request_wipe_before_they_return<B: Backend>() {
    use midnight_proofs::gpu::{
        self as hooks, GpuHooks,
        adapter::{affine, scalar, scalars, scalars_mut},
        test_switches::RandomCircuit,
    };
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    assert!(!ctx::request_open::<B>());
    assert!(crate::set_min_k::<B>(1));
    // two passes: data and scratch buffers
    let log_n = 12;
    let input = stock::scalars(1 << log_n, 21);
    let w = scalar(stock::root(log_n));

    let mut a = input.clone();
    assert!(crate::fft::<B>(scalars_mut(&mut a), w, log_n));
    assert_wiped::<B>("fft");
    let points = stock::points(1 << 10, 22);
    let r = crate::msm_g1::<B>(scalars(&input[..1 << 10]), affine(&points));
    assert!(r.is_some(), "{:?}", crate::declines::<B>());
    assert_wiped::<B>("msm_g1");
    assert!(hooks::install(GpuHooks {
        quotient: Some(crate::quotient::<B>),
        ..Default::default()
    }));
    let q = crate::stats::<B>().quotient.served;
    let (_, verified) = RandomCircuit::new(8, true, 9).prove();
    assert!(verified);
    assert_eq!(
        crate::stats::<B>().quotient.served - q,
        1,
        "{:?}",
        crate::declines::<B>()
    );
    assert_wiped::<B>("quotient");

    crate::begin_request::<B>();
    assert!(ctx::request_open::<B>());
    let mut a = input.clone();
    assert!(crate::fft::<B>(scalars_mut(&mut a), w, log_n));
    let first = ctx::request_buffers::<B>();
    assert!(
        first.iter().any(|r| r.0 == "ntt-data"),
        "{:?}",
        ctx::request_buffers::<B>().into_iter().map(|r| r.0).collect::<Vec<_>>()
    );
    let mut a = input.clone();
    assert!(crate::fft::<B>(scalars_mut(&mut a), w, log_n));
    let second = ctx::request_buffers::<B>();
    let addr =
        |rows: &[(&str, usize, bool, u64)]| rows.iter().find(|r| r.0 == "ntt-data").map(|r| r.3);
    assert_eq!(addr(&first), addr(&second), "the same buffer, reused");
    assert!(crate::msm_g1::<B>(scalars(&input[..1 << 10]), affine(&points)).is_some());
    assert!(
        ctx::request_buffers::<B>().len() > second.len(),
        "the msm's scratch"
    );
    assert!(crate::end_request::<B>());
    assert!(!ctx::request_open::<B>());
    assert_wiped::<B>("end_request");
    crate::reset_min_k::<B>();
}

/// A panic inside a backend is a counted decline `panic`; the caller's
/// buffer is unchanged and the call still wipes its scratch.
pub fn a_backend_panic_is_a_counted_decline<B: Backend>() {
    use midnight_proofs::gpu::adapter::{affine, scalar, scalars, scalars_mut};
    let _g = serial::<B>();
    assert!(crate::end_request::<B>());
    assert!(crate::set_min_k::<B>(1));
    let log_n = 12;
    let n = 1usize << log_n;
    let input = stock::scalars(n, 31);
    let w = stock::root(log_n);
    let wi = ff::Field::invert(&w).unwrap();
    let div = ff::Field::invert(&midnight_curves::Fq::from(n as u64)).unwrap();

    let points = stock::points(1 << 10, 32);
    let transform = |inverse: bool| {
        let mut a = input.clone();
        let ok = if inverse {
            crate::ifft::<B>(scalars_mut(&mut a), scalar(wi), log_n, scalar(div))
        } else {
            crate::fft::<B>(scalars_mut(&mut a), scalar(w), log_n)
        };
        assert_eq!(a, input, "the caller's buffer is unchanged");
        ok
    };
    // the switch panics before the job is read, so a placeholder job is safe
    let quotient = || {
        let mut out = vec![[7u64; 4]; 4];
        let job = &0u64 as *const u64 as *const std::ffi::c_void;
        // SAFETY: the test switch panics before the job is read.
        let ok = unsafe { crate::quotient::<B>(job, &mut out) };
        assert_eq!(out, vec![[7u64; 4]; 4]);
        ok
    };
    let calls: [(&str, &dyn Fn() -> bool); 4] = [
        ("fft", &|| transform(false)),
        ("ifft", &|| transform(true)),
        ("msm", &|| {
            crate::msm_g1::<B>(scalars(&input[..1 << 10]), affine(&points)).is_some()
        }),
        ("quotient", &quotient),
    ];
    let panics = crate::hook_panics::<B>();
    for (class, call) in calls {
        let before = decline_count::<B>(class, "panic");
        crate::set_hook_panic::<B>(class, true);
        let ok = call();
        crate::set_hook_panic::<B>(class, false);
        assert!(!ok, "{class}");
        assert_eq!(decline_count::<B>(class, "panic") - before, 1, "{class}");
        assert_wiped::<B>(class);
    }
    assert_eq!(crate::hook_panics::<B>() - panics, 4);
    let mut a = input[..1 << log_n].to_vec();
    assert!(crate::fft::<B>(scalars_mut(&mut a), scalar(w), log_n));
    crate::reset_min_k::<B>();
    assert!(crate::end_request::<B>());
}

/// A pipeline maximum below a planned threadgroup size would decline every
/// transform of that size.
pub fn ntt_pass_takes_the_planned_threadgroups<B: Backend>() {
    require_device::<B>();
    let max = ctx::context::<B>().unwrap().max_group("ntt_pass").unwrap();
    for log_n in 1..=crate::MAX_LOG_N {
        for t in crate::ntt::pass_group_threads(log_n) {
            assert!(
                t >= 1 && t <= max,
                "2^{log_n}: {t} threads, pipeline max {max}"
            );
        }
    }
}
