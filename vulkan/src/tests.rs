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

//! Tests of the field arithmetic against blst, in both variants, and of the
//! device context.

use ff::Field;
use midnight_curves::{Fp, Fq};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

use crate::{
    Error, Grid, Params, PipelineOptions, Variant,
    test_support::{context, run},
};

// Operation codes of shaders/field_test.comp.
const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_SQR: u32 = 3;
const OP_NEG: u32 = 4;
const OP_DBL: u32 = 5;
const OP_IS_REDUCED: u32 = 6;
const OP_ONE: u32 = 7;
const OP_EQ: u32 = 8;

/// The scalar-field modulus `r`, little-endian limbs.
const FR_MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

/// The base-field modulus `p`, little-endian limbs.
const FP_MODULUS: [u64; 6] = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

/// Whether `a < m`.
fn below<const N: usize>(a: &[u64; N], m: &[u64; N]) -> bool {
    a.iter().rev().lt(m.iter().rev())
}

/// Runs `kernel` (`field_test_fr` / `field_test_fp`) on `a`, `b`.
fn run_op<const N: usize>(
    kernel: &str,
    variant: Variant,
    op: u32,
    a: &[[u64; N]],
    b: &[[u64; N]],
) -> Vec<[u64; N]> {
    let c = context();
    let (ba, bb) = (c.upload(a).unwrap(), c.upload(b).unwrap());
    let out = c.alloc(size_of_val(a)).unwrap();
    let n = a.len() as u32;
    let params = Params::new().addr(&ba).addr(&bb).addr(&out).u32(op).u32(n);
    run(kernel, variant, &params, n as u64).unwrap();
    c.read(&out).unwrap()
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
        if below(&a, m) {
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
/// `is_reduced` against the host on reduced and unreduced values, in both
/// variants.
fn field_matches_blst<F: Field, const N: usize>(kernel: &str, m: &[u64; N], seed: u64) {
    let one = blst_op::<F, N>(OP_ONE, &[0; N], &[0; N]);
    for variant in Variant::ALL {
        for op in [
            OP_ADD, OP_SUB, OP_MUL, OP_SQR, OP_NEG, OP_DBL, OP_ONE, OP_EQ,
        ] {
            let (a, b): (Vec<_>, Vec<_>) = pairs(m, one, seed ^ op as u64).into_iter().unzip();
            let got = run_op(kernel, variant, op, &a, &b);
            for i in 0..a.len() {
                let want = blst_op::<F, N>(op, &a[i], &b[i]);
                assert_eq!(
                    got[i], want,
                    "{kernel} {variant:?} op {op}: a = {:x?}, b = {:x?}",
                    a[i], b[i]
                );
            }
        }
        let a = unreduced(m);
        let got = run_op(kernel, variant, OP_IS_REDUCED, &a, &a);
        let mut both = [0usize; 2];
        for (x, g) in a.iter().zip(&got) {
            assert_eq!(
                g[0],
                below(x, m) as u64,
                "{kernel} {variant:?} is_reduced {x:x?}"
            );
            both[g[0] as usize] += 1;
        }
        assert!(both[0] > 0 && both[1] > 0);
    }
}

#[test]
fn fr_matches_blst() {
    field_matches_blst::<Fq, 4>("field_test_fr", &FR_MODULUS, 0x5eed);
}

#[test]
fn fp_matches_blst() {
    field_matches_blst::<Fp, 6>("field_test_fp", &FP_MODULUS, 0xf00d);
}

/// A grid folded into two dimensions covers every index once, and the
/// context's errors are values.
#[test]
fn dispatch_grid_and_errors() {
    let c = context();
    let info = c.info();
    let wg = 64u32;
    // just above one row of workgroups where the device's limit is small, so
    // the grid has a second row
    let n = (info.max_compute_work_group_count[0] as u64 * wg as u64 * 2).min(1 << 26) as u32;
    let p = c.pipeline("diag_iota", Variant::Int32, PipelineOptions::workgroup(wg)).unwrap();
    let out = c.alloc(n as usize * 4 + 64).unwrap();
    c.write(&out, &vec![u32::MAX; n as usize + 16]).unwrap();
    let mut cmds = c.commands().unwrap();
    cmds.dispatch(
        &p,
        &Params::new().addr(&out).u32(n - 3),
        Grid::Threads(n as u64),
    )
    .unwrap();
    cmds.submit().unwrap();
    let got: Vec<u32> = c.read(&out).unwrap();
    assert!(got[..n as usize - 3].iter().enumerate().all(|(i, v)| *v == i as u32));
    assert!(
        got[n as usize - 3..].iter().all(|v| *v == u32::MAX),
        "a write past n"
    );

    let e = c
        .pipeline("no_such_kernel", Variant::Int32, PipelineOptions::default())
        .unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "{e}");
    let wide = PipelineOptions::workgroup(info.max_compute_work_group_size[0] + 1);
    let e = c.pipeline("diag_iota", Variant::Int32, wide).unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "{e}");
    // the subgroup width the MSM kernels pin
    let pinned = PipelineOptions {
        subgroup: Some(32),
        ..PipelineOptions::workgroup(64)
    };
    let p32 = c.pipeline("diag_iota", Variant::Int32, pinned).unwrap();
    let mut cmds = c.commands().unwrap();
    cmds.dispatch(&p32, &Params::new().addr(&out).u32(100), Grid::Threads(100))
        .unwrap();
    cmds.submit().unwrap();
    let odd = PipelineOptions {
        subgroup: Some(3),
        ..PipelineOptions::workgroup(64)
    };
    let e = c.pipeline("diag_iota", Variant::Int32, odd).unwrap_err();
    assert!(
        matches!(e, Error::Invalid(_) | Error::Unsupported(_)),
        "{e}"
    );
    let e = c.alloc(usize::MAX / 2).unwrap_err();
    assert_eq!(e, Error::OutOfMemory);
    let mut cmds = c.commands().unwrap();
    let e = cmds.dispatch(&p, &Params::new().words(&[0; 33]), Grid::Threads(1)).unwrap_err();
    assert!(matches!(e, Error::Invalid(_)), "{e}");
    drop(cmds);
    assert!(!c.failed());
}
