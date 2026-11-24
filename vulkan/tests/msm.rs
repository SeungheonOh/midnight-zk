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

//! The G1 curve kernels and the MSM mask against blst, in both variants.

mod common;

use blst::blst_fp;
use common::{Xyzz, affine, fmul, fp, fsqr, xyzz_point};
use group::{Curve, Group, prime::PrimeCurveAffine};
use midnight_curves::{G1Affine, G1Projective};
use midnight_vulkan::{
    Params, Variant,
    test_support::{context, run},
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

// Operation codes of `msm_test_g1` (shaders/msm_test_g1.comp).
const OP_MADD: u32 = 0;
const OP_ADD: u32 = 1;
const OP_FROM_AFFINE: u32 = 2;

/// `(x z^2, y z^3, z^3, z^2)`; infinity is all zero.
fn xyzz_of(a: &G1Affine, z: &blst_fp) -> Xyzz {
    let a = affine(a);
    let mut r = [0u64; 24];
    if a.iter().all(|&w| w == 0) {
        return r;
    }
    let zz = fsqr(z);
    let zzz = fmul(&zz, z);
    r[..6].copy_from_slice(&fmul(&fp(&a[..6]), &zz).l);
    r[6..12].copy_from_slice(&fmul(&fp(&a[6..]), &zzz).l);
    r[12..18].copy_from_slice(&zzz.l);
    r[18..].copy_from_slice(&zz.l);
    r
}

fn g1_kernel(
    variant: Variant,
    op: u32,
    a: &[Xyzz],
    b: &[Xyzz],
    b_aff: &[[u64; 12]],
    flags: &[u32],
) -> Vec<Xyzz> {
    let n = a.len() as u32;
    let c = context();
    let bufs = [
        c.upload(a).unwrap(),
        c.upload(b).unwrap(),
        c.upload(b_aff).unwrap(),
        c.upload(flags).unwrap(),
        c.alloc(size_of_val(a)).unwrap(),
    ];
    let params = bufs.iter().fold(Params::new(), |p, b| p.addr(b)).u32(op).u32(n);
    run("msm_test_g1", variant, &params, n as u64).unwrap();
    c.read(&bufs[4]).unwrap()
}

#[test]
fn g1_kernels_match_blst() {
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let pts: Vec<G1Affine> = (0..64).map(|_| G1Projective::random(&mut rng).to_affine()).collect();
    let inf = G1Affine::identity();
    let mut pairs: Vec<(G1Affine, G1Affine)> = (0..32).map(|i| (pts[i], pts[i + 32])).collect();
    for p in &pts[..4] {
        pairs.extend([(*p, inf), (inf, *p), (*p, *p), (*p, -*p)]);
    }
    pairs.push((inf, inf));
    let z = |i: usize| fp(&affine(&pts[(i + 17) % pts.len()])[..6]);
    let a: Vec<Xyzz> = pairs.iter().enumerate().map(|(i, p)| xyzz_of(&p.0, &z(i))).collect();
    // b in XYZZ with a different scale, so P + P is not a bitwise copy
    let b: Vec<Xyzz> = pairs.iter().enumerate().map(|(i, p)| xyzz_of(&p.1, &z(i + 5))).collect();
    let b_aff: Vec<[u64; 12]> = pairs.iter().map(|p| affine(&p.1)).collect();
    for variant in Variant::ALL {
        for (op, neg) in [
            (OP_MADD, false),
            (OP_MADD, true),
            (OP_FROM_AFFINE, false),
            (OP_FROM_AFFINE, true),
            (OP_ADD, false),
        ] {
            let flags = vec![neg as u32; pairs.len()];
            let got = g1_kernel(variant, op, &a, &b, &b_aff, &flags);
            for (i, (p, q)) in pairs.iter().enumerate() {
                let q = if neg { -q.to_curve() } else { q.to_curve() };
                let want = if op == OP_FROM_AFFINE {
                    q
                } else {
                    p.to_curve() + q
                };
                assert!(
                    xyzz_point(&got[i]) == want,
                    "{variant:?} op {op}, negated {neg}, pair {i}"
                );
                if op == OP_ADD && *p != inf && q == -p.to_curve() {
                    assert_eq!(got[i], [0u64; 24], "{variant:?} P + (-P) at {i}");
                }
            }
        }
    }
}

/// The CUDA crate's mask vector (gpu/src/msm.rs `mask_vector`) from
/// element `i0` on: the splitmix64 stream advanced by `4 i0` outputs.
fn cuda_mask_vector(i0: u64, n: usize) -> Vec<[u64; 4]> {
    const GAMMA: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut x: u64 = 0x2545_f491_4f6c_dd1du64.wrapping_add((4 * i0).wrapping_mul(GAMMA));
    let mut next = || {
        x = x.wrapping_add(GAMMA);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    (0..n).map(|_| [next(), next(), next(), next() >> 3]).collect()
}

/// The mask from `i0` on, `n` elements.
fn mask_kernel(variant: Variant, i0: u32, n: u32) -> Vec<[u64; 4]> {
    let c = context();
    let out = c.alloc(n as usize * 32).unwrap();
    run(
        "msm_test_mask",
        variant,
        &Params::new().addr(&out).u32(i0).u32(n),
        n as u64,
    )
    .unwrap();
    c.read(&out).unwrap()
}

#[test]
fn mask_is_the_cuda_vector() {
    let (i0, n) = (100u32, 5000u32);
    let want = cuda_mask_vector(0, (i0 + n) as usize);
    assert!(want.iter().all(common::fr_is_reduced));
    // indices across 2^32, where the 64-bit index carries
    let far = u32::MAX - 2;
    for variant in Variant::ALL {
        assert_eq!(
            mask_kernel(variant, i0, n),
            &want[i0 as usize..],
            "{variant:?}"
        );
        assert_eq!(
            mask_kernel(variant, far, 6),
            cuda_mask_vector(far as u64, 6),
            "{variant:?} across 2^32"
        );
    }
}
