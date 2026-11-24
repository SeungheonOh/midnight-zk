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

//! Helpers shared by the integration tests: the host limbs of
//! midnight-curves values and XYZZ records.

#![allow(dead_code)]

use blst::{blst_fp, blst_fp_mul};
use midnight_curves::{Fq, G1Affine, G1Projective};

/// An XYZZ record: `X`, `Y`, `ZZZ`, `ZZ` in `Fp` limbs.
pub type Xyzz = [u64; 24];

/// The scalar-field modulus `r`, little-endian limbs.
pub const FR_MODULUS: [u64; 4] = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

/// Whether `a < r`.
pub fn fr_is_reduced(a: &[u64; 4]) -> bool {
    a.iter().rev().lt(FR_MODULUS.iter().rev())
}

/// The limbs of an affine point (`blst_p1_affine`: `x`, `y`).
pub fn affine(a: &G1Affine) -> [u64; 12] {
    const { assert!(size_of::<G1Affine>() == size_of::<[u64; 12]>()) };
    // SAFETY: `G1Affine` is a transparent `blst_p1_affine`, `repr(C)` of two
    // `[u64; 6]` (size checked above).
    unsafe { std::mem::transmute_copy(a) }
}

/// The point of Jacobian limbs (`blst_p1` layout, a point on the curve).
pub fn from_jacobian(a: [u64; 18]) -> G1Projective {
    const { assert!(size_of::<G1Projective>() == size_of::<[u64; 18]>()) };
    // SAFETY: `G1Projective` is a transparent `blst_p1`, `repr(C)` of three
    // `[u64; 6]` (size checked above).
    unsafe { std::mem::transmute_copy(&a) }
}

/// The limbs of a scalar (`blst_fr`).
pub fn scalar(a: &Fq) -> [u64; 4] {
    const { assert!(size_of::<Fq>() == size_of::<[u64; 4]>()) };
    // SAFETY: `Fq` is a transparent `blst_fr`, `[u64; 4]` (size checked
    // above).
    unsafe { std::mem::transmute_copy(a) }
}

pub fn fp(l: &[u64]) -> blst_fp {
    blst_fp {
        l: l.try_into().unwrap(),
    }
}

pub fn fmul(a: &blst_fp, b: &blst_fp) -> blst_fp {
    let mut r = blst_fp::default();
    // SAFETY: valid field elements.
    unsafe { blst_fp_mul(&mut r, a, b) };
    r
}

pub fn fsqr(a: &blst_fp) -> blst_fp {
    fmul(a, a)
}

/// An XYZZ record as a point, checking `ZZ^3 = ZZZ^2`.
pub fn xyzz_point(r: &Xyzz) -> G1Projective {
    let (x, y, zzz, zz) = (fp(&r[..6]), fp(&r[6..12]), fp(&r[12..18]), fp(&r[18..]));
    let zero = |f: &blst_fp| f.l.iter().all(|&w| w == 0);
    assert_eq!(zero(&zz), zero(&zzz), "ZZ and ZZZ disagree on infinity");
    assert_eq!(fmul(&fsqr(&zz), &zz).l, fsqr(&zzz).l, "ZZ^3 != ZZZ^2");
    let mut j = [0u64; 18];
    j[..6].copy_from_slice(&fmul(&x, &zz).l);
    j[6..12].copy_from_slice(&fmul(&y, &zzz).l);
    j[12..].copy_from_slice(&zz.l);
    from_jacobian(j)
}

/// The limbs of scalars.
pub fn scalars(a: &[Fq]) -> Vec<[u64; 4]> {
    a.iter().map(scalar).collect()
}

/// The CUDA crate's mask vector (gpu/src/msm.rs `mask_vector`) from
/// element `i0` on: the splitmix64 stream advanced by `4 i0` outputs.
pub fn cuda_mask_vector(i0: u64, n: usize) -> Vec<[u64; 4]> {
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

/// The scalar of reduced Montgomery limbs.
pub fn fq(a: &[u64; 4]) -> Fq {
    // SAFETY: as `scalar`; `a` is reduced.
    unsafe { std::mem::transmute_copy(a) }
}
