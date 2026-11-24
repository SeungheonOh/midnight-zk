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
//! midnight-curves values.

#![allow(dead_code)]

use midnight_curves::{Fq, G1Affine, G1Projective};

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
