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

//! The BLS12-381 moduli on host limbs and the reduction check every entry
//! point applies. Arithmetic on the host goes through `blst`.

use crate::Scalar;

/// A base-field element: `blst_fp`, six limbs, Montgomery form.
pub type FpLimbs = [u64; 6];

/// The scalar-field modulus `r`, little-endian limbs.
pub const FR_MODULUS: Scalar = [
    0xffff_ffff_0000_0001,
    0x53bd_a402_fffe_5bfe,
    0x3339_d808_09a1_d805,
    0x73ed_a753_299d_7d48,
];

/// The base-field modulus `p`, little-endian limbs.
pub const FP_MODULUS: FpLimbs = [
    0xb9fe_ffff_ffff_aaab,
    0x1eab_fffe_b153_ffff,
    0x6730_d2a0_f6b0_f624,
    0x6477_4b84_f385_12bf,
    0x4b1b_a7b6_434b_acd7,
    0x1a01_11ea_397f_e69a,
];

fn below<const N: usize>(a: &[u64; N], m: &[u64; N]) -> bool {
    for i in (0..N).rev() {
        if a[i] != m[i] {
            return a[i] < m[i];
        }
    }
    false
}

/// Whether `a < r`.
#[inline]
pub fn is_reduced(a: &Scalar) -> bool {
    below(a, &FR_MODULUS)
}

/// Whether `a < p`.
#[inline]
pub fn fp_is_reduced(a: &FpLimbs) -> bool {
    below(a, &FP_MODULUS)
}

/// Whether every scalar is below `r` (in parallel for long slices).
pub fn all_reduced(a: &[Scalar]) -> bool {
    const PART: usize = 1 << 16;
    if a.len() <= PART {
        return a.iter().all(is_reduced);
    }
    let parts = crate::msm::par::par_parts(a.len(), crate::msm::par::host_threads(), PART, |lo, hi| {
        a[lo..hi].iter().all(is_reduced)
    });
    parts.is_some_and(|ok| ok.into_iter().all(|b| b))
}
