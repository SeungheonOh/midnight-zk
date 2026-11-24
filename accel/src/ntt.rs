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

//! The host side of the scalar-field NTT that needs no device: the standard
//! roots of unity, the twiddle powers, the pass plan and the pass parameters
//! of the `ntt_pass` kernel.

use std::sync::OnceLock;

use blst::{blst_fr, blst_fr_inverse, blst_fr_mul, blst_fr_sqr};

use crate::{MAX_LOG_N, Scalar, ctx::Pod};

/// Bytes of one scalar.
pub const SCALAR_BYTES: usize = 32;

/// `2^S` is the largest power of two dividing `r - 1`.
pub const S: u32 = 32;

/// `ROOT_OF_UNITY` of `midnight_curves::Fq`, Montgomery limbs.
pub const ROOT_OF_UNITY: Scalar = [
    0xb9b5_8d8c_5f0e_466a,
    0x5b1b_4c80_1819_d7ec,
    0x0af5_3ae3_52a3_1e64,
    0x5bf3_adda_19e9_b27b,
];

/// The factors of the passes are powers of the `2^UNIVERSAL_LOG`-th root.
pub const UNIVERSAL_LOG: u32 = 28;
const _: () = assert!(MAX_LOG_N <= UNIVERSAL_LOG);

/// Log2 of the length of the high twiddle table (`w_{2^HI_BITS}^j`); the
/// low table (`w_{2^28}^j`) has `2^(28 - HI_BITS)` elements.
pub const HI_BITS: u32 = 16;

/// Bytes of the scaled low twiddle table a transform with a divisor
/// allocates; the quotient's memory need counts it.
pub const SCALE_TABLE_BYTES: usize = SCALAR_BYTES << (UNIVERSAL_LOG - HI_BITS);

/// The largest radix (log2) of a pass: `2^MAX_PASS_LOG` elements fill a
/// threadgroup's 32 KiB.
pub const MAX_PASS_LOG: u32 = 10;

/// Log2 of the elements one threadgroup holds (radix times columns).
pub const GROUP_LOG: u32 = 10;

/// The direction of a transform.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    /// At the standard root `w_n`.
    Forward,
    /// At `w_n^-1`.
    Inverse,
}

impl Dir {
    /// The index of the direction in the root and twiddle tables.
    pub fn index(self) -> usize {
        match self {
            Dir::Forward => 0,
            Dir::Inverse => 1,
        }
    }
}

// ---------------------------------------------------------------------------
// Host field operations (blst)

/// `a * b` in `Fr` (Montgomery limbs).
pub fn mul(a: &Scalar, b: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: valid pointers to blst_fr values.
    unsafe { blst_fr_mul(&mut r, &blst_fr { l: *a }, &blst_fr { l: *b }) };
    r.l
}

/// `a^2` in `Fr`.
pub fn square(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: as `mul`.
    unsafe { blst_fr_sqr(&mut r, &blst_fr { l: *a }) };
    r.l
}

/// `a^-1` in `Fr` (`a` non-zero).
pub fn invert(a: &Scalar) -> Scalar {
    let mut r = blst_fr::default();
    // SAFETY: as `mul`; the argument is a root of unity, never zero.
    unsafe { blst_fr_inverse(&mut r, &blst_fr { l: *a }) };
    r.l
}

/// `roots[dir][m]`: the standard `2^m`-th root of unity (`m <= 28`) or its
/// inverse.
pub fn roots() -> &'static [[Scalar; UNIVERSAL_LOG as usize + 1]; 2] {
    static ROOTS: OnceLock<[[Scalar; UNIVERSAL_LOG as usize + 1]; 2]> = OnceLock::new();
    ROOTS.get_or_init(|| {
        let mut w = ROOT_OF_UNITY;
        for _ in UNIVERSAL_LOG..S {
            w = square(&w);
        }
        let mut fwd = [[0u64; 4]; UNIVERSAL_LOG as usize + 1];
        let mut inv = fwd;
        let mut wi = invert(&w);
        for m in (0..=UNIVERSAL_LOG as usize).rev() {
            fwd[m] = w;
            inv[m] = wi;
            w = square(&w);
            wi = square(&wi);
        }
        [fwd, inv]
    })
}

/// The standard `2^log_n`-th root of unity of `Fr` (`dir` forward) or its
/// inverse; `None` above `2^28`.
pub fn std_root(log_n: u32, dir: Dir) -> Option<Scalar> {
    (log_n <= UNIVERSAL_LOG).then(|| roots()[dir.index()][log_n as usize])
}

/// The direction of `omega` for a `2^log_n` transform, if it is the standard
/// root or its inverse.
pub fn direction(omega: &Scalar, log_n: u32) -> Option<Dir> {
    if std_root(log_n, Dir::Forward).as_ref() == Some(omega) {
        Some(Dir::Forward)
    } else if std_root(log_n, Dir::Inverse).as_ref() == Some(omega) {
        Some(Dir::Inverse)
    } else {
        None
    }
}

/// `[1, w, w^2, ..., w^(len - 1)]`.
pub fn powers(w: &Scalar, len: usize) -> Vec<Scalar> {
    let mut v = Vec::with_capacity(len);
    // Montgomery one: 2^256 mod r
    let mut acc: Scalar = [
        0x0000_0001_ffff_fffe,
        0x5884_b7fa_0003_4802,
        0x998c_4fef_ecbc_4ff5,
        0x1824_b159_acc5_056f,
    ];
    for _ in 0..len {
        v.push(acc);
        acc = mul(&acc, w);
    }
    v
}

/// The parameters of one pass (`ntt::PassParams` of the shader).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PassParams {
    pub log_n: u32,
    pub log_r: u32,
    pub log_p: u32,
    pub log_g: u32,
    pub flags: u32,
    pub n_src: u32,
    pub hi_bits: u32,
    pub pad: u32,
    pub z1: Scalar,
    pub z2: Scalar,
}
// SAFETY: `repr(C)`, eight `u32` then two `[u64; 4]`: no padding, every bit
// pattern valid.
unsafe impl Pod for PassParams {}

/// Pass flag: multiply the inputs by the pass factors.
pub const F_TWIDDLE: u32 = 1;
/// Pass flag: the factors read the high and the low table.
pub const F_TWO_LEVEL: u32 = 2;
/// Pass flag: the input is `coeff_to_extended`'s (distributed, zero-padded).
pub const F_COSET: u32 = 4;

/// The radices (log2) of the passes of a `2^log_n` transform: as few passes
/// as the threadgroup memory allows, of near-equal radices (larger first).
pub fn plan(log_n: u32) -> Vec<u32> {
    let passes = log_n.div_ceil(MAX_PASS_LOG).max(1);
    let base = log_n / passes;
    let extra = log_n % passes;
    (0..passes).map(|i| base + u32::from(i < extra)).collect()
}

/// Log2 of the columns of one threadgroup of a pass of radix `2^r` in a
/// `2^log_n` transform.
pub fn group_columns_log(log_n: u32, r: u32) -> u32 {
    GROUP_LOG.saturating_sub(r).min(log_n - r)
}

/// Threads of one threadgroup of a pass holding `2^e_log` elements: one per
/// four elements, the measured best.
pub fn group_threads(e_log: u32) -> usize {
    1usize << e_log.saturating_sub(2)
}

/// The threadgroup size of each `ntt_pass` dispatch of a `2^log_n` transform.
pub fn pass_group_threads(log_n: u32) -> Vec<usize> {
    plan(log_n)
        .into_iter()
        .map(|r| group_threads(r + group_columns_log(log_n, r)))
        .collect()
}

/// Scratch bytes of a [`ntt_dev`] or [`coeff_to_extended_dev_at`] of `2^log_n`:
/// none for one pass (`log_n <= 10`), else the transform's length.
pub fn scratch_bytes(log_n: u32) -> usize {
    if plan(log_n).len() > 1 {
        SCALAR_BYTES << log_n
    } else {
        0
    }
}

