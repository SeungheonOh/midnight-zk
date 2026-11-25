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

//! BLS12-381 G1 GLV batched-affine Pippenger MSM for the CPU share.
//!
//! Every base entering [`msm_glv`] is already verified to lie in the
//! prime-order subgroup `G1` (guaranteed by table registration or the one-shot
//! subgroup check), so the endomorphism `phi(x, y) = (beta * x, y) = lambda * P`
//! with `lambda = -u^2 mod r` (`u = -0xd201000000010000`) holds unconditionally.
//! Since `r = u^4 - u^2 + 1`, dividing any `s in [0, r)` by `u^2` gives exact
//! 128-bit unsigned half-scalars `s = k1 - k2 * lambda (mod r)` with
//! `0 <= k1, k2 < u^2 < 2^128`, so `s * P = k1 * P + k2 * (beta * x, -y)`.
//!
//! Splitting 255-bit scalars into 128-bit half-scalars doubles the effective
//! point count per window, allowing wider windows (`c = 13..14`, 9–10 windows
//! vs 22 without GLV) and ~17% fewer batched-affine curve additions, using
//! `blst`'s assembly field arithmetic directly.

use std::sync::{
    OnceLock,
    atomic::{AtomicUsize, Ordering},
};

use blst::{
    blst_fp, blst_fp_add, blst_fp_cneg, blst_fp_eucl_inverse, blst_fp_mul, blst_fp_sqr,
    blst_fp_sub, blst_fr, blst_p1, blst_p1_add_or_double, blst_p1_double, blst_scalar,
    blst_scalar_from_fr,
};
use rayon::prelude::*;

use super::secret::{ScrubOnExit, Secret, wipe, wipe_slice};
use crate::{G1AffineLimbs, Scalar};

/// `u^2 = (-0xd201000000010000)^2`, exact 128-bit GLV divisor (`r = u^4 - u^2 + 1`).
const U2: u128 = 0xac45_a401_0001_a402_0000_0001_0000_0000;
/// `U2 / 2 < 0.673 * 2^127`, upper bound on signed GLV half-scalar magnitudes.
const HALF_U2: u128 = U2 >> 1;
/// Sign bit stored in bit 127 of each encoded half-scalar (`|k| <= HALF_U2 < 2^127`).
const SIGN_BIT: u128 = 1u128 << 127;
/// Low 128 bits of `floor(2^256 / U2) = 2^128 + MU_LO`.
const MU_LO: u128 = 0x7c6b_ecf1_e01f_aadd_63f6_e522_f6cf_ee2e;

/// Montgomery `1` in BLS12-381 `Fp`: `2^384 mod p`.
const FP_ONE: blst_fp = blst_fp {
    l: [
        0x7609_0000_0002_fffd,
        0xebf4_000b_c40c_0002,
        0x5f48_9857_53c7_58ba,
        0x77ce_5853_7052_5745,
        0x5c07_1a97_a256_ec6d,
        0x15f6_5ec3_fa80_e493,
    ],
};

/// Cube root of unity `beta` in Montgomery form (`beta * 2^384 mod p`) such
/// that `lambda * (x, y) = (beta * x, y)` for `lambda = -u^2 mod r` on G1.
const BETA_MONT: blst_fp = blst_fp {
    l: [
        0x30f1_361b_798a_64e8,
        0xf3b8_ddab_7ece_5a2a,
        0x16a8_ca3a_c615_77f7,
        0xc26a_2ff8_74fd_029b,
        0x3636_b766_6070_1c6e,
        0x051b_a4ab_241b_6160,
    ],
};

const NBITS: usize = 127;
const LANES: usize = 256;
/// Scalars decomposed together lazily by whichever tile visits a block first.
const SCALAR_BLOCK: usize = 512;

/// Lazily initialized blocks of signed 127-bit half-scalars; every initialized
/// block is wiped on drop.
struct SecretBlocks(Vec<OnceLock<Vec<(u128, u128)>>>);

impl Drop for SecretBlocks {
    fn drop(&mut self) {
        for cell in &mut self.0 {
            if let Some(v) = cell.get_mut() {
                wipe_slice(v);
            }
        }
    }
}

#[inline(always)]
fn fp_zero() -> blst_fp {
    blst_fp { l: [0u64; 6] }
}

#[inline(always)]
fn fp_is_zero(a: &blst_fp) -> bool {
    (a.l[0] | a.l[1] | a.l[2] | a.l[3] | a.l[4] | a.l[5]) == 0
}

#[inline(always)]
fn fp_add(ret: &mut blst_fp, a: &blst_fp, b: &blst_fp) {
    // SAFETY: valid blst_fp pointers.
    unsafe { blst_fp_add(ret, a, b) };
}

#[inline(always)]
fn fp_add_assign(a: &mut blst_fp, b: &blst_fp) {
    let p = a as *mut blst_fp;
    // SAFETY: blst_fp_add supports in-place aliasing (`ret == a`).
    unsafe { blst_fp_add(p, p, b) };
}

#[inline(always)]
fn fp_sub(ret: &mut blst_fp, a: &blst_fp, b: &blst_fp) {
    unsafe { blst_fp_sub(ret, a, b) };
}

#[inline(always)]
fn fp_sub_assign(a: &mut blst_fp, b: &blst_fp) {
    let p = a as *mut blst_fp;
    // SAFETY: blst_fp_sub supports in-place aliasing (`ret == a`).
    unsafe { blst_fp_sub(p, p, b) };
}

#[inline(always)]
fn fp_rsub_assign(a: &mut blst_fp, b: &blst_fp) {
    let p = a as *mut blst_fp;
    // SAFETY: blst_fp_sub supports in-place aliasing (`ret == b`).
    unsafe { blst_fp_sub(p, b, p) };
}

#[inline(always)]
fn fp_mul(ret: &mut blst_fp, a: &blst_fp, b: &blst_fp) {
    unsafe { blst_fp_mul(ret, a, b) };
}

#[inline(always)]
fn fp_mul_assign(a: &mut blst_fp, b: &blst_fp) {
    let p = a as *mut blst_fp;
    // SAFETY: blst_fp_mul supports in-place aliasing (`ret == a`).
    unsafe { blst_fp_mul(p, p, b) };
}

#[inline(always)]
fn fp_sqr(ret: &mut blst_fp, a: &blst_fp) {
    unsafe { blst_fp_sqr(ret, a) };
}

#[inline(always)]
fn fp_cneg(ret: &mut blst_fp, a: &blst_fp, flag: bool) {
    unsafe { blst_fp_cneg(ret, a, flag) };
}

#[inline(always)]
fn fp_inv(ret: &mut blst_fp, a: &blst_fp) {
    unsafe { blst_fp_eucl_inverse(ret, a) };
}

#[inline(always)]
fn mul_u128_wide(a: u128, b: u128) -> (u128, u128) {
    let (a0, a1) = (a as u64 as u128, a >> 64);
    let (b0, b1) = (b as u64 as u128, b >> 64);
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let (mid, c1) = lh.overflowing_add(hl);
    let (lo, c2) = ll.overflowing_add(mid << 64);
    let hi = hh + (mid >> 64) + ((c1 as u128) << 64) + (c2 as u128);
    (lo, hi)
}

/// Decomposes Montgomery scalar `s` into signed 127-bit half-scalars
/// `(k1 | (neg1 << 127), k2 | (neg2 << 127))` with `0 <= k1, k2 <= U2 / 2 < 2^127`
/// such that `s * P = (±k1) * P + (±k2) * phi(P)`, where `phi(x, y) = (beta * x, y)`.
#[inline(always)]
fn decompose(s: &Scalar) -> (u128, u128) {
    let mut sc = blst_scalar::default();
    // SAFETY: `Scalar` is `[u64; 4]`, identical to `blst_fr`.
    unsafe { blst_scalar_from_fr(&mut sc, &blst_fr { l: *s }) };
    let s_lo = u128::from_le_bytes(sc.b[..16].try_into().unwrap());
    let s_hi = u128::from_le_bytes(sc.b[16..].try_into().unwrap());
    let (_, q_add) = mul_u128_wide(s_hi, MU_LO);
    let mut q = s_hi + q_add;
    let (prod_lo, prod_hi) = mul_u128_wide(q, U2);
    let (mut rem, borrow) = s_lo.overflowing_sub(prod_lo);
    let mut rem_hi = s_hi.wrapping_sub(prod_hi).wrapping_sub(borrow as u128);
    while rem_hi != 0 || rem >= U2 {
        let (r2, b) = rem.overflowing_sub(U2);
        rem = r2;
        rem_hi = rem_hi.wrapping_sub(b as u128);
        q += 1;
    }
    // Center `rem` into [-HALF_U2, HALF_U2] using (U2, -1):
    // `s = rem + q * U2 = (±k1) + q' * U2`, with `0 <= q' <= U2 - 1` since
    // `r = U2 * (U2 - 1) + 1` implies `rem == 0` whenever `q == U2 - 1`.
    let (mut k1, mut neg1) = if rem > HALF_U2 {
        q += 1;
        (U2 - rem, true)
    } else {
        (rem, false)
    };
    // Center `q` into [-HALF_U2, HALF_U2] using `r = U2 * (U2 - 1) + 1`:
    // `+q * U2 = -q * lambda` (neg2 = true);
    // when `q > HALF_U2`: `q * U2 ≡ -(U2 - 1 - q) * U2 - 1 = +(U2 - 1 - q) * lambda - 1 (mod r)`.
    let (k2, neg2) = if q > HALF_U2 {
        if neg1 {
            k1 += 1;
        } else if k1 == 0 {
            k1 = 1;
            neg1 = true;
        } else {
            k1 -= 1;
        }
        ((U2 - 1) - q, false)
    } else {
        (q, true)
    };
    (
        k1 | if neg1 { SIGN_BIT } else { 0 },
        k2 | if neg2 { SIGN_BIT } else { 0 },
    )
}

/// Signed Booth digit of window `w` (width `c`) of a 127-bit unsigned magnitude,
/// in `-2^(c-1)..=2^(c-1)`.
#[inline(always)]
fn booth_u128(k: u128, w: usize, c: usize) -> i32 {
    let shifted = if w == 0 {
        (k as u32) << 1
    } else {
        let bit = w * c - 1;
        if bit >= NBITS {
            return 0;
        }
        (k >> bit) as u32
    };
    let tmp = shifted & ((1u32 << (c + 1)) - 1);
    let sign = (tmp & (1u32 << c)) == 0;
    let mag = (tmp + 1) >> 1;
    if sign {
        mag as i32
    } else {
        -(((1u32 << c) - mag) as i32)
    }
}

#[derive(Clone, Copy)]
struct Xyzz {
    x: blst_fp,
    y: blst_fp,
    zz: blst_fp,
    zzz: blst_fp,
}

impl Xyzz {
    #[inline(always)]
    fn identity() -> Self {
        Self {
            x: fp_zero(),
            y: fp_zero(),
            zz: fp_zero(),
            zzz: fp_zero(),
        }
    }

    #[inline(always)]
    fn is_identity(&self) -> bool {
        fp_is_zero(&self.zz) && fp_is_zero(&self.zzz)
    }

    fn from_affine(x: &blst_fp, y: &blst_fp) -> Self {
        Self {
            x: *x,
            y: *y,
            zz: FP_ONE,
            zzz: FP_ONE,
        }
    }

    fn double(&mut self) {
        if self.is_identity() {
            return;
        }
        let (mut u, mut v, mut w, mut s, mut xsq, mut m) = (
            fp_zero(),
            fp_zero(),
            fp_zero(),
            fp_zero(),
            fp_zero(),
            fp_zero(),
        );
        fp_add(&mut u, &self.y, &self.y);
        fp_sqr(&mut v, &u);
        fp_mul(&mut w, &v, &u);
        fp_mul(&mut s, &self.x, &v);
        fp_sqr(&mut xsq, &self.x);
        fp_add(&mut m, &xsq, &xsq);
        fp_add_assign(&mut m, &xsq);
        fp_sqr(&mut self.x, &m);
        fp_sub_assign(&mut self.x, &s);
        fp_sub_assign(&mut self.x, &s);
        fp_sub_assign(&mut s, &self.x);
        fp_mul_assign(&mut s, &m);
        fp_mul_assign(&mut self.y, &w);
        fp_rsub_assign(&mut self.y, &s);
        fp_mul_assign(&mut self.zz, &v);
        fp_mul_assign(&mut self.zzz, &w);
    }

    fn add_affine(&mut self, x2: &blst_fp, y2: &blst_fp, negate: bool) {
        let mut ny2 = fp_zero();
        let y2 = if negate {
            fp_cneg(&mut ny2, y2, true);
            &ny2
        } else {
            y2
        };
        if self.is_identity() {
            *self = Self::from_affine(x2, y2);
            return;
        }
        let (mut p, mut r) = (fp_zero(), fp_zero());
        fp_mul(&mut p, x2, &self.zz);
        fp_sub_assign(&mut p, &self.x);
        fp_mul(&mut r, y2, &self.zzz);
        fp_sub_assign(&mut r, &self.y);
        if !fp_is_zero(&p) {
            let (mut pp, mut ppp, mut q) = (fp_zero(), fp_zero(), fp_zero());
            fp_sqr(&mut pp, &p);
            fp_mul(&mut ppp, &pp, &p);
            fp_mul(&mut q, &self.x, &pp);
            fp_sqr(&mut self.x, &r);
            fp_sub_assign(&mut self.x, &ppp);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut q, &self.x);
            fp_mul_assign(&mut q, &r);
            fp_mul_assign(&mut self.y, &ppp);
            fp_rsub_assign(&mut self.y, &q);
            fp_mul_assign(&mut self.zz, &pp);
            fp_mul_assign(&mut self.zzz, &ppp);
        } else if fp_is_zero(&r) {
            *self = Self::from_affine(x2, y2);
            self.double();
        } else {
            *self = Self::identity();
        }
    }

    fn add(&mut self, other: &Self) {
        if other.is_identity() {
            return;
        }
        if self.is_identity() {
            *self = *other;
            return;
        }
        let (mut u1, mut s1, mut p, mut r) = (fp_zero(), fp_zero(), fp_zero(), fp_zero());
        fp_mul(&mut u1, &self.x, &other.zz);
        fp_mul(&mut s1, &self.y, &other.zzz);
        fp_mul(&mut p, &other.x, &self.zz);
        fp_sub_assign(&mut p, &u1);
        fp_mul(&mut r, &other.y, &self.zzz);
        fp_sub_assign(&mut r, &s1);
        if !fp_is_zero(&p) {
            let (mut pp, mut ppp, mut q) = (fp_zero(), fp_zero(), fp_zero());
            fp_sqr(&mut pp, &p);
            fp_mul(&mut ppp, &pp, &p);
            fp_mul(&mut q, &u1, &pp);
            fp_sqr(&mut self.x, &r);
            fp_sub_assign(&mut self.x, &ppp);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut self.x, &q);
            fp_sub_assign(&mut q, &self.x);
            fp_mul_assign(&mut q, &r);
            fp_mul_assign(&mut s1, &ppp);
            fp_sub(&mut self.y, &q, &s1);
            fp_mul_assign(&mut self.zz, &other.zz);
            fp_mul_assign(&mut self.zz, &pp);
            fp_mul_assign(&mut self.zzz, &other.zzz);
            fp_mul_assign(&mut self.zzz, &ppp);
        } else if fp_is_zero(&r) {
            self.double();
        } else {
            *self = Self::identity();
        }
    }

    /// Converts XYZZ `(X, Y, ZZ, ZZZ)` to Jacobian `blst_p1` `(X*ZZ, Y*ZZZ, ZZ)`
    /// without any field inversion.
    fn to_p1(self) -> blst_p1 {
        let mut p = blst_p1 {
            z: self.zz,
            ..Default::default()
        };
        fp_mul(&mut p.x, &self.x, &self.zz);
        fp_mul(&mut p.y, &self.y, &self.zzz);
        p
    }
}

#[derive(Clone, Copy)]
struct AffineBucket {
    x: blst_fp,
    y: blst_fp,
    full: bool,
    queued: bool,
}

impl AffineBucket {
    fn empty() -> Self {
        Self {
            x: fp_zero(),
            y: fp_zero(),
            full: false,
            queued: false,
        }
    }
}

#[derive(Clone, Copy)]
struct Pending {
    bucket: u32,
    point: u32, // (base_idx << 1) | is_phi
    negate: bool,
    same_x: bool,
}

#[inline(always)]
fn invert_all(values: &mut [blst_fp], scratch: &mut [blst_fp]) {
    let Some(last) = values.len().checked_sub(1) else {
        return;
    };
    scratch[0] = values[0];
    for j in 1..values.len() {
        let prev = scratch[j - 1];
        fp_mul(&mut scratch[j], &prev, &values[j]);
    }
    let mut inv = fp_zero();
    fp_inv(&mut inv, &scratch[last]);
    let mut tmp = fp_zero();
    for j in (1..values.len()).rev() {
        fp_mul(&mut tmp, &inv, &scratch[j - 1]);
        let vj = values[j];
        fp_mul_assign(&mut inv, &vj);
        values[j] = tmp;
    }
    values[0] = inv;
}

#[inline(always)]
fn add_with_inverse(
    x1: &mut blst_fp,
    y1: &mut blst_fp,
    x2: &blst_fp,
    y2: &blst_fp,
    negate: bool,
    inv: &blst_fp,
) {
    let (mut lambda, mut t, mut d) = (fp_zero(), fp_zero(), fp_zero());
    if negate {
        fp_add(&mut t, y2, y1);
    } else {
        fp_sub(&mut t, y2, y1);
    }
    fp_mul(&mut lambda, &t, inv);
    fp_sqr(&mut t, &lambda);
    fp_sub_assign(&mut t, x1);
    fp_sub_assign(&mut t, x2);
    if negate {
        fp_sub(&mut d, &t, x1);
    } else {
        fp_sub(&mut d, x1, &t);
    }
    *x1 = t;
    fp_mul(&mut t, &lambda, &d);
    fp_rsub_assign(y1, &t);
}

#[inline(always)]
fn xy_of<'a>(
    bases: &'a [G1AffineLimbs],
    phi_x: &'a [blst_fp],
    pt: u32,
) -> (&'a blst_fp, &'a blst_fp) {
    let idx = (pt >> 1) as usize;
    // SAFETY: `G1AffineLimbs` is `[u64; 12]`, identical layout to `[blst_fp; 2]`.
    let pair = unsafe { &*(bases.as_ptr().add(idx) as *const [blst_fp; 2]) };
    if (pt & 1) == 0 {
        (&pair[0], &pair[1])
    } else {
        (&phi_x[idx], &pair[1])
    }
}

struct AffineBuckets {
    buckets: Vec<AffineBucket>,
    /// Allocated lazily only if a collision/overflow sends a point to XYZZ.
    side: Vec<Xyzz>,
    sides: bool,
    batch: Vec<Pending>,
    dx: Vec<blst_fp>,
    prefix: Vec<blst_fp>,
    waiting: Vec<Pending>,
    spare: Vec<Pending>,
    acc: Vec<AffineBucket>,
    sum: Vec<AffineBucket>,
}

impl AffineBuckets {
    fn new(buckets: usize, batch: usize) -> Self {
        let lanes = LANES.min(batch);
        Self {
            buckets: vec![AffineBucket::empty(); buckets],
            side: Vec::new(),
            sides: false,
            batch: Vec::with_capacity(batch),
            dx: vec![fp_zero(); batch],
            prefix: vec![fp_zero(); batch],
            waiting: Vec::with_capacity(batch),
            spare: Vec::with_capacity(batch),
            acc: vec![AffineBucket::empty(); lanes],
            sum: vec![AffineBucket::empty(); lanes],
        }
    }

    fn ensure(&mut self, buckets: usize, batch: usize) {
        let lanes = LANES.min(batch);
        if self.buckets.len() < buckets {
            self.buckets.resize(buckets, AffineBucket::empty());
        }
        if self.dx.len() < batch {
            self.dx.resize(batch, fp_zero());
            self.prefix.resize(batch, fp_zero());
        }
        if self.acc.len() < lanes {
            self.acc.resize(lanes, AffineBucket::empty());
            self.sum.resize(lanes, AffineBucket::empty());
        }
        self.batch.clear();
        self.waiting.clear();
        self.spare.clear();
        self.sides = false;
    }

    #[inline(never)]
    fn add_side(&mut self, b: usize, x: &blst_fp, y: &blst_fp, negate: bool) {
        if self.side.len() < self.buckets.len() {
            self.side.resize(self.buckets.len(), Xyzz::identity());
        }
        self.side[b].add_affine(x, y, negate);
        self.sides = true;
    }

    fn wipe_all(&mut self, used_buckets: usize) {
        let n = used_buckets.min(self.buckets.len());
        wipe_slice(&mut self.buckets[..n]);
        if !self.side.is_empty() {
            wipe_slice(&mut self.side);
        }
        wipe_slice(&mut self.dx);
        wipe_slice(&mut self.prefix);
        wipe_slice(&mut self.acc);
        wipe_slice(&mut self.sum);
        self.batch.clear();
        self.waiting.clear();
        self.spare.clear();
        self.sides = false;
    }

    #[inline(always)]
    fn add(
        &mut self,
        b: usize,
        pt: u32,
        negate: bool,
        batch_cap: usize,
        bases: &[G1AffineLimbs],
        phi_x: &[blst_fp],
    ) {
        let p = Pending {
            bucket: b as u32,
            point: pt,
            negate,
            same_x: false,
        };
        self.enqueue(p, batch_cap, bases, phi_x);
        if self.batch.len() == batch_cap {
            self.flush(batch_cap, bases, phi_x);
        }
    }

    #[inline(always)]
    fn enqueue(&mut self, p: Pending, batch_cap: usize, bases: &[G1AffineLimbs], phi_x: &[blst_fp]) {
        let bucket = &mut self.buckets[p.bucket as usize];
        if !bucket.full {
            let (x, y) = xy_of(bases, phi_x, p.point);
            bucket.x = *x;
            if p.negate {
                fp_cneg(&mut bucket.y, y, true);
            } else {
                bucket.y = *y;
            }
            bucket.full = true;
        } else if !bucket.queued {
            bucket.queued = true;
            self.batch.push(p);
        } else if self.waiting.len() < batch_cap {
            self.waiting.push(p);
        } else {
            let (x, y) = xy_of(bases, phi_x, p.point);
            self.add_side(p.bucket as usize, x, y, p.negate);
        }
    }

    fn flush(&mut self, batch_cap: usize, bases: &[G1AffineLimbs], phi_x: &[blst_fp]) {
        loop {
            self.add_batch(bases, phi_x);
            std::mem::swap(&mut self.waiting, &mut self.spare);
            for k in 0..self.spare.len() {
                let p = self.spare[k];
                self.enqueue(p, batch_cap, bases, phi_x);
            }
            self.spare.clear();
            if self.batch.len() < batch_cap {
                return;
            }
        }
    }

    fn add_batch(&mut self, bases: &[G1AffineLimbs], phi_x: &[blst_fp]) {
        let len = self.batch.len();
        for (p, dx) in self.batch.iter_mut().zip(&mut self.dx) {
            let (x, _) = xy_of(bases, phi_x, p.point);
            fp_sub(dx, x, &self.buckets[p.bucket as usize].x);
            p.same_x = fp_is_zero(dx);
            if p.same_x {
                *dx = FP_ONE;
            }
        }
        invert_all(&mut self.dx[..len], &mut self.prefix);
        for k in 0..len {
            let p = self.batch[k];
            let inv = self.dx[k];
            let bi = p.bucket as usize;
            let bucket = &mut self.buckets[bi];
            bucket.queued = false;
            let (x, y) = xy_of(bases, phi_x, p.point);
            if p.same_x {
                self.add_side(bi, x, y, p.negate);
            } else {
                add_with_inverse(&mut bucket.x, &mut bucket.y, x, y, p.negate, &inv);
            }
        }
        self.batch.clear();
    }

    /// Batched-affine accumulator for tiny top windows (`used <= 2`, e.g. the
    /// 1-bit top window of `c = 14`), avoiding single-bucket `waiting`/`side`
    /// overflow and `side` buffer allocation. Uses 64 L1-resident lanes per
    /// bucket with Montgomery batch inversion (`5M + 1S` per point).
    #[allow(clippy::too_many_arguments)]
    fn accumulate_tiny(
        &mut self,
        w: usize,
        c: usize,
        used: usize,
        start: usize,
        end: usize,
        blocks: &SecretBlocks,
        scalars: &[Scalar],
        bases: &[G1AffineLimbs],
        phi_slice: &[blst_fp],
        tile_id: usize,
    ) -> Xyzz {
        const TINY_LANES: usize = 64;
        debug_assert!(used <= 2);
        let mut lanes = [AffineBucket::empty(); 2 * TINY_LANES];
        let mut next_lane = [0usize; 2];
        let mut side = [Xyzz::identity(); 2];
        let mut pend = [Pending {
            bucket: 0,
            point: 0,
            negate: false,
            same_x: false,
        }; 2 * TINY_LANES];
        let mut plen = 0usize;

        let flush_tiny = |lanes: &mut [AffineBucket],
                          pend: &mut [Pending],
                          plen: &mut usize,
                          side: &mut [Xyzz; 2],
                          dx: &mut [blst_fp],
                          prefix: &mut [blst_fp]| {
            let len = *plen;
            if len == 0 {
                return;
            }
            for k in 0..len {
                let p = &mut pend[k];
                let (x, _) = xy_of(bases, phi_slice, p.point);
                fp_sub(&mut dx[k], x, &lanes[p.bucket as usize].x);
                p.same_x = fp_is_zero(&dx[k]);
                if p.same_x {
                    dx[k] = FP_ONE;
                }
            }
            invert_all(&mut dx[..len], prefix);
            for k in 0..len {
                let p = pend[k];
                let slot = p.bucket as usize;
                let (x, y) = xy_of(bases, phi_slice, p.point);
                if p.same_x {
                    side[slot / TINY_LANES].add_affine(x, y, p.negate);
                } else {
                    add_with_inverse(&mut lanes[slot].x, &mut lanes[slot].y, x, y, p.negate, &dx[k]);
                }
            }
            *plen = 0;
        };

        let push_pt = |b: usize,
                           pt: u32,
                           negate: bool,
                           lanes: &mut [AffineBucket],
                           next_lane: &mut [usize; 2],
                           pend: &mut [Pending],
                           plen: &mut usize,
                           side: &mut [Xyzz; 2],
                           dx: &mut [blst_fp],
                           prefix: &mut [blst_fp]| {
            let l = next_lane[b];
            next_lane[b] = (l + 1) & (TINY_LANES - 1);
            let slot = b * TINY_LANES + l;
            if !lanes[slot].full {
                let (x, y) = xy_of(bases, phi_slice, pt);
                lanes[slot].x = *x;
                if negate {
                    fp_cneg(&mut lanes[slot].y, y, true);
                } else {
                    lanes[slot].y = *y;
                }
                lanes[slot].full = true;
            } else {
                pend[*plen] = Pending {
                    bucket: slot as u32,
                    point: pt,
                    negate,
                    same_x: false,
                };
                *plen += 1;
                if *plen == TINY_LANES {
                    flush_tiny(lanes, pend, plen, side, dx, prefix);
                }
            }
        };

        let n = scalars.len();
        let b0 = start / SCALAR_BLOCK;
        let b1 = end.div_ceil(SCALAR_BLOCK);
        let nblocks = b1.saturating_sub(b0);
        for j in 0..nblocks {
            let bi = b0 + (j + tile_id) % nblocks;
            let first = bi * SCALAR_BLOCK;
            let blk = get_half_block(blocks, bi, scalars, bases, n);
            let lo = start.saturating_sub(first);
            let hi = end.min(first + blk.len()) - first;
            for (off, &(e1, e2)) in blk[lo..hi].iter().enumerate() {
                let i = first + lo + off;
                let (k1, neg1) = (e1 & !SIGN_BIT, (e1 & SIGN_BIT) != 0);
                let d1 = booth_u128(k1, w, c);
                if d1 != 0 {
                    let b = d1.unsigned_abs() as usize - 1;
                    push_pt(
                        b,
                        (i as u32) << 1,
                        (d1 < 0) ^ neg1,
                        &mut lanes,
                        &mut next_lane,
                        &mut pend,
                        &mut plen,
                        &mut side,
                        &mut self.dx,
                        &mut self.prefix,
                    );
                }
                let (k2, neg2) = (e2 & !SIGN_BIT, (e2 & SIGN_BIT) != 0);
                let d2 = booth_u128(k2, w, c);
                if d2 != 0 {
                    let b = d2.unsigned_abs() as usize - 1;
                    push_pt(
                        b,
                        ((i as u32) << 1) | 1,
                        (d2 < 0) ^ neg2,
                        &mut lanes,
                        &mut next_lane,
                        &mut pend,
                        &mut plen,
                        &mut side,
                        &mut self.dx,
                        &mut self.prefix,
                    );
                }
            }
        }
        flush_tiny(
            &mut lanes,
            &mut pend,
            &mut plen,
            &mut side,
            &mut self.dx,
            &mut self.prefix,
        );

        let mut b_sums = [Xyzz::identity(); 2];
        for b in 0..used {
            let mut s = side[b];
            for l in 0..TINY_LANES {
                let slot = &lanes[b * TINY_LANES + l];
                if slot.full {
                    s.add_affine(&slot.x, &slot.y, false);
                }
            }
            b_sums[b] = s;
        }
        let mut total = b_sums[0];
        if used > 1 && !b_sums[1].is_identity() {
            b_sums[1].double();
            total.add(&b_sums[1]);
        }
        wipe_slice(&mut lanes);
        wipe_slice(&mut side);
        wipe_slice(&mut b_sums);
        total
    }

    fn integrate_and_clear(
        &mut self,
        used: usize,
        batch_cap: usize,
        bases: &[G1AffineLimbs],
        phi_x: &[blst_fp],
    ) -> Xyzz {
        for _ in 0..4 {
            if self.batch.is_empty() && self.waiting.is_empty() {
                break;
            }
            self.flush(batch_cap, bases, phi_x);
        }
        for k in 0..self.batch.len() {
            let p = self.batch[k];
            self.buckets[p.bucket as usize].queued = false;
            self.waiting.push(p);
        }
        self.batch.clear();
        for k in 0..self.waiting.len() {
            let p = self.waiting[k];
            let (x, y) = xy_of(bases, phi_x, p.point);
            self.add_side(p.bucket as usize, x, y, p.negate);
        }
        self.waiting.clear();
        let lanes = (used / 8).clamp(32, LANES.min(batch_cap));
        if !self.sides
            && used >= 4 * lanes
            && let Some(sum) = self.integrate_lanes(used, lanes)
        {
            self.buckets[..used].iter_mut().for_each(|b| b.full = false);
            return sum;
        }
        let mut acc = Xyzz::identity();
        let mut sum = Xyzz::identity();
        let has_side = self.sides && self.side.len() >= used;
        for i in (0..used).rev() {
            let bucket = &mut self.buckets[i];
            if bucket.full {
                acc.add_affine(&bucket.x, &bucket.y, false);
                bucket.full = false;
            }
            if has_side && !self.side[i].is_identity() {
                acc.add(&self.side[i]);
                self.side[i] = Xyzz::identity();
            }
            sum.add(&acc);
        }
        self.sides = false;
        sum
    }

    fn integrate_lanes(&mut self, used: usize, lanes: usize) -> Option<Xyzz> {
        let seg = used / lanes;
        for s in self.acc[..lanes].iter_mut().chain(&mut self.sum[..lanes]) {
            (s.full, s.queued) = (false, false);
        }
        for j in (0..seg).rev() {
            let (acc, sum) = (&mut self.acc[..lanes], &mut self.sum[..lanes]);
            let buckets = &self.buckets;
            if !add_lanes(acc, |l| &buckets[l * seg + j], &mut self.dx, &mut self.prefix)
                || !add_lanes(sum, |l| &acc[l], &mut self.dx, &mut self.prefix)
            {
                return None;
            }
        }
        let (mut run, mut total, mut sums) = (Xyzz::identity(), Xyzz::identity(), Xyzz::identity());
        for l in (0..lanes).rev() {
            total.add(&run);
            if self.acc[l].full {
                run.add_affine(&self.acc[l].x, &self.acc[l].y, false);
            }
            if self.sum[l].full {
                sums.add_affine(&self.sum[l].x, &self.sum[l].y, false);
            }
        }
        for _ in 0..seg.ilog2() {
            total.double();
        }
        total.add(&sums);
        Some(total)
    }
}

fn add_lanes<'a>(
    dst: &mut [AffineBucket],
    src: impl Fn(usize) -> &'a AffineBucket,
    dx: &mut [blst_fp],
    prefix: &mut [blst_fp],
) -> bool {
    let mut m = 0;
    for (l, d) in dst.iter_mut().enumerate() {
        let s = src(l);
        if !s.full {
            continue;
        }
        if !d.full {
            *d = *s;
            continue;
        }
        fp_sub(&mut dx[m], &s.x, &d.x);
        if fp_is_zero(&dx[m]) {
            return false;
        }
        d.queued = true;
        m += 1;
    }
    invert_all(&mut dx[..m], prefix);
    let queued = dst.iter_mut().enumerate().filter(|(_, d)| d.queued);
    for ((l, d), inv) in queued.zip(dx.iter()) {
        let s = src(l);
        d.queued = false;
        add_with_inverse(&mut d.x, &mut d.y, &s.x, &s.y, false, inv);
    }
    true
}

fn window_size(n: usize) -> usize {
    // 2*n half-points per 128-bit window:
    match n {
        0..=4_095 => 10,
        4_096..=12_287 => 11,
        12_288..=32_767 => 12,
        32_768..=65_535 => 13,
        _ => 14,
    }
}

fn chunk_count(n: usize, c: usize, threads: usize) -> usize {
    let windows = NBITS / c + 1;
    // With c = 14, window 9 is only 2 bits (tiny), so count full windows for load balance.
    let full_wins = if NBITS - (windows - 1) * c <= 3 {
        windows - 1
    } else {
        windows
    };
    let mut best = (usize::MAX, 1);
    for chunks in 1..=threads.clamp(1, 3) {
        let per_chunk = (2 * n).div_ceil(chunks);
        if chunks > 1 && per_chunk < 256 {
            break;
        }
        let rounds = (full_wins * chunks).div_ceil(threads.max(1));
        // Weight bucket integration (2^(c-1) buckets per tile) accurately so we
        // don't over-split into extra integration passes.
        let cost = rounds * (per_chunk + (2 << c)) + full_wins * chunks;
        if cost < best.0 {
            best = (cost, chunks);
        }
    }
    best.1
}

/// Precomputes `phi_x[i] = beta * bases[i].x` in parallel for a registered SRS table.
pub(crate) fn precompute_phi_x(bases: &[G1AffineLimbs]) -> Vec<blst_fp> {
    let mut phi_x = vec![fp_zero(); bases.len()];
    phi_x.par_iter_mut().enumerate().for_each(|(i, px)| {
        // SAFETY: `G1AffineLimbs` is `[u64; 12]`, identical to `[blst_fp; 2]`.
        let pair = unsafe { &*(bases.as_ptr().add(i) as *const [blst_fp; 2]) };
        if !(fp_is_zero(&pair[0]) && fp_is_zero(&pair[1])) {
            fp_mul(px, &pair[0], &BETA_MONT);
        }
    });
    phi_x
}

#[inline(always)]
fn get_half_block<'a>(
    blocks: &'a SecretBlocks,
    bi: usize,
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    n: usize,
) -> &'a [(u128, u128)] {
    blocks.0[bi].get_or_init(|| {
        let first = bi * SCALAR_BLOCK;
        let end = (first + SCALAR_BLOCK).min(n);
        let mut out = Vec::with_capacity(end - first);
        for (off, sc) in scalars[first..end].iter().enumerate() {
            // SAFETY: `G1AffineLimbs` is `[u64; 12]`, identical to `[blst_fp; 2]`.
            let pair = unsafe { &*(bases.as_ptr().add(first + off) as *const [blst_fp; 2]) };
            if fp_is_zero(&pair[0]) && fp_is_zero(&pair[1]) {
                out.push((0, 0));
            } else {
                out.push(decompose(sc));
            }
        }
        out
    })
}

/// Computes `sum scalars[i] * bases[i]` on the current rayon thread pool using
/// the BLS12-381 G1 GLV endomorphism and batched-affine Pippenger.
///
/// Caller guarantees every base in `bases` is in G1 and every scalar is
/// reduced (`< r`). When `cached_phi_x` is provided, it must hold
/// `beta * bases[i].x` for every `i < bases.len()`.
pub(super) fn msm_glv(
    scalars: &[Scalar],
    bases: &[G1AffineLimbs],
    cached_phi_x: Option<&[blst_fp]>,
) -> Secret<blst_p1> {
    let n = scalars.len();
    debug_assert_eq!(n, bases.len());
    if n == 0 {
        return Secret(blst_p1::default());
    }

    // Lazily decompose scalars per block (no serial zeroing or pre-tile
    // barrier); compute `phi_x` in parallel only if not cached on the SRS table.
    let owned_phi: Vec<blst_fp>;
    let phi_slice: &[blst_fp] = if let Some(cached) = cached_phi_x.filter(|c| c.len() >= n) {
        &cached[..n]
    } else {
        owned_phi = precompute_phi_x(bases);
        &owned_phi
    };
    let blocks = SecretBlocks((0..n.div_ceil(SCALAR_BLOCK)).map(|_| OnceLock::new()).collect());

    let c = window_size(n);
    let windows = NBITS / c + 1;
    let threads = rayon::current_num_threads();
    let full_wins = if NBITS - (windows - 1) * c <= 3 {
        windows - 1
    } else {
        windows
    };

    // Build tile list `(w, start, end)`. When `full_wins == threads + 1` (e.g.
    // 9 full 14-bit windows on 8 cores), keep `chunks = 1` on the first
    // `threads` windows and split only the extra window into 2 halves: cuts
    // bucket integrations by 44% while keeping all cores (including faster
    // Prime cores on big.LITTLE SoCs) evenly loaded.
    let mut tile_specs: Vec<(usize, usize, usize)> = Vec::with_capacity(windows * 2);
    if full_wins == threads + 1 && n >= 16_384 {
        for w in 0..threads {
            tile_specs.push((w, 0, n));
        }
        let mid = (n / 2).next_multiple_of(SCALAR_BLOCK).min(n);
        for w in threads..windows {
            if mid > 0 {
                tile_specs.push((w, 0, mid));
            }
            if mid < n {
                tile_specs.push((w, mid, n));
            }
        }
    } else {
        let chunks = chunk_count(n, c, threads).min(2);
        let chunk = n.div_ceil(chunks).next_multiple_of(SCALAR_BLOCK);
        for w in 0..windows {
            for k in 0..chunks {
                let start = (k * chunk).min(n);
                let end = ((k + 1) * chunk).min(n);
                if start < end {
                    tile_specs.push((w, start, end));
                }
            }
        }
    }
    let tiles = tile_specs.len();
    let buckets_len = 1usize << (c - 1);
    let batch = (buckets_len / 4).clamp(64, 512).min(buckets_len);

    let results: Vec<OnceLock<Xyzz>> = (0..tiles).map(|_| OnceLock::new()).collect();
    let next = AtomicUsize::new(0);

    thread_local! {
        static TL_BUCKETS: std::cell::RefCell<Option<AffineBuckets>> = const { std::cell::RefCell::new(None) };
    }

    rayon::in_place_scope(|scope| {
        for _ in 0..threads.min(tiles) {
            let (results, next, blocks, tile_specs) = (&results, &next, &blocks, &tile_specs);
            scope.spawn(move |_| {
                let _scrub = ScrubOnExit;
                TL_BUCKETS.with(|cell| {
                    let mut guard = cell.borrow_mut();
                    let affine = match guard.as_mut() {
                        Some(a) => {
                            a.ensure(buckets_len, batch);
                            a
                        }
                        None => guard.insert(AffineBuckets::new(buckets_len, batch)),
                    };
                    loop {
                        let t = next.fetch_add(1, Ordering::Relaxed);
                        if t >= tiles {
                            break;
                        }
                        let (w, start, end) = tile_specs[t];
                        let used = 1usize << (NBITS - w * c).min(c - 1);
                        let sum = if used <= 2 {
                            affine.accumulate_tiny(
                                w, c, used, start, end, blocks, scalars, bases, phi_slice, t,
                            )
                        } else {
                            let b0 = start / SCALAR_BLOCK;
                            let b1 = end.div_ceil(SCALAR_BLOCK);
                            let nblocks = b1.saturating_sub(b0);
                            for j in 0..nblocks {
                                let bi = b0 + (j + t) % nblocks;
                                let first = bi * SCALAR_BLOCK;
                                let blk = get_half_block(blocks, bi, scalars, bases, n);
                                let lo = start.saturating_sub(first);
                                let hi = end.min(first + blk.len()) - first;
                                for (off, &(e1, e2)) in blk[lo..hi].iter().enumerate() {
                                    let i = first + lo + off;
                                    let (k1, neg1) = (e1 & !SIGN_BIT, (e1 & SIGN_BIT) != 0);
                                    let d1 = booth_u128(k1, w, c);
                                    if d1 != 0 {
                                        affine.add(
                                            d1.unsigned_abs() as usize - 1,
                                            (i as u32) << 1,
                                            (d1 < 0) ^ neg1,
                                            batch,
                                            bases,
                                            phi_slice,
                                        );
                                    }
                                    let (k2, neg2) = (e2 & !SIGN_BIT, (e2 & SIGN_BIT) != 0);
                                    let d2 = booth_u128(k2, w, c);
                                    if d2 != 0 {
                                        affine.add(
                                            d2.unsigned_abs() as usize - 1,
                                            ((i as u32) << 1) | 1,
                                            (d2 < 0) ^ neg2,
                                            batch,
                                            bases,
                                            phi_slice,
                                        );
                                    }
                                }
                            }
                            affine.integrate_and_clear(used, batch, bases, phi_slice)
                        };
                        let _ = results[t].set(sum);
                    }
                    affine.wipe_all(buckets_len);
                });
            });
        }
    });

    // Combine windows via Horner's rule, most significant first.
    let mut win_sums = vec![Xyzz::identity(); windows];
    for (t, &(w, _, _)) in tile_specs.iter().enumerate() {
        if let Some(part) = results[t].get() {
            win_sums[w].add(part);
        }
    }
    let mut acc = Secret(blst_p1::default());
    for w in (0..windows).rev() {
        if w != windows - 1 {
            for _ in 0..c {
                let tmp = *acc;
                // SAFETY: valid blst_p1 point.
                unsafe { blst_p1_double(&mut *acc, &tmp) };
            }
        }
        if !win_sums[w].is_identity() {
            let p = win_sums[w].to_p1();
            let tmp = *acc;
            // SAFETY: valid blst_p1 points.
            unsafe { blst_p1_add_or_double(&mut *acc, &tmp, &p) };
        }
        wipe(&mut win_sums[w]);
    }
    acc
}

#[cfg(test)]
mod tests {
    use blst::blst_p1_is_equal;
    use ff::Field;
    use group::{Curve, Group, prime::PrimeCurveAffine};
    use midnight_curves::{Fq, G1Affine, G1Projective, msm::msm_pippenger};

    use super::*;

    fn next_u64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn fq_from_u128(v: u128) -> Fq {
        let hi = Fq::from((v >> 64) as u64);
        let lo = Fq::from(v as u64);
        let shift = Fq::from(1u64 << 32).square();
        hi * shift + lo
    }

    fn to_limbs<'a>(s: &'a [Fq], b: &'a [G1Affine]) -> (&'a [Scalar], &'a [G1AffineLimbs]) {
        // SAFETY: layout asserted in parent module.
        unsafe {
            (
                &*(s as *const [Fq] as *const [Scalar]),
                &*(b as *const [G1Affine] as *const [G1AffineLimbs]),
            )
        }
    }

    fn run_compare(scalars: &[Fq], bases: &[G1Affine]) {
        let expected: G1Projective = msm_pippenger(scalars, bases);
        let expected_p1: blst_p1 = unsafe { std::mem::transmute(expected) };
        let (sl, bl) = to_limbs(scalars, bases);
        let got = msm_glv(sl, bl, None);
        assert!(
            unsafe { blst_p1_is_equal(&*got, &expected_p1) },
            "GLV CPU MSM mismatch for n = {}",
            scalars.len()
        );
        let cached = precompute_phi_x(bl);
        let got_cached = msm_glv(sl, bl, Some(&cached));
        assert!(
            unsafe { blst_p1_is_equal(&*got_cached, &expected_p1) },
            "GLV CPU MSM (cached phi_x) mismatch for n = {}",
            scalars.len()
        );
    }

    #[test]
    fn glv_matches_stock_pippenger() {
        let mut st = 0x1234_5678_9abc_def0u64;
        let n = 5000usize;
        let g = G1Projective::generator();
        let pts: Vec<G1Affine> = (0..n)
            .map(|i| {
                if i == 17 || i == 2048 {
                    G1Affine::identity()
                } else if i == 33 {
                    g.to_affine()
                } else {
                    (g * Fq::from(next_u64(&mut st) | 1)).to_affine()
                }
            })
            .collect();
        let mut scs: Vec<Fq> = (0..n)
            .map(|_| {
                let a = fq_from_u128((next_u64(&mut st) as u128) | ((next_u64(&mut st) as u128) << 64));
                let b = fq_from_u128((next_u64(&mut st) as u128) | ((next_u64(&mut st) as u128) << 64));
                a * b + Fq::from(next_u64(&mut st))
            })
            .collect();
        // Inject boundary scalars: 0, 1, -1, U2, U2-1, U2+1
        scs[0] = Fq::ZERO;
        scs[1] = Fq::ONE;
        scs[2] = -Fq::ONE;
        scs[3] = fq_from_u128(U2);
        scs[4] = fq_from_u128(U2 - 1);
        scs[5] = fq_from_u128(U2 + 1);
        let mut dup_pts = pts[..256].to_vec();
        dup_pts[10] = dup_pts[5];
        dup_pts[11] = dup_pts[5];
        run_compare(&scs[..256], &dup_pts);
        run_compare(&scs[..1], &pts[..1]);
        run_compare(&scs[..100], &pts[..100]);
        run_compare(&scs, &pts);
    }
}
