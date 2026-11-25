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

// BLS12-381 G1 points for the MSM: affine and XYZZ coordinates.
//
// The formulas are ported from sppark `ec/xyzz_t.hpp` (Copyright Supranational
// LLC, Apache-2.0, https://github.com/supranational/sppark, commit
// 08033103c14ff3c742e710b9463dde42acc0f058): `add-2008-s` and `madd-2008-s`,
// with `dbl-2008-s-1` and `mdbl-2008-s-1` for equal inputs (EFD, short
// Weierstrass, a = 0), as in the Metal crate's `g1.h`.
//
// - `G1Affine`: `x`, `y` in `Fp`, the bytes of a host `blst_p1_affine`;
//   infinity is all zero.
// - `G1Xyzz`: `X`, `Y`, `ZZZ`, `ZZ` (sppark's order), the point
//   `(X / ZZ, Y / ZZZ)`; infinity is stored all zero. As Jacobian coordinates
//   the point is `(X * ZZ, Y * ZZZ, ZZ)`. A negation negates `ZZZ`.

#ifndef MIDNIGHT_VULKAN_G1_GLSL
#define MIDNIGHT_VULKAN_G1_GLSL

#include "fp.glsl"

struct G1Affine {
    Fp x;
    Fp y;
};

struct G1Xyzz {
    Fp X;
    Fp Y;
    Fp ZZZ;
    Fp ZZ;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer AffineRef {
    G1Affine v;
};

layout(buffer_reference, std430, buffer_reference_align = 16) buffer XyzzRef {
    G1Xyzz v;
};

// Element i of the `G1Affine` / `G1Xyzz` array at `base`.
#define AFFINE_AT(base, i) AffineRef(addr_at(base, i, 96u)).v
#define XYZZ_AT(base, i) XyzzRef(addr_at(base, i, 192u)).v

// The field multiplication of the curve formulas, kept out of line by build.rs
// in the kernels that list it. Returns a lazy-reduced result in [0, 2p) for
// inputs a < 4p, b < 2p (or vice versa), skipping 12-limb reduction on every
// multiplication inside `g1_madd` and `g1_add`.
Fp g1_mul(Fp a, Fp b) {
    return fp_mul_lazy(a, b);
}

Fp fp_cneg(Fp a, bool flag) {
    return flag ? fp_neg(a) : a;
}

Fp fp_cneg_2p(Fp a, bool flag) {
    return flag ? fp_neg_2p(a) : a;
}

// Canonicalize a lazy-reduced XYZZ point ([0, 2p) coordinates) into [0, p)
// before writing it to device memory.
void g1_reduce(inout G1Xyzz p) {
    p.X = fp_reduce_once(p.X);
    p.Y = fp_reduce_once(p.Y);
    p.ZZZ = fp_reduce_once(p.ZZZ);
    p.ZZ = fp_reduce_once(p.ZZ);
}

bool g1_is_inf(G1Affine a) {
    return fp_is_zero(a.x) && fp_is_zero(a.y);
}

bool g1_is_inf(G1Xyzz p) {
    return fp_is_zero(p.ZZZ) && fp_is_zero(p.ZZ);
}

G1Xyzz g1_inf() {
    G1Xyzz p;
    p.X = fp_zero();
    p.Y = fp_zero();
    p.ZZZ = fp_zero();
    p.ZZ = fp_zero();
    return p;
}

// sppark `xyzz_t::operator=(const affine_t&)`: ZZZ = ZZ = one, or zero for
// the point at infinity.
G1Xyzz g1_from_affine(G1Affine a) {
    G1Xyzz p;
    p.X = a.x;
    p.Y = a.y;
    p.ZZZ = g1_is_inf(a) ? fp_zero() : fp_one();
    p.ZZ = p.ZZZ;
    return p;
}

void g1_cneg(inout G1Xyzz p, bool neg) {
    p.ZZZ = fp_cneg(p.ZZZ, neg);
}

// p31 += p2 (sppark `xyzz_t::add(const xyzz_t&)`): add-2008-s, doubling
// dbl-2008-s-1 when the points are equal, infinity when they are opposite.
// 12M + 2S for an addition. Maintains coordinates in [0, 2p).
void g1_add(inout G1Xyzz p31, G1Xyzz p2) {
    if (g1_is_inf(p2)) {
        return;
    } else if (g1_is_inf(p31)) {
        p31 = p2;
        return;
    }

    Fp U = g1_mul(p31.X, p2.ZZ);      // U1 = X1*ZZ2  (< 2p)
    Fp S = g1_mul(p31.Y, p2.ZZZ);     // S1 = Y1*ZZZ2 (< 2p)
    Fp P = g1_mul(p2.X, p31.ZZ);      // U2 = X2*ZZ1  (< 2p)
    Fp R = g1_mul(p2.Y, p31.ZZZ);     // S2 = Y2*ZZZ1 (< 2p)
    P = fp_reduce_once(fp_sub_2p(P, U)); // P = U2-U1 (< p)
    R = fp_sub_2p(R, S);              // R = S2-S1 (< 2p; 4p * 2p < p * 2^384)

    if (!fp_is_zero(P)) {             // X1 != X2: add
        Fp PP = g1_mul(P, P);         // PP = P^2 (< 2p)
        Fp PPP = g1_mul(P, PP);       // PPP = P*PP (< 2p)
        p31.ZZ = g1_mul(p31.ZZ, PP);
        p31.ZZZ = g1_mul(p31.ZZZ, PPP);
        Fp Q = g1_mul(U, PP);         // Q = U1*PP (< 2p)
        p31.X = g1_mul(R, R);         // R^2 (< 2p)
        p31.X = fp_sub_2p(p31.X, PPP);
        p31.X = fp_sub_2p(p31.X, Q);
        p31.X = fp_sub_2p(p31.X, Q);  // X3 = R^2-PPP-2*Q (< 2p)
        Q = fp_sub_unred(Q, p31.X);   // Q-X3 (< 4p, branchless)
        Q = g1_mul(Q, R);             // R*(Q-X3) (< 2p since 4p*2p < 9.84p^2)
        p31.Y = g1_mul(S, PPP);       // S1*PPP (< 2p)
        p31.Y = fp_sub_2p(Q, p31.Y);  // Y3 = R*(Q-X3)-S1*PPP (< 2p)
        p31.ZZ = g1_mul(p31.ZZ, p2.ZZ);       // ZZ3 = ZZ1*ZZ2*PP (< 2p)
        p31.ZZZ = g1_mul(p31.ZZZ, p2.ZZZ);    // ZZZ3 = ZZZ1*ZZZ2*PPP (< 2p)
    } else if (fp_is_zero(fp_reduce_once(R))) { // X1 == X2, Y1 == Y2: double (cold)
        Fp X1 = fp_reduce_once(p31.X);
        Fp Y1 = fp_reduce_once(p31.Y);
        U = fp_dbl(Y1);               // U = 2*Y1 (< p)
        Fp V = fp_mul_compact(U, U);          // V = U^2 (< 2p)
        Fp W = fp_mul_compact(U, V);          // W = U*V (< 2p)
        S = fp_mul_compact(X1, V);            // S = X1*V (< 2p)
        Fp M = fp_reduce_once(fp_mul_compact(X1, X1));
        M = fp_add(fp_dbl(M), M);     // M = 3*X1^2 (< p)
        p31.X = fp_mul_compact(M, M);
        p31.X = fp_sub_2p(p31.X, S);
        p31.X = fp_sub_2p(p31.X, S);  // X3 = M^2-2*S (< 2p)
        p31.Y = fp_mul_compact(Y1, W);        // W*Y1 (< 2p)
        S = fp_sub_unred(S, p31.X);   // S-X3 (< 4p)
        S = fp_mul_compact(S, M);             // M*(S-X3) (< 2p)
        p31.Y = fp_sub_2p(S, p31.Y);  // Y3 = M*(S-X3)-W*Y1 (< 2p)
        p31.ZZ = fp_mul_compact(p31.ZZ, V);   // ZZ3 = V*ZZ1 (< 2p)
        p31.ZZZ = fp_mul_compact(p31.ZZZ, W); // ZZZ3 = W*ZZZ1 (< 2p)
    } else {                          // X1 == X2, Y1 == -Y2
        p31 = g1_inf();
    }
}

// p31 += p2, or p31 -= p2 when `subtract` (sppark `xyzz_t::add(const
// affine_t&, bool)`): madd-2008-s, doubling mdbl-2008-s-1 when the points
// are equal, infinity when they are opposite. 8M + 2S for an addition.
// Maintains `p31` coordinates in [0, 2p); `p2` is canonical in [0, p).
void g1_madd(inout G1Xyzz p31, G1Affine p2, bool subtract) {
    if (g1_is_inf(p2)) {
        return;
    } else if (g1_is_inf(p31)) {
        p31.X = p2.x;
        p31.Y = p2.y;
        p31.ZZZ = fp_cneg(fp_one(), subtract);
        p31.ZZ = fp_one();
        return;
    }

    Fp R = g1_mul(p2.y, p31.ZZZ);     // S2 = Y2*ZZZ1 (< 2p)
    R = fp_cneg_2p(R, subtract);      // (< 2p)
    R = fp_sub_2p(R, p31.Y);          // R = S2-Y1 (< 2p; 4p * 2p < p * 2^384)
    Fp P = g1_mul(p2.x, p31.ZZ);      // U2 = X2*ZZ1 (< 2p)
    P = fp_reduce_once(fp_sub_2p(P, p31.X)); // P = U2-X1 (< p)

    if (!fp_is_zero(P)) {             // X1 != X2: add
        Fp PP = g1_mul(P, P);         // PP = P^2 (< 2p)
        Fp PPP = g1_mul(P, PP);       // PPP = P*PP (< 2p)
        p31.ZZ = g1_mul(p31.ZZ, PP);          // ZZ3 = ZZ1*PP (< 2p)
        p31.ZZZ = g1_mul(p31.ZZZ, PPP);       // ZZZ3 = ZZZ1*PPP (< 2p)
        Fp Q = g1_mul(PP, p31.X);     // Q = X1*PP (< 2p)
        p31.X = g1_mul(R, R);         // R^2 (< 2p)
        p31.X = fp_sub_2p(p31.X, PPP);
        p31.X = fp_sub_2p(p31.X, Q);
        p31.X = fp_sub_2p(p31.X, Q);  // X3 = R^2-PPP-2*Q (< 2p)
        Q = fp_sub_unred(Q, p31.X);   // Q-X3 (< 4p, branchless)
        Q = g1_mul(Q, R);             // R*(Q-X3) (< 2p since 4p*2p < 9.84p^2)
        p31.Y = g1_mul(p31.Y, PPP);           // Y1*PPP (< 2p)
        p31.Y = fp_sub_2p(Q, p31.Y);  // Y3 = R*(Q-X3)-Y1*PPP (< 2p)
    } else if (fp_is_zero(fp_reduce_once(R))) { // X1 == X2, Y1 == +-Y2: double p2
        Fp U = fp_dbl(p2.y);          // U = 2*Y1 (< p)
        p31.ZZ = g1_mul(U, U);        // ZZ3 = V = U^2 (< 2p)
        p31.ZZZ = g1_mul(p31.ZZ, U);          // ZZZ3 = W = U*V (< 2p)
        Fp S = g1_mul(p2.x, p31.ZZ);          // S = X1*V (< 2p)
        Fp M = fp_reduce_once(g1_mul(p2.x, p2.x));
        M = fp_add(fp_dbl(M), M);     // M = 3*X1^2 (< p)
        p31.X = g1_mul(M, M);
        p31.X = fp_sub_2p(p31.X, S);
        p31.X = fp_sub_2p(p31.X, S);  // X3 = M^2-2*S (< 2p)
        p31.Y = g1_mul(p31.ZZZ, p2.y);        // W*Y1 (< 2p)
        S = fp_sub_unred(S, p31.X);   // S-X3 (< 4p)
        S = g1_mul(S, M);             // M*(S-X3) (< 2p)
        p31.Y = fp_sub_2p(S, p31.Y);  // Y3 = M*(S-X3)-W*Y1 (< 2p)
        p31.ZZZ = fp_cneg_2p(p31.ZZZ, subtract);
    } else {                          // X1 == X2, Y1 == -+Y2
        p31 = g1_inf();
    }
}

// Fast mixed addition (`madd-2008-s`) for `msm_accumulate`: handles infinity,
// distinct-x addition, and opposite points (`P + (-P) = O`), returning `true`.
// On equal points (doubling, probability < 2^-255 on masked honest MSMs),
// returns `false` so `msm_accumulate_fallback` completes the bucket. Cuts
// `g1_mul` call sites in `msm_accumulate` from 17 to 10 (-41% code size) and
// loads `p2.x` / `p2.y` sequentially so they are never live together (-12 regs).
bool g1_madd_fast(inout G1Xyzz p31, uvec2 points, uint pt_idx, bool subtract) {
    Fp p2x = FP_AT(points, 2u * pt_idx);
    bool p2x_zero = fp_is_zero(p2x);
    if (g1_is_inf(p31)) {
        Fp p2y = FP_AT(points, 2u * pt_idx + 1u);
        if (p2x_zero && fp_is_zero(p2y)) return true;
        p31.X = p2x;
        p31.Y = p2y;
        p31.ZZZ = fp_cneg(fp_one(), subtract);
        p31.ZZ = fp_one();
        return true;
    }

    Fp P = g1_mul(p2x, p31.ZZ);              // U2 = X2*ZZ1 (< 2p); p2x dead here
    P = fp_reduce_once(fp_sub_2p(P, p31.X)); // P = U2-X1 (< p)

    Fp p2y = FP_AT(points, 2u * pt_idx + 1u);
    if (p2x_zero && fp_is_zero(p2y)) return true;
    Fp R = g1_mul(p2y, p31.ZZZ);             // S2 = Y2*ZZZ1 (< 2p); p2y dead here
    R = fp_cneg_2p(R, subtract);             // (< 2p)
    R = fp_sub_2p(R, p31.Y);                 // R = S2-Y1 (< 2p)

    if (!fp_is_zero(P)) {                    // X1 != X2: add
        Fp PP = g1_mul(P, P);
        Fp PPP = g1_mul(P, PP);
        p31.ZZ = g1_mul(p31.ZZ, PP);
        p31.ZZZ = g1_mul(p31.ZZZ, PPP);
        Fp Q = g1_mul(PP, p31.X);
        p31.X = g1_mul(R, R);
        p31.X = fp_sub_2p(p31.X, PPP);
        p31.X = fp_sub_2p(p31.X, Q);
        p31.X = fp_sub_2p(p31.X, Q);
        Q = fp_sub_unred(Q, p31.X);
        Q = g1_mul(Q, R);
        p31.Y = g1_mul(p31.Y, PPP);
        p31.Y = fp_sub_2p(Q, p31.Y);
        return true;
    }
    if (!fp_is_zero(fp_reduce_once(R))) {    // X1 == X2, Y1 == -+Y2
        p31 = g1_inf();
        return true;
    }
    return false;                            // X1 == X2, Y1 == +-Y2: delegate doubling
}

#endif // MIDNIGHT_VULKAN_G1_GLSL
