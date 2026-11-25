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

// The field multiplication of the curve formulas. build.rs keeps it out of
// line (SPIR-V `DontInline`) in the kernels that list it, as it keeps
// `fp_mul` in every kernel. A driver may still inline it.
Fp g1_mul(Fp a, Fp b) {
    return fp_mul(a, b);
}

Fp fp_cneg(Fp a, bool flag) {
    return flag ? fp_neg(a) : a;
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
// 12M + 2S for an addition.
//
// All field multiplications go through a single `g1_mul` call site in a
// `[[dont_unroll]]` step loop so drivers that ignore SPIR-V `DontInline`
// (e.g. Qualcomm Adreno) compile one unrolled `fp_mul` per function instead
// of 23 copies.
void g1_add(inout G1Xyzz p31, G1Xyzz p2) {
    if (g1_is_inf(p2)) {
        return;
    } else if (g1_is_inf(p31)) {
        p31 = p2;
        return;
    }

    Fp U, S, P, R, PP, PPP, Q;
    uint step = 0u;
    [[dont_unroll]] while (step < 23u) {
        Fp ma, mb;
        switch (step) {
        case 0u:  ma = p31.X;   mb = p2.ZZ;   break;
        case 1u:  ma = p31.Y;   mb = p2.ZZZ;  break;
        case 2u:  ma = p2.X;    mb = p31.ZZ;  break;
        case 3u:  ma = p2.Y;    mb = p31.ZZZ; break;
        // add-2008-s
        case 4u:  ma = P;       mb = P;       break;
        case 5u:  ma = P;       mb = PP;      break;
        case 6u:  ma = p31.ZZ;  mb = PP;      break;
        case 7u:  ma = p31.ZZZ; mb = PPP;     break;
        case 8u:  ma = U;       mb = PP;      break;
        case 9u:  ma = R;       mb = R;       break;
        case 10u: ma = Q;       mb = R;       break;
        case 11u: ma = S;       mb = PPP;     break;
        case 12u: ma = p31.ZZ;  mb = p2.ZZ;   break;
        case 13u: ma = p31.ZZZ; mb = p2.ZZZ;  break;
        // dbl-2008-s-1
        case 14u: ma = U;       mb = U;       break;
        case 15u: ma = U;       mb = PP;      break;
        case 16u: ma = p31.X;   mb = PP;      break;
        case 17u: ma = p31.X;   mb = p31.X;   break;
        case 18u: ma = Q;       mb = Q;       break;
        case 19u: ma = p31.Y;   mb = PPP;     break;
        case 20u: ma = S;       mb = Q;       break;
        case 21u: ma = p31.ZZ;  mb = PP;      break;
        default:  ma = p31.ZZZ; mb = PPP;     break;
        }
        Fp m = g1_mul(ma, mb);
        switch (step) {
        case 0u:  U = m; step = 1u; break;
        case 1u:  S = m; step = 2u; break;
        case 2u:  P = fp_sub(m, U); step = 3u; break;
        case 3u:
            R = fp_sub(m, S);
            if (!fp_is_zero(P)) {
                step = 4u;
            } else if (fp_is_zero(R)) {
                U = fp_dbl(p31.Y);
                step = 14u;
            } else {
                p31 = g1_inf();
                step = 23u;
            }
            break;
        // add-2008-s
        case 4u:  PP = m;      step = 5u;  break;
        case 5u:  PPP = m;     step = 6u;  break;
        case 6u:  p31.ZZ = m;  step = 7u;  break;
        case 7u:  p31.ZZZ = m; step = 8u;  break;
        case 8u:  Q = m;       step = 9u;  break;
        case 9u:
            p31.X = fp_sub(fp_sub(fp_sub(m, PPP), Q), Q);
            Q = fp_sub(Q, p31.X);
            step = 10u;
            break;
        case 10u: Q = m; step = 11u; break;
        case 11u: p31.Y = fp_sub(Q, m); step = 12u; break;
        case 12u: p31.ZZ = m;  step = 13u; break;
        case 13u: p31.ZZZ = m; step = 23u; break;
        // dbl-2008-s-1
        case 14u: PP = m;  step = 15u; break;
        case 15u: PPP = m; step = 16u; break;
        case 16u: S = m;   step = 17u; break;
        case 17u: Q = fp_add(fp_dbl(m), m); step = 18u; break;
        case 18u:
            p31.X = fp_sub(fp_sub(m, S), S);
            step = 19u;
            break;
        case 19u:
            p31.Y = m;
            S = fp_sub(S, p31.X);
            step = 20u;
            break;
        case 20u: p31.Y = fp_sub(m, p31.Y); step = 21u; break;
        case 21u: p31.ZZ = m;  step = 22u; break;
        default:  p31.ZZZ = m; step = 23u; break;
        }
    }
}

// p31 += p2, or p31 -= p2 when `subtract` (sppark `xyzz_t::add(const
// affine_t&, bool)`): madd-2008-s, doubling mdbl-2008-s-1 when the points
// are equal, infinity when they are opposite. 8M + 2S for an addition.
//
// Single `g1_mul` call site in a `[[dont_unroll]]` loop; see `g1_add`.
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

    Fp R, P, PP, PPP, Q;
    uint step = 0u;
    [[dont_unroll]] while (step < 17u) {
        Fp ma, mb;
        switch (step) {
        case 0u:  ma = p2.y;    mb = p31.ZZZ; break;
        case 1u:  ma = p2.x;    mb = p31.ZZ;  break;
        // madd-2008-s
        case 2u:  ma = P;       mb = P;       break;
        case 3u:  ma = P;       mb = PP;      break;
        case 4u:  ma = p31.ZZ;  mb = PP;      break;
        case 5u:  ma = p31.ZZZ; mb = PPP;     break;
        case 6u:  ma = PP;      mb = p31.X;   break;
        case 7u:  ma = R;       mb = R;       break;
        case 8u:  ma = Q;       mb = R;       break;
        case 9u:  ma = p31.Y;   mb = PPP;     break;
        // mdbl-2008-s-1
        case 10u: ma = P;       mb = P;       break;
        case 11u: ma = p31.ZZ;  mb = P;       break;
        case 12u: ma = p2.x;    mb = p31.ZZ;  break;
        case 13u: ma = p2.x;    mb = p2.x;    break;
        case 14u: ma = Q;       mb = Q;       break;
        case 15u: ma = p31.ZZZ; mb = p2.y;    break;
        default:  ma = R;       mb = Q;       break;
        }
        Fp m = g1_mul(ma, mb);
        switch (step) {
        case 0u:
            R = fp_sub(fp_cneg(m, subtract), p31.Y);
            step = 1u;
            break;
        case 1u:
            P = fp_sub(m, p31.X);
            if (!fp_is_zero(P)) {
                step = 2u;
            } else if (fp_is_zero(R)) {
                P = fp_dbl(p2.y);
                step = 10u;
            } else {
                p31 = g1_inf();
                step = 17u;
            }
            break;
        // madd-2008-s
        case 2u:  PP = m;      step = 3u; break;
        case 3u:  PPP = m;     step = 4u; break;
        case 4u:  p31.ZZ = m;  step = 5u; break;
        case 5u:  p31.ZZZ = m; step = 6u; break;
        case 6u:  Q = m;       step = 7u; break;
        case 7u:
            p31.X = fp_sub(fp_sub(fp_sub(m, PPP), Q), Q);
            Q = fp_sub(Q, p31.X);
            step = 8u;
            break;
        case 8u:  Q = m; step = 9u; break;
        case 9u:  p31.Y = fp_sub(Q, m); step = 17u; break;
        // mdbl-2008-s-1
        case 10u: p31.ZZ = m;  step = 11u; break;
        case 11u: p31.ZZZ = m; step = 12u; break;
        case 12u: R = m;       step = 13u; break;
        case 13u: Q = fp_add(fp_dbl(m), m); step = 14u; break;
        case 14u: p31.X = fp_sub(fp_sub(m, R), R); step = 15u; break;
        case 15u:
            p31.Y = m;
            R = fp_sub(R, p31.X);
            step = 16u;
            break;
        default:
            p31.Y = fp_sub(m, p31.Y);
            p31.ZZZ = fp_cneg(p31.ZZZ, subtract);
            step = 17u;
            break;
        }
    }
}

#endif // MIDNIGHT_VULKAN_G1_GLSL
