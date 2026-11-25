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

// Montgomery arithmetic over a prime field of N 32-bit limbs, included once
// per field by fr.glsl and fp.glsl (so no include guard). An element is
// reduced (below p).
//
// Parameters:
//   MONT_T       the element struct, `struct { uint l[MONT_N]; }`
//   MONT_N       number of 32-bit limbs
//   MONT_INV     -p^-1 mod 2^32
//   MONT_P       a `const uint[MONT_N]` of the limbs of p
//   MONT_ONE     a `const uint[MONT_N]` of 2^(32 N) mod p
//   MONT_PREFIX  the function name prefix (`fr`, `fp`)
//
// The top limb of p must be below 2^31 - 1 (the "no-carry" condition of the
// CIOS multiplication), so that `a + b < 2p` fits in N limbs and the products
// never carry out of the top limb.
//
// The functions follow the Metal crate's `mont.h` operation for operation.

#define MONT_FN(name) CAT(MONT_PREFIX, name)

MONT_T MONT_FN(_zero)() {
    MONT_T r;
    [[unroll]] for (int i = 0; i < MONT_N; i++) r.l[i] = 0u;
    return r;
}

// The Montgomery form of 1.
MONT_T MONT_FN(_one)() {
    MONT_T r;
    [[unroll]] for (int i = 0; i < MONT_N; i++) r.l[i] = MONT_ONE[i];
    return r;
}

bool MONT_FN(_is_zero)(MONT_T a) {
    uint acc = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) acc |= a.l[i];
    return acc == 0u;
}

bool MONT_FN(_eq)(MONT_T a, MONT_T b) {
    uint acc = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) acc |= a.l[i] ^ b.l[i];
    return acc == 0u;
}

// a < p.
bool MONT_FN(_is_reduced)(MONT_T a) {
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) sub_bb(a.l[i], MONT_P[i], borrow);
    return borrow != 0u;
}

// t - p if t >= p, else t (t < 2p).
MONT_T MONT_FN(_reduce_once)(MONT_T t) {
    MONT_T d;
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) d.l[i] = sub_bb(t.l[i], MONT_P[i], borrow);
    bool keep = borrow != 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) d.l[i] = keep ? t.l[i] : d.l[i];
    return d;
}

MONT_T MONT_FN(_add)(MONT_T a, MONT_T b) {
    MONT_T s;
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) s.l[i] = add_cc(a.l[i], b.l[i], carry);
    // a + b < 2p < 2^(32 N): no carry out of the top limb
    return MONT_FN(_reduce_once)(s);
}

MONT_T MONT_FN(_dbl)(MONT_T a) {
    MONT_T s;
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) {
        s.l[i] = (a.l[i] << 1) | carry;
        carry = a.l[i] >> 31;
    }
    return MONT_FN(_reduce_once)(s);
}

MONT_T MONT_FN(_sub)(MONT_T a, MONT_T b) {
    MONT_T d;
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) d.l[i] = sub_bb(a.l[i], b.l[i], borrow);
    // on a borrow add p back, branch free
    uint mask = 0u - borrow;
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) d.l[i] = add_cc(d.l[i], MONT_P[i] & mask, carry);
    return d;
}

MONT_T MONT_FN(_neg)(MONT_T a) {
    MONT_T d;
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) d.l[i] = sub_bb(MONT_P[i], a.l[i], borrow);
    // -0 = 0
    uint mask = MONT_FN(_is_zero)(a) ? 0u : ~0u;
    [[unroll]] for (int i = 0; i < MONT_N; i++) d.l[i] &= mask;
    return d;
}

#if defined(MONT_FR_SPECIAL) && MONT_INT64
#define MONT_CIOS_STEP(bi_val)                                                                     \
    do {                                                                                           \
        uint _bi = (bi_val);                                                                       \
        uint A, C;                                                                                 \
        uint t0 = mad_cc(a.l[0], _bi, t.l[0], 0u, A);                                              \
        uint m = 0u - t0;                                                                          \
        C = t0 != 0u ? 1u : 0u;                                                                    \
        {                                                                                          \
            uint x1 = mad_cc(a.l[1], _bi, t.l[1], A, A);                                           \
            uint64_t y1 = (uint64_t(m) << 32) - uint64_t(m) + uint64_t(x1) + uint64_t(C);          \
            t.l[0] = uint(y1);                                                                     \
            C = uint(y1 >> 32);                                                                    \
        }                                                                                          \
        [[unroll]] for (int j = 2; j < MONT_N; j++) {                                              \
            uint x = mad_cc(a.l[j], _bi, t.l[j], A, A);                                            \
            t.l[j - 1] = mad_cc(m, MONT_P[j], x, C, C);                                            \
        }                                                                                          \
        t.l[MONT_N - 1] = C + A;                                                                   \
    } while (false)
#else
#define MONT_CIOS_STEP(bi_val)                                                                     \
    do {                                                                                           \
        uint _bi = (bi_val);                                                                       \
        uint A, C;                                                                                 \
        uint t0 = mad_cc(a.l[0], _bi, t.l[0], 0u, A);                                              \
        uint m = t0 * MONT_INV;                                                                    \
        mad_cc(m, MONT_P[0], t0, 0u, C);                                                           \
        [[unroll]] for (int j = 1; j < MONT_N; j++) {                                              \
            uint x = mad_cc(a.l[j], _bi, t.l[j], A, A);                                            \
            t.l[j - 1] = mad_cc(m, MONT_P[j], x, C, C);                                            \
        }                                                                                          \
        t.l[MONT_N - 1] = C + A;                                                                   \
    } while (false)
#endif

// a * b (unreduced, < 2p for BLS12-381 Fp when a < 4p and b < 2p): CIOS,
// no-carry variant, 32-bit limbs.
// For 12-limb Fp (`MONT_N > 8`) or `MONT_ROLL_OUTER`, keep the outer loop
// rolled 1x with register rotation of `b` (no dynamic array indexing): keeps
// code size minimal so drivers that ignore SPIR-V `DontInline` (e.g. Qualcomm
// Adreno) compile fast without register spilling or I-cache thrashing.
MONT_T MONT_FN(_mul_lazy)(MONT_T a, MONT_T b) {
#if defined(MONT_DELAYED_13_FP) && MONT_INT64
    // 13-bit delayed-carry Montgomery multiply for BLS12-381 Fp (30 x 13 = 390 bits):
    // multiplies (a << 6) * b * 2^-390 = a * b * 2^-384 mod p. Zero uint64_t, zero
    // inner-loop carries (strict upper bound 0.739 * 2^32 < 2^32), native 1-cycle
    // mad.u16 on Adreno.
    const uint16_t FP_P13[30] = uint16_t[30](
        uint16_t(0x0aabu), uint16_t(0x1ffdu), uint16_t(0x1fffu), uint16_t(0x1dffu),
        uint16_t(0x1b9fu), uint16_t(0x1fffu), uint16_t(0x054fu), uint16_t(0x1fd6u),
        uint16_t(0x0bffu), uint16_t(0x00f5u), uint16_t(0x1d89u), uint16_t(0x0d61u),
        uint16_t(0x0a0fu), uint16_t(0x1869u), uint16_t(0x1d9cu), uint16_t(0x0257u),
        uint16_t(0x1385u), uint16_t(0x1c27u), uint16_t(0x1dd2u), uint16_t(0x0ec8u),
        uint16_t(0x1acdu), uint16_t(0x01a5u), uint16_t(0x1ed9u), uint16_t(0x0374u),
        uint16_t(0x1a4bu), uint16_t(0x1f34u), uint16_t(0x0e5fu), uint16_t(0x03d4u),
        uint16_t(0x0011u), uint16_t(0x000du));
    uint16_t a13[30];
    a13[0] = uint16_t((a.l[0] & 0x7fu) << 6u);
    a13[1] = uint16_t((a.l[0] >> 7u) & 0x1fffu);
    a13[2] = uint16_t((a.l[0] >> 20u) | ((a.l[1] & 0x1u) << 12u));
    a13[3] = uint16_t((a.l[1] >> 1u) & 0x1fffu);
    a13[4] = uint16_t((a.l[1] >> 14u) & 0x1fffu);
    a13[5] = uint16_t((a.l[1] >> 27u) | ((a.l[2] & 0xffu) << 5u));
    a13[6] = uint16_t((a.l[2] >> 8u) & 0x1fffu);
    a13[7] = uint16_t((a.l[2] >> 21u) | ((a.l[3] & 0x3u) << 11u));
    a13[8] = uint16_t((a.l[3] >> 2u) & 0x1fffu);
    a13[9] = uint16_t((a.l[3] >> 15u) & 0x1fffu);
    a13[10] = uint16_t((a.l[3] >> 28u) | ((a.l[4] & 0x1ffu) << 4u));
    a13[11] = uint16_t((a.l[4] >> 9u) & 0x1fffu);
    a13[12] = uint16_t((a.l[4] >> 22u) | ((a.l[5] & 0x7u) << 10u));
    a13[13] = uint16_t((a.l[5] >> 3u) & 0x1fffu);
    a13[14] = uint16_t((a.l[5] >> 16u) & 0x1fffu);
    a13[15] = uint16_t((a.l[5] >> 29u) | ((a.l[6] & 0x3ffu) << 3u));
    a13[16] = uint16_t((a.l[6] >> 10u) & 0x1fffu);
    a13[17] = uint16_t((a.l[6] >> 23u) | ((a.l[7] & 0xfu) << 9u));
    a13[18] = uint16_t((a.l[7] >> 4u) & 0x1fffu);
    a13[19] = uint16_t((a.l[7] >> 17u) & 0x1fffu);
    a13[20] = uint16_t((a.l[7] >> 30u) | ((a.l[8] & 0x7ffu) << 2u));
    a13[21] = uint16_t((a.l[8] >> 11u) & 0x1fffu);
    a13[22] = uint16_t((a.l[8] >> 24u) | ((a.l[9] & 0x1fu) << 8u));
    a13[23] = uint16_t((a.l[9] >> 5u) & 0x1fffu);
    a13[24] = uint16_t((a.l[9] >> 18u) & 0x1fffu);
    a13[25] = uint16_t((a.l[9] >> 31u) | ((a.l[10] & 0xfffu) << 1u));
    a13[26] = uint16_t((a.l[10] >> 12u) & 0x1fffu);
    a13[27] = uint16_t((a.l[10] >> 25u) | ((a.l[11] & 0x3fu) << 7u));
    a13[28] = uint16_t((a.l[11] >> 6u) & 0x1fffu);
    a13[29] = uint16_t(a.l[11] >> 19u);

#if defined(MONT_ROLL_OUTER_2X)
    uint b13p[15];
    b13p[0] = (b.l[0] & 0x1fffu) | (((b.l[0] >> 13u) & 0x1fffu) << 16u);
    b13p[1] = ((b.l[0] >> 26u) | ((b.l[1] & 0x7fu) << 6u)) | (((b.l[1] >> 7u) & 0x1fffu) << 16u);
    b13p[2] = ((b.l[1] >> 20u) | ((b.l[2] & 0x1u) << 12u)) | (((b.l[2] >> 1u) & 0x1fffu) << 16u);
    b13p[3] = ((b.l[2] >> 14u) & 0x1fffu) | (((b.l[2] >> 27u) | ((b.l[3] & 0xffu) << 5u)) << 16u);
    b13p[4] = ((b.l[3] >> 8u) & 0x1fffu) | (((b.l[3] >> 21u) | ((b.l[4] & 0x3u) << 11u)) << 16u);
    b13p[5] = ((b.l[4] >> 2u) & 0x1fffu) | (((b.l[4] >> 15u) & 0x1fffu) << 16u);
    b13p[6] = ((b.l[4] >> 28u) | ((b.l[5] & 0x1ffu) << 4u)) | (((b.l[5] >> 9u) & 0x1fffu) << 16u);
    b13p[7] = ((b.l[5] >> 22u) | ((b.l[6] & 0x7u) << 10u)) | (((b.l[6] >> 3u) & 0x1fffu) << 16u);
    b13p[8] = ((b.l[6] >> 16u) & 0x1fffu) | (((b.l[6] >> 29u) | ((b.l[7] & 0x3ffu) << 3u)) << 16u);
    b13p[9] = ((b.l[7] >> 10u) & 0x1fffu) | (((b.l[7] >> 23u) | ((b.l[8] & 0xfu) << 9u)) << 16u);
    b13p[10] = ((b.l[8] >> 4u) & 0x1fffu) | (((b.l[8] >> 17u) & 0x1fffu) << 16u);
    b13p[11] = ((b.l[8] >> 30u) | ((b.l[9] & 0x7ffu) << 2u)) | (((b.l[9] >> 11u) & 0x1fffu) << 16u);
    b13p[12] = ((b.l[9] >> 24u) | ((b.l[10] & 0x1fu) << 8u)) | (((b.l[10] >> 5u) & 0x1fffu) << 16u);
    b13p[13] = ((b.l[10] >> 18u) & 0x1fffu) | (((b.l[10] >> 31u) | ((b.l[11] & 0xfffu) << 1u)) << 16u);
    b13p[14] = ((b.l[11] >> 12u) & 0x1fffu) | ((b.l[11] >> 25u) << 16u);
#endif

    uint t[29];
    [[unroll]] for (int j = 0; j < 29; j++) t[j] = 0u;

#define FP13_STEP(bi_expr)                                                                         \
    do {                                                                                           \
        uint16_t _bi = uint16_t(bi_expr);                                                          \
        uint _t0 = t[0] + uint(a13[0]) * uint(_bi);                                                \
        uint16_t _m = uint16_t((0u - 3u * _t0) & 0x1fffu);                                         \
        uint _c0 = (_t0 + uint(_m) * uint(FP_P13[0])) >> 13u;                                      \
        t[0] = t[1] + uint(a13[1]) * uint(_bi) + uint(_m) * uint(FP_P13[1]) + _c0;                 \
        [[unroll]] for (int j = 2; j < 29; j++) {                                                  \
            t[j - 1] = t[j] + uint(a13[j]) * uint(_bi) + uint(_m) * uint(FP_P13[j]);               \
        }                                                                                          \
        t[28] = uint(a13[29]) * uint(_bi) + uint(_m) * uint(FP_P13[29]);                           \
    } while (false)

#if defined(MONT_ROLL_OUTER_2X)
    [[dont_unroll]] for (int i = 0; i < 15; i++) {
        uint bw = b13p[0];
        [[unroll]] for (int k = 0; k < 14; k++) b13p[k] = b13p[k + 1];
        FP13_STEP(bw & 0xffffu);
        FP13_STEP(bw >> 16u);
    }
#else
    [[dont_unroll]] for (int i = 0; i < 30; i++) {
        uint bi = b.l[0] & 0x1fffu;
        [[unroll]] for (int k = 0; k < 11; k++) b.l[k] = (b.l[k] >> 13u) | (b.l[k + 1] << 19u);
        b.l[11] >>= 13u;
        FP13_STEP(bi);
    }
#endif
#undef FP13_STEP

    uint c = 0u;
    [[unroll]] for (int j = 0; j < 29; j++) {
        uint v = t[j] + c;
        t[j] = v & 0x1fffu;
        c = v >> 13u;
    }
    MONT_T r;
    r.l[0] = t[0] | (t[1] << 13u) | (t[2] << 26u);
    r.l[1] = (t[2] >> 6u) | (t[3] << 7u) | (t[4] << 20u);
    r.l[2] = (t[4] >> 12u) | (t[5] << 1u) | (t[6] << 14u) | (t[7] << 27u);
    r.l[3] = (t[7] >> 5u) | (t[8] << 8u) | (t[9] << 21u);
    r.l[4] = (t[9] >> 11u) | (t[10] << 2u) | (t[11] << 15u) | (t[12] << 28u);
    r.l[5] = (t[12] >> 4u) | (t[13] << 9u) | (t[14] << 22u);
    r.l[6] = (t[14] >> 10u) | (t[15] << 3u) | (t[16] << 16u) | (t[17] << 29u);
    r.l[7] = (t[17] >> 3u) | (t[18] << 10u) | (t[19] << 23u);
    r.l[8] = (t[19] >> 9u) | (t[20] << 4u) | (t[21] << 17u) | (t[22] << 30u);
    r.l[9] = (t[22] >> 2u) | (t[23] << 11u) | (t[24] << 24u);
    r.l[10] = (t[24] >> 8u) | (t[25] << 5u) | (t[26] << 18u) | (t[27] << 31u);
    r.l[11] = (t[27] >> 1u) | (t[28] << 12u) | (c << 25u);
    return r;
#elif defined(MONT_DELAYED_13_FR) && MONT_INT64
    // 13-bit delayed-carry Montgomery multiply for BLS12-381 Fr (20 x 13 = 260 bits):
    // multiplies (a << 4) * b * 2^-260 = a * b * 2^-256 mod r. Zero uint64_t, zero
    // inner-loop carries (strict upper bound 0.45 * 2^32), FR_P13[0] = 1 and FR_P13[1] = 0.
    const uint FR_P13[20] = uint[20](
        0x0001u, 0x0000u, 0x1fc0u, 0x1fffu, 0x0fffu, 0x0dffu, 0x1ff9u, 0x005fu,
        0x1da4u, 0x0a9du, 0x1601u, 0x1343u, 0x0080u, 0x1cecu, 0x00ccu, 0x0fa9u,
        0x099du, 0x1a99u, 0x1b69u, 0x00e7u);
    uint a13[20];
    a13[0] = ((a.l[0] & 0x1ffu) << 4u);
    a13[1] = ((a.l[0] >> 9u) & 0x1fffu);
    a13[2] = (a.l[0] >> 22u) | ((a.l[1] & 0x7u) << 10u);
    a13[3] = ((a.l[1] >> 3u) & 0x1fffu);
    a13[4] = ((a.l[1] >> 16u) & 0x1fffu);
    a13[5] = (a.l[1] >> 29u) | ((a.l[2] & 0x3ffu) << 3u);
    a13[6] = ((a.l[2] >> 10u) & 0x1fffu);
    a13[7] = (a.l[2] >> 23u) | ((a.l[3] & 0xfu) << 9u);
    a13[8] = ((a.l[3] >> 4u) & 0x1fffu);
    a13[9] = ((a.l[3] >> 17u) & 0x1fffu);
    a13[10] = (a.l[3] >> 30u) | ((a.l[4] & 0x7ffu) << 2u);
    a13[11] = ((a.l[4] >> 11u) & 0x1fffu);
    a13[12] = (a.l[4] >> 24u) | ((a.l[5] & 0x1fu) << 8u);
    a13[13] = ((a.l[5] >> 5u) & 0x1fffu);
    a13[14] = ((a.l[5] >> 18u) & 0x1fffu);
    a13[15] = (a.l[5] >> 31u) | ((a.l[6] & 0xfffu) << 1u);
    a13[16] = ((a.l[6] >> 12u) & 0x1fffu);
    a13[17] = (a.l[6] >> 25u) | ((a.l[7] & 0x3fu) << 7u);
    a13[18] = ((a.l[7] >> 6u) & 0x1fffu);
    a13[19] = (a.l[7] >> 19u);

#if defined(MONT_ROLL_OUTER_2X)
    uint b13p[10];
    b13p[0] = (b.l[0] & 0x1fffu) | (((b.l[0] >> 13u) & 0x1fffu) << 16u);
    b13p[1] = ((b.l[0] >> 26u) | ((b.l[1] & 0x7fu) << 6u)) | (((b.l[1] >> 7u) & 0x1fffu) << 16u);
    b13p[2] = ((b.l[1] >> 20u) | ((b.l[2] & 0x1u) << 12u)) | (((b.l[2] >> 1u) & 0x1fffu) << 16u);
    b13p[3] = ((b.l[2] >> 14u) & 0x1fffu) | (((b.l[2] >> 27u) | ((b.l[3] & 0xffu) << 5u)) << 16u);
    b13p[4] = ((b.l[3] >> 8u) & 0x1fffu) | (((b.l[3] >> 21u) | ((b.l[4] & 0x3u) << 11u)) << 16u);
    b13p[5] = ((b.l[4] >> 2u) & 0x1fffu) | (((b.l[4] >> 15u) & 0x1fffu) << 16u);
    b13p[6] = ((b.l[4] >> 28u) | ((b.l[5] & 0x1ffu) << 4u)) | (((b.l[5] >> 9u) & 0x1fffu) << 16u);
    b13p[7] = ((b.l[5] >> 22u) | ((b.l[6] & 0x7u) << 10u)) | (((b.l[6] >> 3u) & 0x1fffu) << 16u);
    b13p[8] = ((b.l[6] >> 16u) & 0x1fffu) | (((b.l[6] >> 29u) | ((b.l[7] & 0x3ffu) << 3u)) << 16u);
    b13p[9] = ((b.l[7] >> 10u) & 0x1fffu) | ((b.l[7] >> 23u) << 16u);
#endif

    uint t[19];
    [[unroll]] for (int j = 0; j < 19; j++) t[j] = 0u;

#define FR13_STEP(bi_expr)                                                                         \
    do {                                                                                           \
        uint _bi = (bi_expr);                                                                      \
        uint _t0 = t[0] + a13[0] * _bi;                                                            \
        uint _m = (0u - _t0) & 0x1fffu;                                                            \
        uint _c0 = (_t0 + _m) >> 13u;                                                              \
        t[0] = t[1] + a13[1] * _bi + _c0;                                                          \
        [[unroll]] for (int j = 2; j < 19; j++) {                                                  \
            t[j - 1] = t[j] + a13[j] * _bi + _m * FR_P13[j];                                       \
        }                                                                                          \
        t[18] = a13[19] * _bi + _m * FR_P13[19];                                                   \
    } while (false)

#if defined(MONT_ROLL_OUTER_2X)
    [[dont_unroll]] for (int i = 0; i < 10; i++) {
        uint bw = b13p[0];
        [[unroll]] for (int k = 0; k < 9; k++) b13p[k] = b13p[k + 1];
        FR13_STEP(bw & 0xffffu);
        FR13_STEP(bw >> 16u);
    }
#else
    [[dont_unroll]] for (int i = 0; i < 20; i++) {
        uint bi = b.l[0] & 0x1fffu;
        [[unroll]] for (int k = 0; k < 7; k++) b.l[k] = (b.l[k] >> 13u) | (b.l[k + 1] << 19u);
        b.l[7] >>= 13u;
        FR13_STEP(bi);
    }
#endif
#undef FR13_STEP

    uint c = 0u;
    [[unroll]] for (int j = 0; j < 19; j++) {
        uint v = t[j] + c;
        t[j] = v & 0x1fffu;
        c = v >> 13u;
    }
    MONT_T r;
    r.l[0] = t[0] | (t[1] << 13u) | (t[2] << 26u);
    r.l[1] = (t[2] >> 6u) | (t[3] << 7u) | (t[4] << 20u);
    r.l[2] = (t[4] >> 12u) | (t[5] << 1u) | (t[6] << 14u) | (t[7] << 27u);
    r.l[3] = (t[7] >> 5u) | (t[8] << 8u) | (t[9] << 21u);
    r.l[4] = (t[9] >> 11u) | (t[10] << 2u) | (t[11] << 15u) | (t[12] << 28u);
    r.l[5] = (t[12] >> 4u) | (t[13] << 9u) | (t[14] << 22u);
    r.l[6] = (t[14] >> 10u) | (t[15] << 3u) | (t[16] << 16u) | (t[17] << 29u);
    r.l[7] = (t[17] >> 3u) | (t[18] << 10u) | (c << 23u);
    return r;
#else
    MONT_T t = MONT_FN(_zero)();
#if defined(MONT_ROLL_OUTER_2X)
    [[dont_unroll]] for (int i = 0; i < MONT_N / 2; i++) {
        uint b0 = b.l[0];
        uint b1 = b.l[1];
        [[unroll]] for (int k = 0; k < MONT_N - 2; k++) b.l[k] = b.l[k + 2];
        MONT_CIOS_STEP(b0);
        MONT_CIOS_STEP(b1);
    }
#elif MONT_N > 8 || defined(MONT_ROLL_OUTER)
    [[dont_unroll]] for (int i = 0; i < MONT_N; i++) {
        uint bi = b.l[0];
        [[unroll]] for (int k = 0; k < MONT_N - 1; k++) b.l[k] = b.l[k + 1];
        MONT_CIOS_STEP(bi);
    }
#else
    [[unroll]] for (int i = 0; i < MONT_N; i++) {
        MONT_CIOS_STEP(b.l[i]);
    }
#endif
    return t;
#endif
}

// Compact 32-bit rolled CIOS multiplication for cold doubling branches so
// drivers that inline all calls compile fast without inflating hot kernels.
MONT_T MONT_FN(_mul_compact)(MONT_T a, MONT_T b) {
    MONT_T t = MONT_FN(_zero)();
    [[dont_unroll]] for (int i = 0; i < MONT_N; i++) {
        uint bi = b.l[0];
        [[unroll]] for (int k = 0; k < MONT_N - 1; k++) b.l[k] = b.l[k + 1];
        MONT_CIOS_STEP(bi);
    }
    return t;
}

#undef MONT_CIOS_STEP

MONT_T MONT_FN(_mul)(MONT_T a, MONT_T b) {
    return MONT_FN(_reduce_once)(MONT_FN(_mul_lazy)(a, b));
}

MONT_T MONT_FN(_sqr)(MONT_T a) {
    return MONT_FN(_mul)(a, a);
}

#undef MONT_FN
