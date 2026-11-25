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
}

#undef MONT_CIOS_STEP

MONT_T MONT_FN(_mul)(MONT_T a, MONT_T b) {
    return MONT_FN(_reduce_once)(MONT_FN(_mul_lazy)(a, b));
}

MONT_T MONT_FN(_sqr)(MONT_T a) {
    return MONT_FN(_mul)(a, a);
}

#undef MONT_FN
