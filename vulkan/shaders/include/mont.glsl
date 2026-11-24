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

// Montgomery arithmetic over a prime field of N 32-bit limbs, instantiated
// once per field: fr.glsl and fp.glsl define the parameters, include this
// file, and undefine them. It has no include guard on purpose.
//
// An element is N little-endian 32-bit limbs of the Montgomery form
// `a * 2^(32 N) mod p`, reduced (below p): the bytes of the host's N/2
// little-endian 64-bit blst limbs.
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
// CIOS multiplication; both BLS12-381 fields meet it), so that `a + b < 2p`
// fits in N limbs and the products never carry out of the top limb.
//
// The functions are those of the Metal crate's `mont.h`, operation for
// operation; the carry primitives (common.glsl) select variant A or B.

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
    // on a borrow add p back; the mask keeps the code branch free
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

// a * b: CIOS, no-carry variant, 32-bit limbs.
MONT_T MONT_FN(_mul)(MONT_T a, MONT_T b) {
    MONT_T t = MONT_FN(_zero)();
    [[unroll]] for (int i = 0; i < MONT_N; i++) {
        uint A, C;
        uint t0 = mad_cc(a.l[0], b.l[i], t.l[0], 0u, A);
        uint m = t0 * MONT_INV;
        // the low word of m * p[0] + t0 is zero
        mad_cc(m, MONT_P[0], t0, 0u, C);
        [[unroll]] for (int j = 1; j < MONT_N; j++) {
            uint x = mad_cc(a.l[j], b.l[i], t.l[j], A, A);
            t.l[j - 1] = mad_cc(m, MONT_P[j], x, C, C);
        }
        t.l[MONT_N - 1] = C + A;
    }
    return MONT_FN(_reduce_once)(t);
}

MONT_T MONT_FN(_sqr)(MONT_T a) {
    return MONT_FN(_mul)(a, a);
}

#undef MONT_FN
