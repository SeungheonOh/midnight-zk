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

// The BLS12-381 base field `Fp` (blst_fp).
//
// An `Fp` is twelve little-endian 32-bit limbs of the Montgomery form
// `a * 2^384 mod p`: the same 48 bytes as the host's `[u64; 6]` blst_fp limbs.
// A host `blst_p1_affine` is `x` then `y`, two `Fp`.
//
// p = 0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab

#ifndef MIDNIGHT_VULKAN_FP_GLSL
#define MIDNIGHT_VULKAN_FP_GLSL

#include "common.glsl"

struct Fp {
    uint l[12];
};

const uint FP_P[12] = uint[12](0xffffaaabu, 0xb9feffffu, 0xb153ffffu, 0x1eabfffeu,
                               0xf6b0f624u, 0x6730d2a0u, 0xf38512bfu, 0x64774b84u,
                               0x434bacd7u, 0x4b1ba7b6u, 0x397fe69au, 0x1a0111eau);

// 2 * p (BLS12-381 Fp has 3 spare bits in the top limb, so 2p < 2^382 and
// lazy-reduced values in [0, 2p) stay strictly < 2p through CIOS fp_mul_lazy).
const uint FP_2P[12] = uint[12](0xffff5556u, 0x73fdffffu, 0x62a7ffffu, 0x3d57fffdu,
                                0xed61ec48u, 0xce61a541u, 0xe70a257eu, 0xc8ee9709u,
                                0x869759aeu, 0x96374f6cu, 0x72ffcd34u, 0x340223d4u);

// 2^384 mod p
const uint FP_ONE[12] = uint[12](0x0002fffdu, 0x76090000u, 0xc40c0002u, 0xebf4000bu,
                                 0x53c758bau, 0x5f489857u, 0x70525745u, 0x77ce5853u,
                                 0xa256ec6du, 0x5c071a97u, 0xfa80e493u, 0x15f65ec3u);

layout(buffer_reference, std430, buffer_reference_align = 16) buffer FpRef {
    Fp v;
};

// Element i of the `Fp` array at `base`.
#define FP_AT(base, i) FpRef(addr_at(base, i, 48u)).v

#define MONT_T Fp
#define MONT_N 12
#define MONT_INV 0xfffcfffdu
#define MONT_P FP_P
#define MONT_ONE FP_ONE
#define MONT_PREFIX fp
#if !defined(MSM_FALLBACK_COMPACT)
#define MONT_DELAYED_13_FP
#endif
#include "mont.glsl"
#undef MONT_DELAYED_13_FP
#undef MONT_T
#undef MONT_N
#undef MONT_INV
#undef MONT_P
#undef MONT_ONE
#undef MONT_PREFIX

// Lazy subtraction in [0, 2p): a - b + (borrow ? 2p : 0) in [0, 2p).
Fp fp_sub_2p(Fp a, Fp b) {
    Fp d;
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < 12; i++) d.l[i] = sub_bb(a.l[i], b.l[i], borrow);
    uint mask = 0u - borrow;
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < 12; i++) d.l[i] = add_cc(d.l[i], FP_2P[i] & mask, carry);
    return d;
}

// Branchless unreduced subtraction for a, b in [0, 2p): (a + 2p) - b in (0, 4p).
// Valid directly as an operand to `fp_mul_lazy` when the other operand is < 2p.
Fp fp_sub_unred(Fp a, Fp b) {
    Fp s;
    uint carry = 0u;
    [[unroll]] for (int i = 0; i < 12; i++) s.l[i] = add_cc(a.l[i], FP_2P[i], carry);
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < 12; i++) s.l[i] = sub_bb(s.l[i], b.l[i], borrow);
    return s;
}

// Lazy negation in [0, 2p): 2p - a (with -0 = 0), result in [0, 2p).
Fp fp_neg_2p(Fp a) {
    Fp d;
    uint borrow = 0u;
    [[unroll]] for (int i = 0; i < 12; i++) d.l[i] = sub_bb(FP_2P[i], a.l[i], borrow);
    uint mask = fp_is_zero(a) ? 0u : ~0u;
    [[unroll]] for (int i = 0; i < 12; i++) d.l[i] &= mask;
    return d;
}

#endif // MIDNIGHT_VULKAN_FP_GLSL
