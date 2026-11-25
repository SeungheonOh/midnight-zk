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

// The BLS12-381 scalar field `Fr` (midnight_curves::Fq, blst_fr).
//
// An `Fr` is eight little-endian 32-bit limbs of the Montgomery form
// `a * 2^256 mod r`: the same 32 bytes as the host's `[u64; 4]` blst_fr limbs,
// so host scalars need no conversion.
//
// r = 0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001

#ifndef MIDNIGHT_VULKAN_FR_GLSL
#define MIDNIGHT_VULKAN_FR_GLSL

#include "common.glsl"

struct Fr {
    uint l[8];
};

const uint FR_P[8] = uint[8](0x00000001u, 0xffffffffu, 0xfffe5bfeu, 0x53bda402u,
                             0x09a1d805u, 0x3339d808u, 0x299d7d48u, 0x73eda753u);

// 2^256 mod r
const uint FR_ONE[8] = uint[8](0xfffffffeu, 0x00000001u, 0x00034802u, 0x5884b7fau,
                               0xecbc4ff5u, 0x998c4fefu, 0xacc5056fu, 0x1824b159u);

layout(buffer_reference, std430, buffer_reference_align = 16) buffer FrRef {
    Fr v;
};

// Element i of the `Fr` array at `base`.
#define FR_AT(base, i) FrRef(addr_at(base, i, 32u)).v

#define MONT_T Fr
#define MONT_N 8
#define MONT_INV 0xffffffffu
#define MONT_P FR_P
#define MONT_ONE FR_ONE
#define MONT_PREFIX fr
#define MONT_FR_SPECIAL
#define MONT_DELAYED_13_FR
#include "mont.glsl"
#undef MONT_DELAYED_13_FR
#undef MONT_FR_SPECIAL
#undef MONT_T
#undef MONT_N
#undef MONT_INV
#undef MONT_P
#undef MONT_ONE
#undef MONT_PREFIX

// Convert out of Montgomery form (`a * 2^-256 mod r`) without the 64
// multiply-by-zero products of `fr_mul(a, 1)`.
Fr fr_from_mont(Fr t) {
    [[unroll]] for (int i = 0; i < 8; i++) {
        uint t0 = t.l[0];
        uint C;
#if MONT_INT64
        uint m = 0u - t0;
        C = t0 != 0u ? 1u : 0u;
        {
            uint64_t y1 = (uint64_t(m) << 32) - uint64_t(m) + uint64_t(t.l[1]) + uint64_t(C);
            t.l[0] = uint(y1);
            C = uint(y1 >> 32);
        }
        [[unroll]] for (int j = 2; j < 8; j++) {
            t.l[j - 1] = mad_cc(m, FR_P[j], t.l[j], C, C);
        }
#else
        uint m = t0 * 0xffffffffu;
        mad_cc(m, FR_P[0], t0, 0u, C);
        [[unroll]] for (int j = 1; j < 8; j++) {
            t.l[j - 1] = mad_cc(m, FR_P[j], t.l[j], C, C);
        }
#endif
        t.l[7] = C;
    }
    return fr_reduce_once(t);
}

#endif // MIDNIGHT_VULKAN_FR_GLSL
