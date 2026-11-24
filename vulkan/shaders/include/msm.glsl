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

// The skew mask of the MSM: a fixed public pseudo-random vector `r` that
// depends only on the index. Element i, limb j (64-bit, Montgomery limbs used
// as they are) is the splitmix64 output number 4 i + j + 1 from the seed
// below, limb 3 shifted right by 3 so that every element is below the scalar
// modulus. It is the CUDA crate's `mask_vector` (gpu/src/msm.rs), computed
// per element as in the Metal crate's `msm.h`.
//
// A 64-bit value is a `uvec2` (low word, high word) in both variants; variant
// A computes with `uint64_t`, variant B with 32-bit word pairs.

#ifndef MIDNIGHT_VULKAN_MSM_GLSL
#define MIDNIGHT_VULKAN_MSM_GLSL

#include "fr.glsl"

const uvec2 MASK_SEED = uvec2(0x4f6cdd1du, 0x2545f491u);
const uvec2 MASK_GAMMA = uvec2(0x7f4a7c15u, 0x9e3779b9u);
const uvec2 MASK_MUL1 = uvec2(0x1ce4e5b9u, 0xbf58476du);
const uvec2 MASK_MUL2 = uvec2(0x133111ebu, 0x94d049bbu);

#if MONT_INT64

uint64_t to_u64(uvec2 a) {
    return uint64_t(a.x) | (uint64_t(a.y) << 32);
}

// Output number k (from 1) of splitmix64 started at MASK_SEED.
uvec2 splitmix(uvec2 k) {
    uint64_t z = to_u64(MASK_SEED) + to_u64(k) * to_u64(MASK_GAMMA);
    z = (z ^ (z >> 30)) * to_u64(MASK_MUL1);
    z = (z ^ (z >> 27)) * to_u64(MASK_MUL2);
    z ^= z >> 31;
    return uvec2(uint(z), uint(z >> 32));
}

#else

// a ^ (a >> s) for 0 < s < 32.
uvec2 xorshr64(uvec2 a, uint s) {
    return a ^ uvec2((a.x >> s) | (a.y << (32u - s)), a.y >> s);
}

uvec2 splitmix(uvec2 k) {
    uvec2 z = add64(MASK_SEED, mul64(k, MASK_GAMMA));
    z = mul64(xorshr64(z, 30u), MASK_MUL1);
    z = mul64(xorshr64(z, 27u), MASK_MUL2);
    return xorshr64(z, 31u);
}

#endif

// The mask element of index i (a 64-bit index).
Fr mask_elem(uvec2 i) {
    Fr r;
    // 4 i mod 2^64
    uvec2 i4 = uvec2(i.x << 2, (i.y << 2) | (i.x >> 30));
    [[unroll]] for (int j = 0; j < 4; j++) {
        uint c;
        uint lo = uaddCarry(i4.x, uint(j + 1), c);
        uvec2 w = splitmix(uvec2(lo, i4.y + c));
        if (j == 3) w = uvec2((w.x >> 3) | (w.y << 29), w.y >> 3);
        r.l[2 * j] = w.x;
        r.l[2 * j + 1] = w.y;
    }
    return r;
}

#endif // MIDNIGHT_VULKAN_MSM_GLSL
