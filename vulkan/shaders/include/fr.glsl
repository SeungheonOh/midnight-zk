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
// `a * 2^256 mod r`, reduced: the same 32 bytes as the host's `[u64; 4]`
// blst_fr limbs, so a std430 array of `Fr` over a buffer of host scalars
// needs no conversion pass.
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

#define MONT_T Fr
#define MONT_N 8
#define MONT_INV 0xffffffffu
#define MONT_P FR_P
#define MONT_ONE FR_ONE
#define MONT_PREFIX fr
#include "mont.glsl"
#undef MONT_T
#undef MONT_N
#undef MONT_INV
#undef MONT_P
#undef MONT_ONE
#undef MONT_PREFIX

#endif // MIDNIGHT_VULKAN_FR_GLSL
