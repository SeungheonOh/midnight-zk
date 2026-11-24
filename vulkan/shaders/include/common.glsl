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

// Definitions every kernel shares: the workgroup size, the global thread
// index, and the 32-bit carry primitives of the two arithmetic variants.
//
// `MONT_INT64` (set by build.rs) selects the variant:
// - 1 (variant A): 64-bit accumulators; needs `shaderInt64`.
// - 0 (variant B): 32-bit words only, `umulExtended` / `uaddCarry` /
//   `usubBorrow`; the module declares no `Int64` capability.

#ifndef MIDNIGHT_VULKAN_COMMON_GLSL
#define MIDNIGHT_VULKAN_COMMON_GLSL

#extension GL_EXT_control_flow_attributes : require

#ifndef MONT_INT64
#error "MONT_INT64 must be defined (build.rs)"
#endif

#if MONT_INT64
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

// The workgroup width is specialization constant 0; the host dispatches a
// grid of (x, y) workgroups when x alone would exceed
// `maxComputeWorkGroupCount[0]`.
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

// The linear index of this invocation in the dispatched grid.
uint global_index() {
    return gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) +
           gl_GlobalInvocationID.x;
}

// ---------------------------------------------------------------------------
// Carry primitives. Every `carry` / `borrow` is 0 or 1.

#if MONT_INT64

// a + b + carry; carry out.
uint add_cc(uint a, uint b, inout uint carry) {
    uint64_t x = uint64_t(a) + b + carry;
    carry = uint(x >> 32);
    return uint(x);
}

// a - b - borrow; borrow out.
uint sub_bb(uint a, uint b, inout uint borrow) {
    uint64_t x = uint64_t(a) - b - borrow;
    borrow = uint(x >> 63);
    return uint(x);
}

// The low word of a * b + c + d (below 2^64); the high word in `hi`.
uint mad_cc(uint a, uint b, uint c, uint d, out uint hi) {
    uint64_t x = uint64_t(a) * b + c + d;
    hi = uint(x >> 32);
    return uint(x);
}

#else

uint add_cc(uint a, uint b, inout uint carry) {
    uint c1, c2;
    uint s = uaddCarry(a, b, c1);
    // a + b overflowed: s <= 2^32 - 2, so adding the carry cannot overflow
    s = uaddCarry(s, carry, c2);
    carry = c1 + c2;
    return s;
}

uint sub_bb(uint a, uint b, inout uint borrow) {
    uint b1, b2;
    uint d = usubBorrow(a, b, b1);
    // a - b borrowed: d >= 1, so subtracting the borrow cannot borrow again
    d = usubBorrow(d, borrow, b2);
    borrow = b1 + b2;
    return d;
}

uint mad_cc(uint a, uint b, uint c, uint d, out uint hi) {
    uint lo, c1, c2;
    umulExtended(a, b, hi, lo);
    lo = uaddCarry(lo, c, c1);
    lo = uaddCarry(lo, d, c2);
    // a * b + c + d < 2^64: the high word does not overflow
    hi += c1 + c2;
    return lo;
}

#endif

#endif // MIDNIGHT_VULKAN_COMMON_GLSL
