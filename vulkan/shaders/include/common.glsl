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

// `MONT_INT64` (set by build.rs) selects the arithmetic variant:
// - 1 (variant A): 64-bit accumulators; needs `shaderInt64`.
// - 0 (variant B): 32-bit words only; the module declares no `Int64` capability.
//
// A kernel binds no descriptor: its push-constant block holds the device
// addresses of its buffers (`uvec2`: low word, high word), then its scalar
// parameters. Element addresses are computed in 64 bits from 32-bit words
// (`addr_at`), so buffers may exceed 4 GiB in both variants.

#ifndef MIDNIGHT_VULKAN_COMMON_GLSL
#define MIDNIGHT_VULKAN_COMMON_GLSL

#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require

#ifndef MONT_INT64
#error "MONT_INT64 must be defined (build.rs)"
#endif

#if MONT_INT64
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#endif

// The workgroup width is specialization constant 0. The host folds a grid of
// more than `maxComputeWorkGroupCount[0]` workgroups into (x, y).
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

#define CAT_(a, b) a##b
#define CAT(a, b) CAT_(a, b)

// The linear index of this invocation in the dispatched grid.
uint global_index() {
    return gl_GlobalInvocationID.y * (gl_NumWorkGroups.x * gl_WorkGroupSize.x) +
           gl_GlobalInvocationID.x;
}

// The linear index of this workgroup.
uint group_index() {
    return gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
}

// The invocations of the grid.
uint grid_threads() {
    return gl_NumWorkGroups.x * gl_NumWorkGroups.y * gl_WorkGroupSize.x;
}

// 64-bit values as (low, high) word pairs.

uvec2 add64(uvec2 a, uvec2 b) {
    uint c;
    uint lo = uaddCarry(a.x, b.x, c);
    return uvec2(lo, a.y + b.y + c);
}

// a * b mod 2^64.
uvec2 mul64(uvec2 a, uvec2 b) {
    uint hi, lo;
    umulExtended(a.x, b.x, hi, lo);
    return uvec2(lo, hi + a.x * b.y + a.y * b.x);
}

// The address of element i of `size` bytes from `base`.
uvec2 addr_at(uvec2 base, uint i, uint size) {
    uvec2 off;
    umulExtended(i, size, off.y, off.x);
    return add64(base, off);
}

// The address of element i (a 64-bit index) of `size` bytes from `base`.
uvec2 addr_at64(uvec2 base, uvec2 i, uint size) {
    return add64(base, mul64(i, uvec2(size, 0u)));
}

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Ref {
    uint v;
};

layout(buffer_reference, std430, buffer_reference_align = 8) buffer U32x2Ref {
    uvec2 v;
};

#define U32_AT(base, i) U32Ref(addr_at(base, i, 4u)).v
#define U32X2_AT(base, i) U32x2Ref(addr_at(base, i, 8u)).v

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
