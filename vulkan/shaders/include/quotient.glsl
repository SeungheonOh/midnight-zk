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

// The tables of the quotient numerator kernel (quotient.comp), as the Metal
// crate's `quotient.h` defines them. Every offset is a word offset into the
// kernel's `words` buffer (32-bit words), except a program's uniforms, which
// index the `uniforms` buffer (field elements).
//
// Handles: `(bank << HANDLE_SHIFT) | index`, the extended coset `index` of a
// bank (BANK_*).
//
// A program (PROG_WORDS words):
//   [ops, nops, reads, nreads, uniforms, outputs, noutputs, 0]
//   ops:     nops x OP_WORDS words  [kind, dst slot, a, b, c] (operands as
//            midnight_proofs::gpu::operand: (tag << 30) | index)
//   reads:   nreads x 2 words       [handle, rotation (i32 bits)]
//   outputs: noutputs words         operands
// The permutation (Params::perm): perm_nsets accumulator handles, then
// perm_ncolumns column handles, then perm_ncolumns sigma handles.
// A lookup (LOOKUP_WORDS words): [aggregator, multiplicities, nchunks, chunks],
//   chunks: nchunks x CHUNK_WORDS words [helper handle, program offset].
// A trash argument (TRASH_WORDS words): [trash handle, selector handle,
//   program offset, 0].

#ifndef MIDNIGHT_VULKAN_QUOTIENT_GLSL
#define MIDNIGHT_VULKAN_QUOTIENT_GLSL

#include "fr.glsl"

const uint HANDLE_SHIFT = 30u;
const uint HANDLE_MASK = (1u << 30) - 1u;
const uint BANK_PROOF = 0u;
const uint BANK_KEY_RES = 1u;
const uint BANK_KEY_TMP = 2u;

const uint OPERAND_SHIFT = 30u;
const uint OPERAND_MASK = (1u << 30) - 1u;
const uint TAG_UNIFORM = 0u;
const uint TAG_READ = 1u;

const uint OP_ADD = 0u;
const uint OP_SUB = 1u;
const uint OP_MUL = 2u;
const uint OP_SQUARE = 3u;
const uint OP_DOUBLE = 4u;
const uint OP_NEGATE = 5u;
const uint OP_WORDS = 5u;

const uint PROG_OPS = 0u;
const uint PROG_NOPS = 1u;
const uint PROG_READS = 2u;
const uint PROG_UNIFORMS = 4u;
const uint PROG_OUTPUTS = 5u;
const uint PROG_NOUTPUTS = 6u;

const uint LOOKUP_WORDS = 4u;
const uint LOOKUP_AGGREGATOR = 0u;
const uint LOOKUP_MULTIPLICITIES = 1u;
const uint LOOKUP_NCHUNKS = 2u;
const uint LOOKUP_CHUNKS = 3u;
const uint CHUNK_WORDS = 2u;
const uint CHUNK_HELPER = 0u;
const uint CHUNK_PROGRAM = 1u;

const uint TRASH_WORDS = 4u;
const uint TRASH_POLY = 0u;
const uint TRASH_SELECTOR = 1u;
const uint TRASH_PROGRAM = 2u;

// One dispatch of the numerator (the host's `quotient::Params`, same layout:
// 24 words, then seven field elements; 320 bytes).
struct Params {
    uint ext_k;
    uint log_scale;          // ext_k - k
    uint mask;               // 2^ext_k - 1
    uint row0;               // first row of this dispatch
    uint row_end;            // one past its last row
    uint threads;            // invocations of the dispatch (the scratch stride)
    uint gates;              // program offset
    uint perm_nsets;         // 0: no permutation argument
    uint perm_ncolumns;
    uint perm_chunk_len;
    int perm_last_rotation;
    uint perm;               // word offset of the permutation handles
    uint nlookups;
    uint lookups;            // word offset
    uint ntrash;
    uint trash;              // word offset
    uint l0;                 // handles
    uint l_last;
    uint l_active;
    uint nwords;             // words of the tables
    uint pad[4];
    Fr y;
    Fr beta;
    Fr gamma;
    Fr delta_start;
    Fr delta;
    Fr omega;                // the extended root
    Fr omega_step;           // omega^threads
};

#endif // MIDNIGHT_VULKAN_QUOTIENT_GLSL
