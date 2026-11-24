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

//! `quotient_numerator` on a small hand-built job against a CPU evaluation
//! of the same tables, in both variants.
//!
//! Scaffolding until the shared backend drives the kernel: the shared
//! backend's tests against `QuotientJob::numerator_reference` on real
//! circuits will supersede this one. The table layout is `quotient.glsl`'s.

mod common;

use common::{scalar, scalars};
use ff::Field;
use midnight_curves::Fq;
use midnight_vulkan::{
    Params, Variant,
    test_support::{context, run},
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

const EXT_K: u32 = 6;
const LOG_SCALE: u32 = 2;
const N: usize = 1 << EXT_K;

const TAG_UNIFORM: u32 = 0;
const TAG_READ: u32 = 1;
const TAG_SCRATCH: u32 = 2;

const OP_ADD: u32 = 0;
const OP_SUB: u32 = 1;
const OP_MUL: u32 = 2;
const OP_SQUARE: u32 = 3;
const OP_DOUBLE: u32 = 4;
const OP_NEGATE: u32 = 5;
const OP_MUL_ADD: u32 = 6;

fn handle(bank: u32, index: u32) -> u32 {
    (bank << 30) | index
}

fn uni(i: u32) -> u32 {
    (TAG_UNIFORM << 30) | i
}

fn read(i: u32) -> u32 {
    (TAG_READ << 30) | i
}

fn slot(i: u32) -> u32 {
    (TAG_SCRATCH << 30) | i
}

/// A program: ops `[kind, dst, a, b, c]`, reads `(handle, rotation)`, the
/// first uniform's index, the output operands.
struct Program {
    ops: Vec<[u32; 5]>,
    reads: Vec<(u32, i32)>,
    uniforms: u32,
    outputs: Vec<u32>,
}

/// The job's tables, as `quotient.glsl` lays them out.
#[derive(Default)]
struct Words(Vec<u32>);

impl Words {
    /// Appends a program; its word offset.
    fn program(&mut self, p: &Program) -> u32 {
        let at = self.0.len() as u32;
        let ops = at + 8;
        let reads = ops + 5 * p.ops.len() as u32;
        let outs = reads + 2 * p.reads.len() as u32;
        self.0.extend([
            ops,
            p.ops.len() as u32,
            reads,
            p.reads.len() as u32,
            p.uniforms,
            outs,
            p.outputs.len() as u32,
            0,
        ]);
        self.0.extend(p.ops.iter().flatten());
        self.0.extend(p.reads.iter().flat_map(|(h, r)| [*h, *r as u32]));
        self.0.extend(&p.outputs);
        at
    }

    fn push(&mut self, w: &[u32]) -> u32 {
        let at = self.0.len() as u32;
        self.0.extend_from_slice(w);
        at
    }
}

/// The three banks of polynomials, each `N` values per polynomial.
struct Banks([Vec<Vec<Fq>>; 3]);

impl Banks {
    fn get(&self, h: u32, row: u32) -> Fq {
        self.0[(h >> 30) as usize][(h & ((1 << 30) - 1)) as usize][row as usize]
    }
}

fn rotate(idx: u32, rot: i32) -> u32 {
    idx.wrapping_add((rot as u32) << LOG_SCALE) & (N as u32 - 1)
}

/// The CPU evaluation of a program at `row`: its outputs.
fn eval(p: &Program, banks: &Banks, uniforms: &[Fq], row: u32) -> Vec<Fq> {
    let mut scratch = vec![Fq::ZERO; 16];
    let operand = |o: u32, scratch: &[Fq]| {
        let i = o & ((1 << 30) - 1);
        match o >> 30 {
            TAG_UNIFORM => uniforms[(p.uniforms + i) as usize],
            TAG_READ => {
                let (h, rot) = p.reads[i as usize];
                banks.get(h, rotate(row, rot))
            }
            _ => scratch[i as usize],
        }
    };
    for op in &p.ops {
        let a = operand(op[2], &scratch);
        let v = match op[0] {
            OP_ADD => a + operand(op[3], &scratch),
            OP_SUB => a - operand(op[3], &scratch),
            OP_MUL => a * operand(op[3], &scratch),
            OP_SQUARE => a.square(),
            OP_DOUBLE => a.double(),
            OP_NEGATE => -a,
            _ => a * operand(op[3], &scratch) + operand(op[4], &scratch),
        };
        scratch[op[1] as usize] = v;
    }
    p.outputs.iter().take(4).map(|o| operand(*o, &scratch)).collect()
}

#[test]
fn quotient_numerator_matches_cpu() {
    let c = context();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut polys = |count: usize| -> Vec<Vec<Fq>> {
        (0..count).map(|_| (0..N).map(|_| Fq::random(&mut rng)).collect()).collect()
    };
    let banks = Banks([polys(6), polys(3), polys(3)]);
    let uniforms: Vec<Fq> = (0..8).map(|i| Fq::from(3 + i as u64)).collect();
    let (p0, p1, p2, p3, p4, p5) = (
        handle(0, 0),
        handle(0, 1),
        handle(0, 2),
        handle(0, 3),
        handle(0, 4),
        handle(0, 5),
    );
    let (k0, k1, k2) = (handle(1, 0), handle(1, 1), handle(1, 2));
    let (t0, t1, t2) = (handle(2, 0), handle(2, 1), handle(2, 2));

    // every op kind, reads at several rotations, uniforms
    let gates = Program {
        ops: vec![
            [OP_ADD, 0, read(0), read(1), 0],
            [OP_SUB, 1, slot(0), uni(0), 0],
            [OP_MUL, 2, slot(1), read(2), 0],
            [OP_SQUARE, 3, slot(2), 0, 0],
            [OP_DOUBLE, 4, slot(3), 0, 0],
            [OP_NEGATE, 5, slot(4), 0, 0],
            [OP_MUL_ADD, 6, slot(5), read(3), uni(1)],
        ],
        reads: vec![(p0, 0), (p1, 1), (k0, -1), (t1, 2)],
        uniforms: 0,
        outputs: vec![slot(6)],
    };
    let chunk0 = Program {
        ops: vec![
            [OP_MUL, 0, read(0), uni(0), 0],
            [OP_ADD, 1, slot(0), read(1), 0],
        ],
        reads: vec![(p2, 0), (k1, 3)],
        uniforms: 2,
        outputs: vec![slot(1), read(0), uni(1), slot(0)],
    };
    let chunk1 = Program {
        ops: vec![[OP_SUB, 0, read(0), uni(0), 0]],
        reads: vec![(t2, -2)],
        uniforms: 4,
        outputs: vec![slot(0), read(0)],
    };
    let trash = Program {
        ops: vec![[OP_MUL_ADD, 0, read(0), read(1), uni(0)]],
        reads: vec![(p5, 0), (k2, 1)],
        uniforms: 6,
        outputs: vec![slot(0)],
    };

    let mut w = Words::default();
    let gates_at = w.program(&gates);
    let chunk0_at = w.program(&chunk0);
    let chunk1_at = w.program(&chunk1);
    let trash_at = w.program(&trash);
    // permutation: two sets, three columns, chunks of two
    let (zs, cols, sigma) = ([p2, p3], [p0, k1, t0], [k2, t2, p4]);
    let perm_at = w.push(&[zs.as_slice(), &cols, &sigma].concat());
    let (agg, mult, helpers) = (p4, t1, [p1, k2]);
    let chunks_at = w.push(&[helpers[0], chunk0_at, helpers[1], chunk1_at]);
    let lookups_at = w.push(&[agg, mult, 2, chunks_at]);
    let (trash_poly, trash_sel) = (p3, t0);
    let trash_tbl = w.push(&[trash_poly, trash_sel, trash_at, 0]);
    let (l0, l_last, l_active) = (p5, k0, t2);
    let (chunk_len, last_rot) = (2u32, -3i32);

    let r: Vec<Fq> = (0..5).map(|_| Fq::random(&mut rng)).collect();
    let (y, beta, gamma, delta_start, delta) = (r[0], r[1], r[2], r[3], r[4]);
    let omega = Fq::from(7);
    let threads = 16u32;
    let omega_step = omega.pow_vartime([threads as u64]);

    // the CPU evaluation
    let want: Vec<Fq> = (0..N as u32)
        .map(|idx| {
            let col = |h: u32, row: u32| banks.get(h, row);
            let mut value = eval(&gates, &banks, &uniforms, idx)[0];
            let (l0v, l_lastv, l_activev) = (col(l0, idx), col(l_last, idx), col(l_active, idx));
            let r_next = rotate(idx, 1);
            let r_last = rotate(idx, last_rot);
            let z_first = col(zs[0], idx);
            let z_last = col(zs[1], idx);
            value = value * y + (Fq::ONE - z_first) * l0v;
            value = value * y + (z_last.square() - z_last) * l_lastv;
            value = value * y + (col(zs[1], idx) - col(zs[0], r_last)) * l0v;
            let mut current_delta = delta_start * omega.pow_vartime([idx as u64]);
            for (s, z) in zs.iter().enumerate() {
                let ks = s * chunk_len as usize..((s + 1) * chunk_len as usize).min(cols.len());
                let mut left = col(*z, r_next);
                for k in ks.clone() {
                    left *= col(cols[k], idx) + beta * col(sigma[k], idx) + gamma;
                }
                let mut right = col(*z, idx);
                for k in ks {
                    right *= col(cols[k], idx) + current_delta + gamma;
                    current_delta *= delta;
                }
                value = value * y + (left - right) * l_activev;
            }
            let agg_i = col(agg, idx);
            value = value * y + agg_i * (l0v + l_lastv);
            let (mut sum_helpers, mut table, mut selector) = (Fq::ZERO, Fq::ZERO, Fq::ZERO);
            for (fi, (h, prog)) in helpers.iter().zip([&chunk0, &chunk1]).enumerate() {
                let o = eval(prog, &banks, &uniforms, idx);
                if fi == 0 {
                    table = o[2];
                    selector = o[3];
                }
                let hv = col(*h, idx);
                value = value * y + hv * o[1] - o[0];
                sum_helpers += hv;
            }
            let inner =
                ((col(agg, r_next) - agg_i) - selector * sum_helpers) * table + col(mult, idx);
            value = value * y + l_activev * inner;
            let o = eval(&trash, &banks, &uniforms, idx);
            value * y + (o[0] - (Fq::ONE - col(trash_sel, idx)) * col(trash_poly, idx))
        })
        .collect();

    let bank = |b: usize| c.upload(&scalars(&banks.0[b].concat())).unwrap();
    let (proof, key_res, key_tmp) = (bank(0), bank(1), bank(2));
    let words = c.upload(&w.0).unwrap();
    let unis = c.upload(&scalars(&uniforms)).unwrap();
    let scratch = c.alloc(16 * threads as usize * 32).unwrap();
    for variant in Variant::ALL {
        let nu = c.upload(&vec![[0u64; 4]; N]).unwrap();
        // two dispatches over [0, 40) and [40, N), several rows per invocation
        for (row0, row_end) in [(0u32, 40u32), (40, N as u32)] {
            let mut p: Vec<u32> = vec![
                EXT_K,
                LOG_SCALE,
                N as u32 - 1,
                row0,
                row_end,
                threads,
                gates_at,
                zs.len() as u32,
                cols.len() as u32,
                chunk_len,
                last_rot as u32,
                perm_at,
                1,
                lookups_at,
                1,
                trash_tbl,
                l0,
                l_last,
                l_active,
            ];
            p.resize(24, 0);
            for v in [y, beta, gamma, delta_start, delta, omega, omega_step] {
                p.extend(scalar(&v).iter().flat_map(|l| [*l as u32, (*l >> 32) as u32]));
            }
            assert_eq!(p.len() * 4, 320);
            let params = c.upload(&p).unwrap();
            let push = Params::new()
                .addr(&proof)
                .addr(&key_res)
                .addr(&key_tmp)
                .addr(&words)
                .addr(&unis)
                .addr(&nu)
                .addr(&scratch)
                .addr(&params);
            // more invocations than `threads`: the extra ones return
            run("quotient_numerator", variant, &push, threads as u64 + 5).unwrap();
        }
        let got: Vec<[u64; 4]> = c.read(&nu).unwrap();
        assert_eq!(got, scalars(&want), "{variant:?}");
    }
}
