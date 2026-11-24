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

//! The MSM kernels stage by stage against CPU models, in both variants: the
//! digits reconstruct the scalars, the sort groups every digit under its
//! bucket, the bucket and `msm_ones` sums are the sums of their points,
//! `msm_integrate` is sppark's running sum, and the window sums give the MSM.
//!
//! Scaffolding until the shared backend drives these kernels (its MSM tests
//! against `msm_pippenger` will supersede these): the plan, buffer sizes and
//! dispatch shapes below copy the Metal host's.

mod common;

use common::{affine, cuda_mask_vector, fq, scalars, xyzz_point};
use ff::Field;
use group::{Curve, Group};
use midnight_curves::{Fq, G1Affine, G1Projective};
use midnight_vulkan::{
    Buffer, Commands, Grid, Params, Pipeline, PipelineOptions, Variant, test_support::context,
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

const NBITS: u32 = 255;
const SORT_GROUPS: u32 = 8;
const SORT_BLOCK: u32 = 1024;
const ONES_GROUPS: u64 = 32;
const ONES_BLOCK: u32 = 256;
const ACC_GROUP: u32 = 256;
const NTHR: u32 = 256;
const MODE_PLAIN: u32 = 0;
const MODE_MASKED: u32 = 1;
const MODE_MASK: u32 = 2;

/// A window plan small enough to check every bucket.
#[derive(Clone, Copy, Debug)]
struct Plan {
    wbits: u32,
    nwins: u32,
    stride: u32,
    bits: u32,
    top_bits: u32,
    low_bits: u32,
    top: u32,
    pass: u32,
}

fn plan(n: u32, wbits: u32, pass: u32) -> Plan {
    let nwins = (NBITS - 1) / wbits + 1;
    let bits = wbits - 1;
    Plan {
        wbits,
        nwins,
        stride: n.next_multiple_of(32),
        bits,
        top_bits: bits / 2,
        low_bits: bits - bits / 2,
        top: NBITS - wbits * (nwins - 1),
        pass,
    }
}

/// The pipeline of an MSM kernel: subgroups of 32, the Metal host's
/// workgroup size.
fn pipe(name: &str, variant: Variant, workgroup: u32) -> std::sync::Arc<Pipeline> {
    let options = PipelineOptions {
        subgroup: Some(32),
        ..PipelineOptions::workgroup(workgroup)
    };
    context().pipeline(name, variant, options).unwrap()
}

/// The device buffers of one chunk.
struct Bufs {
    points: Buffer,
    scalars: Buffer,
    digits: Buffer,
    temps: Buffer,
    counts: Buffer,
    hist: Buffer,
    buckets: Buffer,
    ones: Buffer,
    hot: Buffer,
}

impl Bufs {
    fn new(pl: &Plan, points: &[[u64; 12]], scalars: &[[u64; 4]]) -> Bufs {
        let c = context();
        let rows = (pl.nwins as usize) << pl.bits;
        let ntop = 1usize << pl.top_bits;
        let stride = pl.stride as usize;
        Bufs {
            points: c.upload(points).unwrap(),
            scalars: c.upload(scalars).unwrap(),
            digits: c.alloc(pl.nwins as usize * stride * 4).unwrap(),
            temps: c.alloc(pl.pass as usize * stride * 8).unwrap(),
            counts: c.alloc(pl.pass as usize * ntop * SORT_GROUPS as usize * 4).unwrap(),
            hist: c.alloc(rows * 4).unwrap(),
            buckets: c.alloc(rows * 192).unwrap(),
            ones: c.alloc((ONES_GROUPS as usize * ONES_BLOCK as usize / 32) * 192).unwrap(),
            hot: c.alloc(4).unwrap(),
        }
    }
}

fn digits<'a>(
    cmds: &mut Commands<'a>,
    p: &'a Pipeline,
    b: &'a Bufs,
    pl: &Plan,
    n: u32,
    offset: u32,
    mode: u32,
) {
    let params = Params::new()
        .addr(&b.scalars)
        .addr(&b.digits)
        .addr(&b.hot)
        .words(&[n, offset, pl.nwins, pl.wbits, pl.stride, mode]);
    cmds.dispatch(p, &params, Grid::Threads(n as u64)).unwrap();
}

/// The four sort kernels of every pass.
fn sort<'a>(
    cmds: &mut Commands<'a>,
    p: &'a [std::sync::Arc<Pipeline>; 4],
    b: &'a Bufs,
    pl: &Plan,
    n: u32,
) {
    let ntop = 1u64 << pl.top_bits;
    for w0 in (0..pl.nwins).step_by(pl.pass as usize) {
        let sp = [
            n,
            pl.stride,
            w0,
            pl.nwins,
            SORT_GROUPS,
            pl.top_bits,
            pl.low_bits,
            pl.bits,
            pl.top - 1,
        ];
        let pw = (pl.nwins - w0).min(pl.pass) as u64;
        let groups = pw * SORT_GROUPS as u64;
        let two = |a: &'a Buffer, c: &'a Buffer| Params::new().addr(a).addr(c).words(&sp);
        cmds.dispatch(&p[0], &two(&b.digits, &b.counts), Grid::Groups(groups)).unwrap();
        cmds.dispatch(&p[1], &two(&b.counts, &b.hist), Grid::Groups(pw)).unwrap();
        let scatter = Params::new().addr(&b.digits).addr(&b.counts).addr(&b.temps).words(&sp);
        cmds.dispatch(&p[2], &scatter, Grid::Groups(groups)).unwrap();
        let lower = Params::new().addr(&b.digits).addr(&b.temps).addr(&b.hist).words(&sp);
        cmds.dispatch(&p[3], &lower, Grid::Groups(pw * ntop)).unwrap();
    }
}

/// The magnitude of a `msm_digits` entry: its low `wbits` bits (sppark
/// stores a negative Booth digit as `0 - m`, whose low bits are `m`).
fn magnitude(pl: &Plan, d: u32) -> u32 {
    d & ((1 << pl.wbits) - 1)
}

/// The signed digit of a `msm_digits` entry (bit 31 the sign), as a scalar.
fn digit_value(pl: &Plan, d: u32) -> Fq {
    let mag = Fq::from(magnitude(pl, d) as u64);
    if d >> 31 != 0 { -mag } else { mag }
}

/// `2^e` in `Fq`.
fn pow2(e: u32) -> Fq {
    Fq::from(2).pow_vartime([e as u64])
}

/// The bucket key of an entry of window `win`: `(magnitude - 1)` above the
/// low `lshift` bits of the point index (the top window's sub-buckets).
fn bucket_of(pl: &Plan, win: u32, mag: u32, idx: u32) -> u32 {
    let lshift = if win == pl.nwins - 1 {
        pl.bits - (pl.top - 1)
    } else {
        0
    };
    ((mag - 1) << lshift) | (idx & ((1u32 << lshift) - 1))
}

/// Points with the identity and repeats, scalars with edge values.
fn inputs(n: usize, seed: u64) -> (Vec<G1Affine>, Vec<Fq>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut pts: Vec<G1Affine> =
        (0..n).map(|_| G1Projective::random(&mut rng).to_affine()).collect();
    for i in (0..n).step_by(97) {
        pts[i] = G1Affine::default();
    }
    for i in (5..n).step_by(31) {
        pts[i] = pts[i - 5];
    }
    let mut s: Vec<Fq> = (0..n).map(|_| Fq::random(&mut rng)).collect();
    for (i, v) in [Fq::ZERO, Fq::ONE, -Fq::ONE, Fq::from(2).invert().unwrap()]
        .into_iter()
        .enumerate()
    {
        s[i * 7] = v;
    }
    (pts, s)
}

/// Every stage of one chunk against its CPU model.
#[test]
fn msm_stages_match_cpu_models() {
    let c = context();
    let n = 3000u32;
    let offset = 1234u32;
    // three sort passes, the last one shorter
    let pl = plan(n, 11, 10);
    let (pts, s) = inputs(n as usize, 1);
    let limbs: Vec<[u64; 12]> = pts.iter().map(affine).collect();
    let mask = cuda_mask_vector(offset as u64, n as usize);
    let rows = pl.nwins << pl.bits;
    let row_len = 1usize << pl.bits;
    for variant in Variant::ALL {
        let b = Bufs::new(&pl, &limbs, &scalars(&s));
        let p_digits = pipe("msm_digits", variant, 256);
        let p_sort = [
            pipe("msm_sort_count", variant, SORT_BLOCK),
            pipe("msm_sort_scan", variant, SORT_BLOCK),
            pipe("msm_sort_scatter", variant, SORT_BLOCK),
            pipe("msm_sort_lower", variant, SORT_BLOCK),
        ];
        let p_guard = pipe("msm_bucket_guard", variant, 256);
        let p_ones = pipe("msm_ones", variant, ONES_BLOCK);
        let p_acc = pipe("msm_accumulate", variant, ACC_GROUP);
        let p_int = pipe("msm_integrate", variant, NTHR);

        // digits: every mode reconstructs its scalar
        let mut digits_masked = Vec::new();
        for mode in [MODE_PLAIN, MODE_MASK, MODE_MASKED] {
            let mut cmds = c.commands().unwrap();
            digits(&mut cmds, &p_digits, &b, &pl, n, offset, mode);
            cmds.submit().unwrap();
            let d: Vec<u32> = c.read(&b.digits).unwrap();
            for i in 0..n as usize {
                let want = match mode {
                    MODE_PLAIN => s[i],
                    MODE_MASK => fq(&mask[i]),
                    _ => s[i] + fq(&mask[i]),
                };
                let mut got = Fq::ZERO;
                for w in 0..pl.nwins {
                    let e = d[w as usize * pl.stride as usize + i];
                    assert!(
                        magnitude(&pl, e) <= 1 << pl.bits,
                        "{variant:?} mode {mode}: digit {e:x}"
                    );
                    got += digit_value(&pl, e) * pow2(w * pl.wbits);
                }
                assert_eq!(got, want, "{variant:?} mode {mode}: digits of scalar {i}");
            }
            if mode == MODE_MASKED {
                digits_masked = d;
            }
        }
        let d = digits_masked;

        // the sort: each row holds every nonzero digit's (index, sign) under
        // its bucket, the hist row the inclusive bucket ends
        let mut cmds = c.commands().unwrap();
        sort(&mut cmds, &p_sort, &b, &pl, n);
        cmds.submit().unwrap();
        let sorted: Vec<u32> = c.read(&b.digits).unwrap();
        let hist: Vec<u32> = c.read(&b.hist).unwrap();
        // want[w][bucket]: the entries (index | sign), sorted for comparison
        let mut want = vec![vec![Vec::new(); row_len]; pl.nwins as usize];
        for w in 0..pl.nwins {
            for i in 0..n {
                let e = d[(w * pl.stride + i) as usize];
                if e != 0 {
                    let k = bucket_of(&pl, w, magnitude(&pl, e), i);
                    want[w as usize][k as usize].push(i | (e & 0x8000_0000));
                }
            }
        }
        for w in 0..pl.nwins as usize {
            let h = &hist[w * row_len..(w + 1) * row_len];
            let row = &sorted[w * pl.stride as usize..];
            let mut start = 0usize;
            for (k, entries) in want[w].iter_mut().enumerate() {
                let end = h[k] as usize;
                assert_eq!(
                    end - start,
                    entries.len(),
                    "{variant:?} window {w} bucket {k}"
                );
                let mut got = row[start..end].to_vec();
                got.sort_unstable();
                entries.sort_unstable();
                assert_eq!(&got, entries, "{variant:?} window {w} bucket {k}");
                start = end;
            }
        }

        // the bucket sums and the digit-1 bucket of window 0
        let acc_params = [pl.nwins, pl.bits, pl.stride, 1 << 16];
        let mut cmds = c.commands().unwrap();
        let guard = Params::new().addr(&b.hist).addr(&b.hot).words(&acc_params);
        cmds.dispatch(&p_guard, &guard, Grid::Threads(rows as u64)).unwrap();
        let ones = Params::new().addr(&b.ones).addr(&b.points).addr(&b.digits).addr(&b.hist);
        cmds.dispatch(&p_ones, &ones, Grid::Groups(ONES_GROUPS)).unwrap();
        let acc = Params::new()
            .addr(&b.buckets)
            .addr(&b.points)
            .addr(&b.digits)
            .addr(&b.hist)
            .addr(&b.hot)
            .words(&acc_params);
        cmds.dispatch(&p_acc, &acc, Grid::Threads(rows as u64)).unwrap();
        cmds.submit().unwrap();
        assert_eq!(c.read::<u32>(&b.hot).unwrap()[0], 0, "{variant:?} hot");
        let point = |e: u32| {
            let p = G1Projective::from(pts[(e & 0x7fff_ffff) as usize]);
            if e >> 31 != 0 { -p } else { p }
        };
        let sum = |es: &[u32]| es.iter().map(|e| point(*e)).sum::<G1Projective>();
        let buckets: Vec<[u64; 24]> = c.read(&b.buckets).unwrap();
        let mut bucket_points = vec![G1Projective::identity(); rows as usize];
        for w in 0..pl.nwins as usize {
            for (k, entries) in want[w].iter().enumerate() {
                let got = xyzz_point(&buckets[w * row_len + k]);
                let expect = if w == 0 && k == 0 {
                    G1Projective::identity()
                } else {
                    sum(entries)
                };
                assert!(got == expect, "{variant:?} bucket ({w}, {k})");
                bucket_points[w * row_len + k] = got;
            }
        }
        let ones: Vec<[u64; 24]> = c.read(&b.ones).unwrap();
        let ones_sum: G1Projective = ones.iter().map(xyzz_point).sum();
        assert!(ones_sum == sum(&want[0][0]), "{variant:?} msm_ones");

        // the window sums of the buckets (sub-buckets of the top window
        // weighted by their digit) give the MSM of the masked scalars
        let mut total = G1Projective::identity();
        for w in (0..pl.nwins as usize).rev() {
            let lshift = if w as u32 == pl.nwins - 1 {
                pl.bits - (pl.top - 1)
            } else {
                0
            };
            let mut run = G1Projective::identity();
            let mut win = G1Projective::identity();
            let mut weight_prev = ((row_len as u32 - 1) >> lshift) + 1;
            for k in (0..row_len).rev() {
                let weight = (k as u32 >> lshift) + 1;
                // add the running sum once per step down in weight
                while weight_prev > weight {
                    win += run;
                    weight_prev -= 1;
                }
                run += bucket_points[w * row_len + k];
            }
            while weight_prev > 0 {
                win += run;
                weight_prev -= 1;
            }
            for _ in 0..pl.wbits {
                total = total.double();
            }
            total += win;
        }
        total += ones_sum;
        let masked: Vec<Fq> = (0..n as usize).map(|i| s[i] + fq(&mask[i])).collect();
        let msm: G1Projective =
            masked.iter().zip(&pts).map(|(s, p)| G1Projective::from(*p) * s).sum();
        assert!(total == msm, "{variant:?} window sums");

        // msm_integrate: sppark's running sums over each thread's run
        let mut cmds = c.commands().unwrap();
        let int = Params::new().addr(&b.buckets).words(&[pl.nwins, pl.wbits, NBITS]);
        cmds.dispatch(&p_int, &int, Grid::Groups(pl.nwins as u64)).unwrap();
        cmds.submit().unwrap();
        let out: Vec<[u64; 24]> = c.read(&b.buckets).unwrap();
        for w in 0..pl.nwins {
            let row = &bucket_points[(w as usize) * row_len..][..row_len];
            for (t, (q, acc)) in integrate(&pl, w, row).into_iter().enumerate() {
                let base = w as usize * row_len + 2 * t;
                assert!(
                    xyzz_point(&out[base]) == q,
                    "{variant:?} integrate ({w}, {t}) q"
                );
                assert!(
                    xyzz_point(&out[base + 1]) == acc,
                    "{variant:?} integrate ({w}, {t}) acc"
                );
            }
        }
    }
}

/// The CPU model of `msm_integrate` for window `w`: each thread's `(q, acc)`.
fn integrate(pl: &Plan, w: u32, row: &[G1Projective]) -> Vec<(G1Projective, G1Projective)> {
    let per = 1usize << (pl.bits - NTHR.trailing_zeros());
    let mask = if (w + 1) * pl.wbits > NBITS {
        let lsbits = NBITS - w * pl.wbits;
        (1usize << (pl.wbits - lsbits)) - 1
    } else {
        0
    };
    (0..NTHR as usize)
        .map(|tid| {
            let run = &row[tid * per..(tid + 1) * per];
            let mut i = per - 1;
            let mut acc = run[i];
            let mut res = if i & mask != 0 {
                G1Projective::identity()
            } else {
                acc
            };
            let mut q = res;
            while i > 0 {
                i -= 1;
                q = run[i];
                let mut pc = if i & mask != 0 { 2 } else { 0 };
                loop {
                    q += acc;
                    if pc == 1 {
                        res = q;
                    } else {
                        acc = q;
                        if pc == 0 {
                            q = res;
                        }
                    }
                    pc += 1;
                    if pc >= 2 {
                        break;
                    }
                }
            }
            (q, acc)
        })
        .collect()
}

/// A bucket above the hot limit makes `msm_accumulate` write infinity to
/// every bucket and leaves `hot` set.
#[test]
fn hot_bucket_guard_skips_accumulation() {
    let c = context();
    let n = 512u32;
    let pl = plan(n, 11, 24);
    let (pts, _) = inputs(n as usize, 2);
    // every scalar equal: each window's digits share one bucket
    let s = vec![Fq::from(0x1234_5678u64); n as usize];
    let limbs: Vec<[u64; 12]> = pts.iter().map(affine).collect();
    let rows = pl.nwins << pl.bits;
    for variant in Variant::ALL {
        let b = Bufs::new(&pl, &limbs, &scalars(&s));
        let p_digits = pipe("msm_digits", variant, 256);
        let p_sort = [
            pipe("msm_sort_count", variant, SORT_BLOCK),
            pipe("msm_sort_scan", variant, SORT_BLOCK),
            pipe("msm_sort_scatter", variant, SORT_BLOCK),
            pipe("msm_sort_lower", variant, SORT_BLOCK),
        ];
        let p_guard = pipe("msm_bucket_guard", variant, 256);
        let p_acc = pipe("msm_accumulate", variant, ACC_GROUP);
        c.write(&b.buckets, &vec![0xffu8; rows as usize * 192]).unwrap();
        for limit in [n, n - 1] {
            let acc_params = [pl.nwins, pl.bits, pl.stride, limit];
            let mut cmds = c.commands().unwrap();
            digits(&mut cmds, &p_digits, &b, &pl, n, 0, MODE_PLAIN);
            sort(&mut cmds, &p_sort, &b, &pl, n);
            let guard = Params::new().addr(&b.hist).addr(&b.hot).words(&acc_params);
            cmds.dispatch(&p_guard, &guard, Grid::Threads(rows as u64)).unwrap();
            let acc = Params::new()
                .addr(&b.buckets)
                .addr(&b.points)
                .addr(&b.digits)
                .addr(&b.hist)
                .addr(&b.hot)
                .words(&acc_params);
            cmds.dispatch(&p_acc, &acc, Grid::Threads(rows as u64)).unwrap();
            cmds.submit().unwrap();
            let hot = c.read::<u32>(&b.hot).unwrap()[0];
            let buckets: Vec<[u64; 24]> = c.read(&b.buckets).unwrap();
            let all_inf = buckets.iter().all(|r| *r == [0u64; 24]);
            assert_eq!(
                (hot, all_inf),
                ((limit < n) as u32, limit < n),
                "{variant:?} limit {limit}"
            );
        }
    }
}
