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

//! The NTT kernels against `midnight_curves::fft`, in both variants.
//!
//! Scaffolding until the shared backend drives these kernels: the pass plan,
//! twiddle tables and pass parameters below are a copy of the Metal crate's
//! host code (`midnight-accel` `ntt`), and these tests will be superseded by
//! the shared backend's transform tests.

mod common;

use common::{scalar, scalars};
use ff::{Field, PrimeField};
use midnight_curves::{Fq, fft::best_fft};
use midnight_vulkan::{
    Buffer, Grid, Params, PipelineOptions, Variant,
    test_support::{context, run},
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

const UNIVERSAL_LOG: u32 = 28;
const HI_BITS: u32 = 16;
const MAX_PASS_LOG: u32 = 10;
const GROUP_LOG: u32 = 10;
const F_TWIDDLE: u32 = 1;
const F_TWO_LEVEL: u32 = 2;
const F_COSET: u32 = 4;

/// The standard `2^m`-th root of unity, or its inverse.
fn root(m: u32, inverse: bool) -> Fq {
    let mut w = Fq::ROOT_OF_UNITY;
    for _ in m..Fq::S {
        w = w.square();
    }
    if inverse { w.invert().unwrap() } else { w }
}

fn powers(w: Fq, len: usize) -> Vec<Fq> {
    std::iter::successors(Some(Fq::ONE), |x| Some(x * w)).take(len).collect()
}

fn plan(log_n: u32) -> Vec<u32> {
    let passes = log_n.div_ceil(MAX_PASS_LOG).max(1);
    let (base, extra) = (log_n / passes, log_n % passes);
    (0..passes).map(|i| base + u32::from(i < extra)).collect()
}

fn group_columns_log(log_n: u32, r: u32) -> u32 {
    GROUP_LOG.saturating_sub(r).min(log_n - r)
}

/// The words of a `PassParams`.
fn pass_params(
    log_n: u32,
    r: u32,
    log_p: u32,
    flags: u32,
    coset: Option<(u32, [Fq; 2])>,
) -> Vec<u32> {
    let (n_src, z) = coset.unwrap_or((0, [Fq::ZERO; 2]));
    let mut w = vec![
        log_n,
        r,
        log_p,
        group_columns_log(log_n, r),
        flags,
        n_src,
        HI_BITS,
        0,
    ];
    for z in z {
        w.extend(scalar(&z).iter().flat_map(|l| [*l as u32, (*l >> 32) as u32]));
    }
    w
}

/// A transform of `2^log_n` elements: forward or inverse, the divisor folded
/// into the last pass, the first pass reading a coset source.
struct Job<'b> {
    log_n: u32,
    inverse: bool,
    src: &'b Buffer,
    dst: &'b Buffer,
    tmp: &'b Buffer,
    scale: Option<Fq>,
    coset: Option<(u32, [Fq; 2])>,
}

/// Runs the passes of `job` in one command buffer, as the Metal host does.
fn transform(variant: Variant, job: &Job<'_>) {
    let c = context();
    let lb = UNIVERSAL_LOG - HI_BITS;
    let hi = c.upload(&scalars(&powers(root(HI_BITS, job.inverse), 1 << HI_BITS))).unwrap();
    let lo = c.upload(&scalars(&powers(root(UNIVERSAL_LOG, job.inverse), 1 << lb))).unwrap();
    let scaled = c.alloc(32 << lb).unwrap();
    let n = 1u64 << job.log_n;
    let radices = plan(job.log_n);
    let passes = radices.len();
    let mut pipes = Vec::new();
    for &r in &radices {
        let e_log = r + group_columns_log(job.log_n, r);
        let options = PipelineOptions {
            spec: [8 << e_log, 0, 0],
            ..PipelineOptions::workgroup(1 << e_log.saturating_sub(2))
        };
        pipes.push((c.pipeline("ntt_pass", variant, options).unwrap(), e_log));
    }
    let copy = c.pipeline("ntt_scale_copy", variant, PipelineOptions::default()).unwrap();

    let mut cmds = c.commands().unwrap();
    if let Some(s) = job.scale {
        let params = Params::new().addr(&lo).addr(&scaled).u64s(&scalar(&s)).u32(1 << lb);
        cmds.dispatch(&copy, &params, Grid::Threads(1 << lb)).unwrap();
    }
    let mut cur = job.src;
    let mut log_p = 0;
    for (q, &r) in radices.iter().enumerate() {
        let last = q + 1 == passes;
        let out = if last || q % 2 == 1 { job.dst } else { job.tmp };
        let scale_here = last && job.scale.is_some();
        let mut flags = 0;
        if log_p > 0 || scale_here {
            flags |= F_TWIDDLE;
        }
        if log_p + r > HI_BITS || scale_here {
            flags |= F_TWO_LEVEL;
        }
        let coset = if q == 0 { job.coset } else { None };
        if coset.is_some() {
            flags |= F_COSET;
        }
        let lo_tab = if scale_here { &scaled } else { &lo };
        let params = Params::new()
            .addr(cur)
            .addr(out)
            .addr(&hi)
            .addr(lo_tab)
            .words(&pass_params(job.log_n, r, log_p, flags, coset));
        let (pipe, e_log) = &pipes[q];
        cmds.dispatch(pipe, &params, Grid::Groups(n >> e_log)).unwrap();
        cur = out;
        log_p += r;
    }
    cmds.submit().unwrap();
}

fn random(n: usize, seed: u64) -> Vec<Fq> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n).map(|_| Fq::random(&mut rng)).collect()
}

fn read(b: &Buffer, n: usize) -> Vec<[u64; 4]> {
    let mut v: Vec<[u64; 4]> = context().read(b).unwrap();
    v.truncate(n);
    v
}

/// Forward and scaled inverse transforms over one-, two- and three-pass
/// plans, in place and through the scratch buffer, the two-level factors
/// included (from `2^17`).
#[test]
fn ntt_passes_match_cpu_fft() {
    let c = context();
    for variant in Variant::ALL {
        for log_n in [1u32, 5, 10, 11, 17, 21] {
            let n = 1usize << log_n;
            let a = random(n, log_n as u64);
            let data = c.upload(&scalars(&a)).unwrap();
            let tmp = c.alloc(32 * n).unwrap();

            let mut want = a.clone();
            best_fft(&mut want, root(log_n, false), log_n);
            let job = Job {
                log_n,
                inverse: false,
                src: &data,
                dst: &data,
                tmp: &tmp,
                scale: None,
                coset: None,
            };
            transform(variant, &job);
            assert_eq!(
                read(&data, n),
                scalars(&want),
                "{variant:?} forward 2^{log_n}"
            );

            // back, with the divisor 1/n folded into the last pass
            let n_inv = Fq::from(n as u64).invert().unwrap();
            transform(
                variant,
                &Job {
                    inverse: true,
                    scale: Some(n_inv),
                    ..job
                },
            );
            assert_eq!(read(&data, n), scalars(&a), "{variant:?} inverse 2^{log_n}");
        }
    }
}

/// `coeff_to_extended`'s first pass: the source distributed by
/// `[1, z1, z2][i mod 3]` and zero-padded from `2^k` to `2^ext_k`.
#[test]
fn coset_pass_matches_cpu() {
    let c = context();
    let z = random(2, 99);
    for variant in Variant::ALL {
        for (k, ext_k) in [(3u32, 5u32), (9, 11), (14, 16)] {
            let (n_src, n) = (1usize << k, 1usize << ext_k);
            let a = random(n_src, 7 + k as u64);
            let src = c.upload(&scalars(&a)).unwrap();
            let dst = c.alloc(32 * n).unwrap();
            let tmp = c.alloc(32 * n).unwrap();
            let mut want: Vec<Fq> =
                a.iter().enumerate().map(|(i, x)| [Fq::ONE, z[0], z[1]][i % 3] * x).collect();
            want.resize(n, Fq::ZERO);
            best_fft(&mut want, root(ext_k, false), ext_k);
            transform(
                variant,
                &Job {
                    log_n: ext_k,
                    inverse: false,
                    src: &src,
                    dst: &dst,
                    tmp: &tmp,
                    scale: None,
                    coset: Some((n_src as u32, [z[0], z[1]])),
                },
            );
            assert_eq!(
                read(&dst, n),
                scalars(&want),
                "{variant:?} coset 2^{k} -> 2^{ext_k}"
            );
        }
    }
}

/// The elementwise kernels on a length that is not a multiple of the
/// workgroup: every element is computed, none past the end is written.
#[test]
fn elementwise_kernels_match_cpu() {
    let c = context();
    let n = 1000usize;
    let a = random(n + 8, 1);
    let t = random(64, 2);
    let k = random(3, 3);
    let guard = |v: &[[u64; 4]], what: &str| {
        assert_eq!(&v[n..], &scalars(&a[n..]), "{what} wrote past n");
    };
    for variant in Variant::ALL {
        let buf = c.upload(&scalars(&a)).unwrap();
        let params = Params::new().addr(&buf).u64s(&scalar(&k[0])).u32(n as u32);
        run("ntt_scale", variant, &params, n as u64).unwrap();
        let got = read(&buf, n + 8);
        let want: Vec<Fq> = a[..n].iter().map(|x| x * k[0]).collect();
        assert_eq!(&got[..n], &scalars(&want), "{variant:?} ntt_scale");
        guard(&got, "ntt_scale");

        let src = c.upload(&scalars(&a)).unwrap();
        let dst = c.upload(&scalars(&a)).unwrap();
        let params = Params::new().addr(&src).addr(&dst).u64s(&scalar(&k[1])).u32(n as u32);
        run("ntt_scale_copy", variant, &params, n as u64).unwrap();
        let got = read(&dst, n + 8);
        let want: Vec<Fq> = a[..n].iter().map(|x| x * k[1]).collect();
        assert_eq!(&got[..n], &scalars(&want), "{variant:?} ntt_scale_copy");
        guard(&got, "ntt_scale_copy");

        let buf = c.upload(&scalars(&a)).unwrap();
        let params =
            Params::new().addr(&buf).u64s(&scalar(&k[1])).u64s(&scalar(&k[2])).u32(n as u32);
        run("ntt_distribute_zeta", variant, &params, n as u64).unwrap();
        let got = read(&buf, n + 8);
        let want: Vec<Fq> = a[..n]
            .iter()
            .enumerate()
            .map(|(i, x)| [Fq::ONE, k[1], k[2]][i % 3] * x)
            .collect();
        assert_eq!(
            &got[..n],
            &scalars(&want),
            "{variant:?} ntt_distribute_zeta"
        );
        guard(&got, "ntt_distribute_zeta");

        let buf = c.upload(&scalars(&a)).unwrap();
        let tb = c.upload(&scalars(&t)).unwrap();
        let params = Params::new().addr(&buf).addr(&tb).u32(63).u32(n as u32);
        run("ntt_mul_periodic", variant, &params, n as u64).unwrap();
        let got = read(&buf, n + 8);
        let want: Vec<Fq> = a[..n].iter().enumerate().map(|(i, x)| x * t[i & 63]).collect();
        assert_eq!(&got[..n], &scalars(&want), "{variant:?} ntt_mul_periodic");
        guard(&got, "ntt_mul_periodic");
    }
}
