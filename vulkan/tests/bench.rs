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

//! Throughput of the arithmetic variants: Fr and Fp multiplication and the
//! G1 mixed addition, as operations per second of GPU wall time (submission
//! to fence). Run on purpose:
//!
//! ```text
//! cargo test --release -p midnight-vulkan --test bench -- --ignored --nocapture
//! ```

mod common;

use std::time::Duration;

use ff::Field;
use group::{Curve, Group};
use midnight_curves::{Fp, Fq, G1Affine, G1Projective};
use midnight_vulkan::{PipelineOptions, Variant, test_support::context};
use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

const LOG_N: u32 = 22;
const REPS: usize = 3;

/// Random reduced field elements of `N` limbs: the top limb below `top`.
fn elements<const N: usize>(n: usize, top: u64, seed: u64) -> Vec<[u64; N]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let mut a: [u64; N] = std::array::from_fn(|_| rng.next_u64());
            a[N - 1] %= top;
            a
        })
        .collect()
}

/// Reinterprets limbs as the midnight-curves field `F` of the same size.
fn field<F: Copy, const N: usize>(a: &[u64; N]) -> F {
    assert_eq!(size_of::<F>(), size_of::<[u64; N]>());
    // SAFETY: `F` is a transparent blst value of `N` Montgomery limbs (size
    // checked above) and `a` is reduced.
    unsafe { std::mem::transmute_copy(a) }
}

/// The best GPU wall time of `REPS` runs of `kernel`.
fn time(
    kernel: &str,
    variant: Variant,
    buffers: &[&midnight_vulkan::Buffer],
    params: &[u32],
    threads: u64,
) -> Duration {
    let c = context();
    let p = c.pipeline(kernel, variant, PipelineOptions::default()).unwrap();
    // a first run compiles and warms the pipeline
    let mut best = Duration::MAX;
    for _ in 0..=REPS {
        let mut cmds = c.commands().unwrap();
        cmds.dispatch(&p, buffers, params, threads).unwrap();
        best = best.min(cmds.submit().unwrap());
    }
    best
}

fn report(what: &str, variant: Variant, ops: u64, t: Duration) {
    println!(
        "{what:<24} {variant:<6?} {:>10.3} ms {:>10.1} M/s",
        t.as_secs_f64() * 1e3,
        ops as f64 / t.as_secs_f64() / 1e6
    );
}

/// Chained multiplications `a * b^iters`, checked on the first elements.
fn bench_mul<F, const N: usize>(kernel: &str, top: u64, iters: u32)
where
    F: Field,
{
    let c = context();
    let n = 1usize << LOG_N;
    let a = elements::<N>(n, top, 1);
    let b = elements::<N>(n, top, 2);
    let (ba, bb) = (c.upload(&a).unwrap(), c.upload(&b).unwrap());
    let out = c.alloc(size_of_val(a.as_slice())).unwrap();
    for variant in Variant::ALL {
        let t = time(
            kernel,
            variant,
            &[&ba, &bb, &out],
            &[n as u32, iters, 0],
            n as u64,
        );
        let got: Vec<[u64; N]> = c.read(&out).unwrap();
        for i in 0..64 {
            let want = field::<F, N>(&a[i]) * field::<F, N>(&b[i]).pow_vartime([iters as u64]);
            assert!(
                field::<F, N>(&got[i]) == want,
                "{kernel} {variant:?} at {i}"
            );
        }
        report(kernel, variant, n as u64 * iters as u64, t);
    }
}

/// Chained mixed additions over a table of points, checked on the first
/// elements.
fn bench_madd(kernel: &str, iters: u32) {
    let c = context();
    let n = 1usize << LOG_N;
    let m = 1usize << 12;
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let proj: Vec<G1Projective> = (0..m).map(|_| G1Projective::random(&mut rng)).collect();
    let mut table = vec![G1Affine::default(); m];
    G1Projective::batch_normalize(&proj, &mut table);
    let limbs: Vec<[u64; 12]> = table.iter().map(common::affine).collect();
    let bt = c.upload(&limbs).unwrap();
    let out = c.alloc(n * 192).unwrap();
    for variant in Variant::ALL {
        let t = time(
            kernel,
            variant,
            &[&bt, &out],
            &[n as u32, iters, m as u32 - 1],
            n as u64,
        );
        let got: Vec<[u64; 24]> = c.read(&out).unwrap();
        for (i, r) in got.iter().enumerate().take(16) {
            let want: G1Projective = (0..=iters as usize).map(|k| proj[(i + k) % m]).sum();
            assert!(common::xyzz_point(r) == want, "{kernel} {variant:?} at {i}");
        }
        report(kernel, variant, n as u64 * iters as u64, t);
    }
}

#[test]
#[ignore = "measurement; run with --ignored --nocapture"]
fn throughput() {
    let info = context().info();
    println!("{info:#?}");
    println!("2^{LOG_N} invocations, best of {REPS} after a warm-up run");
    bench_mul::<Fq, 4>("bench_fr_mul", 0x73ed_a753_299d_7d48, 64);
    bench_mul::<Fp, 6>("bench_fp_mul", 0x1a01_11ea_397f_e69a, 32);
    bench_madd("bench_g1_madd", 8);
    bench_madd("bench_g1_madd_inline", 8);
}
