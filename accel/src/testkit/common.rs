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

//! Helpers shared by the integration tests.

#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard, Once};

use ff::Field;
use midnight_curves::Fq;
use midnight_proofs::gpu::{GpuHooks, QuotientFn, install};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;

use crate::Backend;

/// Serializes the tests that change process-wide thresholds, switches or
/// counters.
pub fn serial<B: Backend>() -> MutexGuard<'static, ()> {
    static LOCK: Mutex<()> = Mutex::new(());
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert!(crate::device::<B>(), "no usable {} device", B::NAME);
    g
}

pub fn random_scalars(n: usize, seed: u64) -> Vec<Fq> {
    (0..n)
        .into_par_iter()
        .chunks(1 << 12)
        .flat_map_iter(|c| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ ((c[0] as u64) << 16));
            c.into_iter().map(move |_| Fq::random(&mut rng))
        })
        .collect()
}

pub fn declines<B: Backend>(class: &str, reason: &str) -> u64 {
    crate::declines::<B>()
        .into_iter()
        .filter(|d| d.0 == class && d.1 == reason)
        .map(|d| d.2)
        .sum()
}

/// Installs the device hooks once per process, with `quotient` as the
/// quotient hook.
pub fn install_hooks<B: Backend>(quotient: QuotientFn) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        assert!(install(GpuHooks {
            msm_g1: Some(crate::msm_g1::<B>),
            fft: Some(crate::fft::<B>),
            ifft: Some(crate::ifft::<B>),
            quotient: Some(quotient),
        }))
    });
}
