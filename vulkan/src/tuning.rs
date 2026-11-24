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

//! The constants of the Vulkan backend ([`TUNING`]). They start from the
//! Metal backend's values, measured on Apple silicon, until they are measured
//! on Vulkan devices.

use midnight_accel::{Thresholds, Tuning};

/// `(log2 n, wbits, chunks)` of the MSM by size.
const MSM_WINDOWS: &[(u32, u32, usize)] = &[
    (13, 11, 1),
    (14, 11, 1),
    (15, 11, 1),
    (16, 13, 1),
    (17, 13, 1),
    (18, 13, 1),
    (19, 13, 1),
    (20, 15, 1),
    (21, 16, 1),
    (22, 16, 1),
];

/// The device share by size, `(log2 n, table path, one-shot path)` in 64ths
/// of the points.
const MSM_SHARES: &[(u32, usize, usize)] = &[
    (13, 64, 40),
    (14, 64, 40),
    (15, 64, 40),
    (16, 64, 48),
    (17, 54, 52),
    (18, 54, 40),
    (19, 54, 40),
    (20, 52, 40),
    (21, 52, 48),
    (22, 52, 44),
];

/// The Vulkan backend's constants.
pub(crate) const TUNING: Tuning = Tuning {
    // per-class break-evens against the stock CPU code; the one-shot MSM
    // never breaks even because of its G1 check
    thresholds: Thresholds {
        fft: 15,
        ifft: 15,
        msm_table: 13,
        msm_one_shot: None,
        quotient: 14,
        explicit: false,
    },
    msm_windows: MSM_WINDOWS,
    msm_shares: MSM_SHARES,
    // a command buffer stays well below the GPU watchdog
    msm_chunk_adds: 1 << 25,
    // sppark: `sm_count / 3`, a power of two
    msm_sort_groups: 8,
    // sppark: `sm_count` blocks
    msm_ones_groups: 32,
    // one invocation sums a bucket serially: this bound keeps that chain far
    // below the watchdog of AMD's graphics ring (10 s), also on integrated
    // GPUs, whose single-thread rate is a fraction of a discrete GPU's
    msm_hot_bucket_limit: 1 << 12,
    // crossovers against the stock CPU MSM: `n/5` at large `n`, plus a fixed
    // part for the device's fixed cost near the threshold
    msm_sparse_div: 5,
    msm_sparse_fixed: 1 << 12,
    // a fraction of a second of GPU time
    ntt_run_budget: 1_500_000_000,
    quotient_target_nanos: 250_000_000,
    // keeps the scratch at `nslots * 2 MiB`
    quotient_threads: 1 << 16,
    // a quarter of the device-local memory budget
    pool_divisor: 4,
};

/// The constants of discrete GPUs (device memory the host does not map).
pub(crate) const DISCRETE: Tuning = Tuning { ..TUNING };

/// The constants of `info`'s device class: [`DISCRETE`] for a GPU whose
/// memory the host does not map, else [`TUNING`] (Apple GPUs through
/// MoltenVK, integrated GPUs, CPU implementations).
pub(crate) fn for_device(info: &crate::device::VulkanInfo) -> &'static Tuning {
    let base = if info.unified_memory { &TUNING } else { &DISCRETE };
    tmp_override(base)
}

// TEMPORARY exploration knobs
fn tmp_override(base: &'static Tuning) -> &'static Tuning {
    let get = |k: &str| std::env::var(k).ok();
    if get("VKT").is_none() {
        return base;
    }
    let mut t = *base;
    let parse3 = |s: &str| -> &'static [(u32, u32, usize)] {
        let v: Vec<(u32, u32, usize)> = s
            .split(';')
            .map(|e| {
                let p: Vec<usize> = e.split(',').map(|x| x.parse().unwrap()).collect();
                (p[0] as u32, p[1] as u32, p[2])
            })
            .collect();
        Box::leak(v.into_boxed_slice())
    };
    if let Some(s) = get("VKT_WINDOWS") {
        t.msm_windows = parse3(&s);
    }
    if let Some(s) = get("VKT_SHARES") {
        let v: Vec<(u32, usize, usize)> = s
            .split(';')
            .map(|e| {
                let p: Vec<usize> = e.split(',').map(|x| x.parse().unwrap()).collect();
                (p[0] as u32, p[1], p[2])
            })
            .collect();
        t.msm_shares = Box::leak(v.into_boxed_slice());
    }
    if let Some(s) = get("VKT_CHUNK") {
        t.msm_chunk_adds = s.parse().unwrap();
    }
    if let Some(s) = get("VKT_SORTG") {
        t.msm_sort_groups = s.parse().unwrap();
    }
    if let Some(s) = get("VKT_ONESG") {
        t.msm_ones_groups = s.parse().unwrap();
    }
    if let Some(s) = get("VKT_NTTB") {
        t.ntt_run_budget = s.parse().unwrap();
    }
    if let Some(s) = get("VKT_QT") {
        t.quotient_threads = s.parse().unwrap();
    }
    if let Some(s) = get("VKT_QNS") {
        t.quotient_target_nanos = s.parse().unwrap();
    }
    if let Some(s) = get("VKT_TH") {
        let p: Vec<u32> = s.split(',').map(|x| x.parse().unwrap()).collect();
        t.thresholds.fft = p[0];
        t.thresholds.ifft = p[1];
        t.thresholds.msm_table = p[2];
        t.thresholds.quotient = p[3];
    }
    Box::leak(Box::new(t))
}
