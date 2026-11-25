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

//! The Vulkan backend's tuning constants: [`TUNING`] (the Metal backend's
//! values except `msm_hot_bucket_limit`) and [`ADRENO`] for Qualcomm Adreno
//! GPUs.

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
    // break-evens against the stock CPU code; the one-shot MSM never breaks
    // even because of its G1 check
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
    msm_integrate_groups: 1,
    // one invocation sums a bucket serially: this bound keeps that chain
    // below AMD's graphics-ring watchdog (10 s), also on integrated GPUs
    msm_hot_bucket_limit: 1 << 12,
    // crossover against the stock CPU MSM: `n/5` plus a fixed cost
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

/// `(log2 n, wbits, chunks)` of the MSM on Qualcomm Adreno GPUs.
/// Avoids `wbits = 14` (19 windows, 2x buckets of `wbits = 13`'s 20 windows)
/// and `wbits = 16` below `2^21` (16 windows, 2x buckets of `wbits = 15`'s
/// exact `255 / 15 = 17` windows) to minimize `g1_madd + 2 * g1_add` work and
/// bucket DRAM footprint.
const ADRENO_MSM_WINDOWS: &[(u32, u32, usize)] = &[
    (13, 11, 1),
    (14, 11, 1),
    (15, 12, 1),
    (16, 13, 1),
    (17, 13, 1),
    (18, 15, 1),
    (19, 15, 1),
    (20, 15, 1),
    (21, 16, 1),
    (22, 16, 1),
];

/// The device share of the MSM on Qualcomm Adreno GPUs (in 64ths): balanced
/// so GPU (with length-binned lazy-Fp accumulation) and 8-core Oryon CPU
/// shares finish together without overloading GPU thermals at k >= 18.
const ADRENO_MSM_SHARES: &[(u32, usize, usize)] = &[
    (13, 30, 30),
    (14, 30, 30),
    (15, 32, 32),
    (16, 35, 35),
    (17, 36, 36),
    (18, 37, 37),
    (19, 39, 39),
    (20, 41, 41),
    (21, 44, 44),
    (22, 46, 46),
];

/// The constants of Qualcomm Adreno GPUs (vendor 0x5143), measured on
/// Snapdragon SM8850 / Adreno 840.
pub(crate) const ADRENO: Tuning = Tuning {
    thresholds: Thresholds {
        fft: 15,
        ifft: 15,
        // below k=15, mobile CPU cores beat GPU dispatch
        msm_table: 15,
        msm_one_shot: None,
        quotient: 14,
        explicit: false,
    },
    msm_windows: ADRENO_MSM_WINDOWS,
    msm_shares: ADRENO_MSM_SHARES,
    // With masked scalars (`s + r`), bucket 1 of window 0 holds only ~m/2^(w-1)
    // points; 4 workgroups (512 threads, 16 output records) keep `msm_ones`
    // sub-millisecond while minimizing CPU `collect()` work.
    msm_ones_groups: 4,
    // 4 workgroups of 64 threads per window (`4 * nwins` = 68-80 single-wave
    // workgroups) keeps all Adreno shader processors busy even with 1 WG/SP
    // SRAM-staged `g1_add_sh` while keeping `collect()` sub-millisecond.
    msm_integrate_groups: 4,
    ..TUNING
};

/// The constants of `info`'s device class: [`ADRENO`] for Qualcomm Adreno
/// GPUs, else [`TUNING`].
pub(crate) fn for_device(info: &crate::device::VulkanInfo) -> &'static Tuning {
    if info.unified_memory && info.vendor_id == 0x5143 {
        &ADRENO
    } else {
        &TUNING
    }
}
