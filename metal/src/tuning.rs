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

//! The constants measured on Apple silicon ([`TUNING`]).

use midnight_accel::{Thresholds, Tuning};

/// `(log2 n, wbits, chunks)`, measured on Apple silicon (table path, CPU share
/// off).
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
/// of the points, measured on Apple silicon. At `2^13` to `2^15` the table
/// path keeps every point on the device: inside a proof the CPU share keeps
/// all CPU threads busy and the device share slows down over successive
/// proofs.
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

/// The Metal backend's constants.
pub(crate) const TUNING: Tuning = Tuning {
    // per-class break-evens against the stock CPU code, measured on whole
    // proofs; the one-shot MSM never breaks even because of its G1 check
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
    // a command buffer stays near or below 0.5 s of GPU time
    msm_chunk_adds: 1 << 25,
    // sppark: `sm_count / 3`, a power of two
    msm_sort_groups: 8,
    // sppark: `sm_count` blocks
    msm_ones_groups: 32,
    msm_hot_bucket_limit: 1 << 16,
    // measured crossovers against the stock CPU MSM: `n/5` at large `n`, plus
    // a fixed part for the device's fixed cost near the threshold
    msm_sparse_div: 5,
    msm_sparse_fixed: 1 << 12,
    // a fraction of a second of GPU time
    ntt_run_budget: 1_500_000_000,
    quotient_target_nanos: 250_000_000,
    // near the measured best numerator time while keeping the scratch at
    // `nslots * 2 MiB`
    quotient_threads: 1 << 16,
    // a quarter of `recommendedMaxWorkingSetSize`
    pool_divisor: 4,
};

#[cfg(test)]
mod tests {
    use midnight_accel::msm::plan as super_plan;
    use midnight_accel::msm::plan::{NBITS, SHARE_DEN, device_share};

    use super::*;
    use crate::Metal;

    fn plan(n: usize) -> midnight_accel::msm::plan::Plan {
        super_plan::plan(n, &TUNING)
    }

    #[test]
    fn plan_follows_measured_table() {
        // (log2 n, wbits, nwins, chunks after the command-buffer bound)
        for (k, wbits, nwins, batch) in [
            (13, 11, 24, 1),
            (14, 11, 24, 1),
            (15, 11, 24, 1),
            (16, 13, 20, 1),
            (17, 13, 20, 1),
            (18, 13, 20, 1),
            (19, 13, 20, 1),
            (20, 15, 17, 1),
            (21, 16, 16, 1),
            (22, 16, 16, 2),
        ] {
            let p = plan(1 << k);
            assert_eq!((p.wbits, p.nwins, p.batch), (wbits, nwins, batch), "2^{k}");
            assert_eq!(p.top, NBITS - wbits * (nwins - 1));
            assert!(p.stride * p.batch >= 1 << k, "2^{k}");
        }
        // outside the table: the end entries; a non-power of two: its
        // nearest log2 (a device share of 52/64 of 2^20 takes the 2^20 entry)
        assert_eq!(plan(1).wbits, 11);
        assert_eq!(plan(1000).wbits, 11);
        assert_eq!(plan((1 << 20) * 52 / 64).wbits, 15);
        assert_eq!(plan((1 << 20) * 46 / 64).wbits, 15);
        assert_eq!(plan((1 << 20) * 45 / 64).wbits, 13);
        assert_eq!(plan((1 << 21) - 1).wbits, 16);
        assert_eq!(plan((1 << 21) + 5).wbits, 16);
        // the command-buffer bound up to 2^28
        for k in 0..=crate::MAX_MSM_LOG {
            for n in [1usize << k, (1usize << k) + 33] {
                let p = plan(n);
                assert!(p.stride * p.nwins as usize <= TUNING.msm_chunk_adds, "{n}");
                assert!(
                    p.stride * p.batch >= n && p.stride * (p.batch - 1) < n,
                    "{n}"
                );
                assert!(p.stride.is_multiple_of(32), "{n}");
                if k >= 22 {
                    assert_eq!(p.wbits, 16, "{n}");
                }
            }
        }
        assert_eq!(plan(1 << 23).batch, 4);
        assert_eq!(plan(1 << 28).batch, 128);
    }

    #[test]
    fn device_share_follows_measured_table() {
        if std::env::var("MIDNIGHT_METAL_MSM_CPU_SHARE").as_deref() == Ok("off") {
            return;
        }
        for &(k, t, o) in MSM_SHARES {
            let n = 1usize << k;
            assert_eq!(device_share::<Metal>(n, true), n * t / SHARE_DEN, "2^{k}");
            assert_eq!(device_share::<Metal>(n, false), n * o / SHARE_DEN, "2^{k}");
        }
        assert_eq!(device_share::<Metal>(1000, true), 1000);
        assert_eq!(device_share::<Metal>(1 << 12, false), 1 << 12);
        // a length just below a power of two takes that power's share
        assert_eq!(
            device_share::<Metal>((1 << 20) - 1, true),
            ((1 << 20) - 1) * 52 / 64
        );
        assert_eq!(device_share::<Metal>(1 << 24, true), (1 << 24) * 52 / 64);
    }
}
