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

//! The plan of one MSM (window width, chunks) and the device share of the
//! CPU/device split. sppark computes the plan by formula; here both come from
//! the tables the backend measured ([`Tuning`]).

use crate::{Backend, Tuning, lg2_nearest};

/// Bits of a scalar (sppark `scalar_t::bit_length()`).
pub const NBITS: u32 = 255;
/// Bytes of the temporary sort buffer one pass may use.
const SORT_TEMP_BYTES: usize = 64 << 20;

/// The window plan of an MSM of `n` points.
#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub n: usize,
    pub wbits: u32,
    pub nwins: u32,
    /// Chunks (command buffers).
    pub batch: usize,
    /// Points per chunk (a multiple of 32).
    pub stride: usize,
    /// `wbits - 1`: bucket index bits.
    pub bits: u32,
    /// Bits of the upper sort (`bits / 2`).
    pub top_bits: u32,
    /// Bits of the lower sort (`bits - top_bits`).
    pub low_bits: u32,
    /// Bits of the top window (sppark `top`).
    pub top: u32,
    /// Windows per sort pass.
    pub pass: u32,
}

/// sppark `lg2`: floor of log2, 0 for 0.
pub const fn lg2(mut n: usize) -> u32 {
    let mut r = 0;
    while n > 1 {
        n >>= 1;
        r += 1;
    }
    r
}

/// `(wbits, chunks)` of [`Tuning::msm_windows`] for `n` points: sizes
/// outside the table take the nearest end entry; other lengths take the
/// entry of [`lg2_nearest`], so a split's device share uses its power's
/// entry.
fn measured(n: usize, t: &Tuning) -> (u32, usize) {
    let w = t.msm_windows;
    let (lo, hi) = (w[0].0, w[w.len() - 1].0);
    let k = lg2_nearest(n).clamp(lo, hi);
    let e = w[(k - lo) as usize];
    (e.1, e.2)
}

/// The plan of an MSM of `n` points.
pub fn plan(n: usize, t: &Tuning) -> Plan {
    let (wbits, batch) = measured(n, t);
    let nwins = (NBITS - 1) / wbits + 1;
    // cap the GPU time of one command buffer (sppark has no such bound)
    let max_stride = (t.msm_chunk_adds / nwins as usize) & !31;
    let stride = n.div_ceil(batch).next_multiple_of(32).min(max_stride);
    let batch = n.div_ceil(stride);
    let bits = wbits - 1;
    let top_bits = bits / 2;
    let low_bits = bits - top_bits;
    let pass = (SORT_TEMP_BYTES / (8 * stride)).clamp(2, nwins as usize) as u32;
    Plan {
        n,
        wbits,
        nwins,
        batch,
        stride,
        bits,
        top_bits,
        low_bits,
        top: NBITS - wbits * (nwins - 1),
        pass,
    }
}

/// The denominator of the shares in [`Tuning::msm_shares`].
pub const SHARE_DEN: usize = 64;

/// The device share `m` of an MSM of `n` points: the device computes `[0, m)`
/// and the CPU `[m, n)`. A function of `n` and the path only;
/// `{ENV_PREFIX}_MSM_CPU_SHARE=off` or no CPU pool gives `m = n`.
pub fn device_share<B: Backend>(n: usize, table: bool) -> usize {
    if std::env::var(format!("{}_MSM_CPU_SHARE", B::ENV_PREFIX)).as_deref() == Ok("off")
        || super::par::cpu_pool::<B>().is_none()
    {
        return n;
    }
    let k = lg2_nearest(n);
    let shares = B::tuning().msm_shares;
    let (lo, hi) = (shares[0].0, shares[shares.len() - 1].0);
    if k < lo {
        return n;
    }
    // in range: `lo <= k.min(hi) <= hi`, one entry per log2
    let e = shares[(k.min(hi) - lo) as usize];
    let frac = if table { e.1 } else { e.2 };
    (n * frac / SHARE_DEN).clamp(1, n)
}
