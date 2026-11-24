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

//! The backend-independent host side of the accelerator hooks of
//! `midnight-proofs`: what does not depend on the GPU API.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod account;
pub mod backend;
pub mod ctx;
pub mod field;
pub mod host;
#[doc(hidden)]
pub mod msm;
#[doc(hidden)]
pub mod ntt;
#[doc(hidden)]
pub mod quotient;

pub use backend::{Backend, DeviceInfo, Encoder, RawBuffer};
pub use host::HostState;

/// A scalar: `blst_fr`, four little-endian limbs, Montgomery form, reduced
/// modulo [`field::FR_MODULUS`].
pub type Scalar = [u64; 4];
/// An affine G1 point: `blst_p1_affine`, `x` then `y`, Montgomery form; all
/// zero is the point at infinity.
pub type G1AffineLimbs = [u64; 12];
/// A Jacobian G1 point: `blst_p1` limbs.
pub type G1JacobianLimbs = [u64; 18];

/// The largest log2 transform size taken.
pub const MAX_LOG_N: u32 = 28;
/// The largest log2 MSM length taken; a longer one declines (`shape`).
pub const MAX_MSM_LOG: u32 = 28;

/// `floor(log2(n * sqrt 2))`: the size class of an MSM length, so the `n - 1`
/// coefficients of a multi-open are in the class of `n`.
pub const fn lg2_nearest(n: usize) -> u32 {
    // 181 / 128 = 1.4140625, just below sqrt 2
    let mut m = ((n as u128 * 181) >> 7) as usize;
    let mut r = 0;
    while m > 1 {
        m >>= 1;
        r += 1;
    }
    r
}

/// The log2 size from which each class goes to the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Thresholds {
    /// Forward FFTs: the transform size `log_n`.
    pub fft: u32,
    /// Inverse FFTs: the transform size `log_n`.
    pub ifft: u32,
    /// MSMs on a registered SRS table: the size class of the length.
    pub msm_table: u32,
    /// MSMs with no registered table: the size class of the length; `None`:
    /// never (the default).
    pub msm_one_shot: Option<u32>,
    /// Quotients: the extended domain `ext_k`.
    pub quotient: u32,
    /// Whether one explicit value applies to every class.
    pub explicit: bool,
}

impl Thresholds {
    /// One value for every class (an explicit `min_k`).
    pub const fn all(k: u32) -> Thresholds {
        Thresholds {
            fft: k,
            ifft: k,
            msm_table: k,
            msm_one_shot: Some(k),
            quotient: k,
            explicit: true,
        }
    }

    /// `(class, log2 size)` pairs for reports; `None`: never.
    pub fn named(&self) -> [(&'static str, Option<u32>); 5] {
        [
            ("fft", Some(self.fft)),
            ("ifft", Some(self.ifft)),
            ("msm-table", Some(self.msm_table)),
            ("msm-one-shot", self.msm_one_shot),
            ("quotient-ext", Some(self.quotient)),
        ]
    }
}

/// Per-class break-evens against the stock CPU code, measured on Apple
/// silicon. The one-shot MSM never breaks even because of its G1 check.
pub const DEFAULT_THRESHOLDS: Thresholds = Thresholds {
    fft: 15,
    ifft: 15,
    msm_table: 13,
    msm_one_shot: None,
    quotient: 14,
    explicit: false,
};

/// Calls served in one class since the counters were last reset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassStats {
    /// Calls the device computed.
    pub served: u64,
    /// GPU time of those calls in nanoseconds.
    pub device_nanos: u64,
}

/// Per-class service counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// MSMs.
    pub msm: ClassStats,
    /// Forward FFTs.
    pub fft: ClassStats,
    /// Inverse FFTs.
    pub ifft: ClassStats,
    /// Quotients.
    pub quotient: ClassStats,
}

/// What a request's end report erased.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WipeStats {
    /// Regions zeroed: the per-request buffers and, in check mode, the idle pool
    /// buffers.
    pub regions: u64,
    /// Their bytes.
    pub bytes: u64,
    /// Whether every region was read back after its zeroing.
    pub checked: bool,
    /// Regions read back (check mode).
    pub checked_regions: u64,
    /// Regions that read back non-zero (check mode).
    pub nonzero_regions: u64,
    /// Regions that could not be read back from the host (check mode).
    pub check_failures: u64,
    /// Of `regions`: idle pool buffers read back (check mode).
    pub pool_regions: u64,
    /// Their bytes.
    pub pool_bytes: u64,
    /// Zeroings that panicked since the last report. Each was redone on the
    /// calling thread and its buffer released, not pooled.
    pub erasure_failures: u64,
}
