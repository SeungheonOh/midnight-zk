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

//! Decline and service accounting. A kernel module ends every call with
//! exactly one of [`served`] or [`decline`].

use std::{
    collections::BTreeMap,
    sync::{
        Mutex,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

use crate::backend::Backend;

/// A hook class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The G1 MSM.
    Msm,
    /// The forward FFT.
    Fft,
    /// The inverse FFT.
    Ifft,
    /// The quotient.
    Quotient,
}

impl Class {
    /// The class's name in the decline report and the reports.
    pub const fn name(self) -> &'static str {
        match self {
            Class::Msm => "msm",
            Class::Fft => "fft",
            Class::Ifft => "ifft",
            Class::Quotient => "quotient",
        }
    }
    /// The index of the class in the service counters.
    const fn index(self) -> usize {
        self as usize
    }
}

/// A decline reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// Stub build, no usable device, or `MIDNIGHT_METAL=off`.
    NoDevice,
    /// The size is below the class's threshold (`crate::thresholds`).
    BelowMinK,
    /// A length, root, job layout or version the device does not take.
    Shape,
    /// A scalar not below the modulus.
    Unreduced,
    /// A buffer above the largest buffer, a failed allocation, or a budget.
    Memory,
    /// A pipeline, encoder or command buffer failed.
    DeviceError,
    /// The backend panicked; the hook entry caught it (`crate::guarded`).
    Panic,
    /// An MSM base outside G1.
    BasesNotInSubgroup,
    /// An MSM whose bases no registered table serves, with the one-shot path
    /// off.
    UnregisteredBases,
}

impl Reason {
    /// The reason's name in the decline report.
    pub const fn name(self) -> &'static str {
        match self {
            Reason::NoDevice => "no-device",
            Reason::BelowMinK => "below-min-k",
            Reason::Shape => "shape",
            Reason::Unreduced => "unreduced",
            Reason::Memory => "memory",
            Reason::DeviceError => "device-error",
            Reason::Panic => "panic",
            Reason::BasesNotInSubgroup => "bases-not-in-subgroup",
            Reason::UnregisteredBases => "unregistered-bases",
        }
    }

    /// The decline reason of a device-context error.
    pub fn of(e: &crate::ctx::Error) -> Self {
        match e {
            crate::ctx::Error::NoDevice(_) => Reason::NoDevice,
            crate::ctx::Error::Memory(_) => Reason::Memory,
            crate::ctx::Error::Device(_) => Reason::DeviceError,
            crate::ctx::Error::Unreduced(_) => Reason::Unreduced,
        }
    }
}

/// The decline and service counters of one backend.
pub(crate) struct Account {
    /// Declines by `(class, reason)`.
    declines: Mutex<BTreeMap<(&'static str, &'static str), u64>>,
    /// Calls served, and their GPU nanoseconds, by [`Class::index`].
    served: [AtomicU64; 4],
    nanos: [AtomicU64; 4],
    /// Host time by [`HostPhase`], in nanoseconds, and the command buffers.
    host: [AtomicU64; 6],
}

/// A part of the host time around the device work ([`crate::HostTimes`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HostPhase {
    Copy = 0,
    Zero = 1,
    ZeroBackground = 2,
    Run = 3,
    RunGpu = 4,
    Runs = 5,
}

/// Adds `n` to the host counter `p`.
pub(crate) fn add_host<B: Backend>(p: HostPhase, n: u64) {
    account::<B>().host[p as usize].fetch_add(n, Relaxed);
}

/// Adds the time since `t` to the host counter `p`.
pub(crate) fn add_host_since<B: Backend>(p: HostPhase, t: std::time::Instant) {
    add_host::<B>(p, t.elapsed().as_nanos().min(u64::MAX as u128) as u64);
}

/// The host times since the last [`reset`].
pub fn host_times<B: Backend>() -> crate::HostTimes {
    let h = &account::<B>().host;
    let at = |p: HostPhase| h[p as usize].load(Relaxed);
    crate::HostTimes {
        copy_nanos: at(HostPhase::Copy),
        zero_nanos: at(HostPhase::Zero),
        zero_background_nanos: at(HostPhase::ZeroBackground),
        run_nanos: at(HostPhase::Run),
        run_gpu_nanos: at(HostPhase::RunGpu),
        runs: at(HostPhase::Runs),
    }
}

impl Account {
    /// Every counter zero.
    pub(crate) const fn new() -> Self {
        Account {
            declines: Mutex::new(BTreeMap::new()),
            served: [const { AtomicU64::new(0) }; 4],
            nanos: [const { AtomicU64::new(0) }; 4],
            host: [const { AtomicU64::new(0) }; 6],
        }
    }
}

fn account<B: Backend>() -> &'static Account {
    &B::host().account
}

/// Counts a decline; returns `false` (the decline value of a `bool` hook).
pub fn decline<B: Backend>(class: Class, reason: Reason) -> bool {
    *account::<B>()
        .declines
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry((class.name(), reason.name()))
        .or_insert(0) += 1;
    false
}

/// Counts a call served by the device, with its GPU time.
pub fn served<B: Backend>(class: Class, device_nanos: u64) {
    let a = account::<B>();
    a.served[class.index()].fetch_add(1, Relaxed);
    a.nanos[class.index()].fetch_add(device_nanos, Relaxed);
}

/// Every decline: `(class, reason, count)`.
pub fn declines<B: Backend>() -> Vec<(&'static str, &'static str, u64)> {
    account::<B>()
        .declines
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .map(|(k, v)| (k.0, k.1, *v))
        .collect()
}

/// The service counters of `class`.
pub fn class_stats<B: Backend>(class: Class) -> crate::ClassStats {
    let a = account::<B>();
    crate::ClassStats {
        served: a.served[class.index()].load(Relaxed),
        device_nanos: a.nanos[class.index()].load(Relaxed),
    }
}

/// Clears every counter.
pub fn reset<B: Backend>() {
    let a = account::<B>();
    for c in a.served.iter().chain(a.nanos.iter()).chain(a.host.iter()) {
        c.store(0, Relaxed);
    }
    a.declines.lock().unwrap_or_else(|e| e.into_inner()).clear();
}
