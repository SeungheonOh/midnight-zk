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

//! The device-neutral parts of the device context: errors, plain-data
//! element types and host copies.

use std::sync::{Arc, Mutex};

use crate::account::Reason;

/// A device failure, mapped to a decline reason by [`Error::reason`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No usable device, or the context could not be made.
    NoDevice(String),
    /// A buffer could not be allocated (or is above `maxBufferLength`).
    Memory(String),
    /// A pipeline, encoder or command buffer failed.
    Device(String),
    /// A scalar argument of a device building block is not reduced.
    Unreduced(String),
}

impl Error {
    /// The decline reason of this error.
    pub fn reason(&self) -> &'static str {
        Reason::of(self).name()
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoDevice(m) => write!(f, "no device: {m}"),
            Error::Memory(m) => write!(f, "memory: {m}"),
            Error::Device(m) => write!(f, "device error: {m}"),
            Error::Unreduced(m) => write!(f, "unreduced: {m}"),
        }
    }
}

/// Plain data: every bit pattern is valid and there is no padding.
///
/// # Safety
/// Implement it only for such types.
pub unsafe trait Pod: Copy + 'static {}
// SAFETY: integers and arrays of them have no padding and no invalid values.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}
// SAFETY: as above.
unsafe impl Pod for u64 {}
// SAFETY: as above.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

// ---------------------------------------------------------------------------
// Host copies

/// Copies `src` into `dst` on several threads when long. Never panics: a part
/// whose thread cannot be made is copied here, so a copy into a caller's slice
/// never stops half way.
pub fn par_copy(dst: &mut [crate::Scalar], src: &[crate::Scalar]) {
    const PART: usize = 1 << 15;
    if src.len() <= 2 * PART {
        dst.copy_from_slice(src);
        return;
    }
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get()).min(8);
    let per = src.len().div_ceil(threads).max(PART);
    std::thread::scope(|s| {
        for (d, c) in dst.chunks_mut(per).zip(src.chunks(per)) {
            // the part stays reachable from here if the thread is not made
            // (a failed `spawn_scoped` drops its closure)
            let part = Arc::new(Mutex::new(Some((d, c))));
            let theirs = Arc::clone(&part);
            let spawned = std::thread::Builder::new().spawn_scoped(s, move || {
                if let Some((d, c)) = theirs.lock().unwrap_or_else(|e| e.into_inner()).take() {
                    d.copy_from_slice(c);
                }
            });
            if spawned.is_err()
                && let Some((d, c)) = part.lock().unwrap_or_else(|e| e.into_inner()).take()
            {
                d.copy_from_slice(c);
            }
        }
    });
}
