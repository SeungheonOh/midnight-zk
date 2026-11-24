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

//! The process-wide host state of one backend. Generic statics do not exist,
//! and two backends may be linked into one binary, so each backend owns one
//! `static` [`HostState`] and returns it from [`Backend::host`].
//!
//! ```ignore
//! static HOST: HostState<MyBackend> = HostState::new();
//! impl Backend for MyBackend {
//!     fn host() -> &'static HostState<Self> { &HOST }
//!     // ...
//! }
//! ```

use std::sync::{
    Condvar, Mutex,
    atomic::{AtomicBool, AtomicU64},
};

use crate::{
    backend::Backend,
    ctx::{Lendable, POOL_LIMIT_UNREAD, Pool, Releases, State},
};

/// Every piece of process-wide state the shared code keeps for backend `B`.
pub struct HostState<B: Backend> {
    /// The pool of zeroed buffers.
    pub(crate) pool: Mutex<Pool<B::Raw>>,
    /// The pool's limit in bytes, or the unread marker.
    pub(crate) pool_limit: AtomicU64,
    /// The tickets of background zeroing.
    pub(crate) releases: Mutex<Releases>,
    /// Notified whenever a ticket of `releases` completes.
    pub(crate) released: Condvar,
    /// Zeroings that panicked or failed since the last request report.
    pub(crate) erase_faults: AtomicU64,
    /// Bytes held in the backend's buffers, idle pool buffers included.
    pub(crate) bytes_now: AtomicU64,
    /// The peak of `bytes_now` since process start or the last request.
    pub(crate) bytes_peak: AtomicU64,
    /// Bytes in buffers that are not idle in the pool.
    pub(crate) in_use: AtomicU64,
    /// The peak of `in_use` since process start or the last request.
    pub(crate) in_use_peak: AtomicU64,
    /// Whether a request is open.
    pub(crate) request_open: AtomicBool,
    /// The device lock and the per-request buffers it guards.
    pub(crate) state: Mutex<Option<State<B>>>,
    /// Whether a per-request buffer may exist.
    pub(crate) scratch_any: AtomicBool,
    /// The idle buffers of `ctx::take_request_buffer`.
    pub(crate) lendable: Mutex<Lendable<B>>,
    /// The test switches.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) switches: crate::ctx::Switches,
}

impl<B: Backend> HostState<B> {
    /// The state of a process that has not used the backend yet.
    #[allow(clippy::new_without_default)]
    pub const fn new() -> Self {
        HostState {
            pool: Mutex::new(Pool::new()),
            pool_limit: AtomicU64::new(POOL_LIMIT_UNREAD),
            releases: Mutex::new(Releases::new()),
            released: Condvar::new(),
            erase_faults: AtomicU64::new(0),
            bytes_now: AtomicU64::new(0),
            bytes_peak: AtomicU64::new(0),
            in_use: AtomicU64::new(0),
            in_use_peak: AtomicU64::new(0),
            request_open: AtomicBool::new(false),
            state: Mutex::new(None),
            scratch_any: AtomicBool::new(false),
            lendable: Mutex::new(Lendable::new()),
            #[cfg(any(test, feature = "test-support"))]
            switches: crate::ctx::Switches::new(),
        }
    }
}

impl<B: Backend> std::fmt::Debug for HostState<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostState").field("backend", &B::NAME).finish_non_exhaustive()
    }
}
