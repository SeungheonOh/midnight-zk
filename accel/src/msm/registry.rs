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

//! Registered SRS base tables and the MSM counters.
//!
//! A table is keyed by the address and length of the registered slice and
//! serves that slice and its prefixes. Every use compares each point of the
//! caller's slice with the table before any scalar is read; a mismatch drops
//! the table. The registry lock is held only to look up and pin a table
//! (an `Arc`), so a table dropped during an MSM lives until that MSM ends.

use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock, RwLock,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

use super::{
    checks::{first_not_in_g1, points_equal},
    par::{lock, read, write},
};
use crate::{
    Backend, G1AffineLimbs, G1JacobianLimbs,
    ctx::{Buffer, Ctx, Error},
};

/// Where a table's points are: the host copy until the first MSM, then the
/// device buffer alone when the host reads device memory in place
/// ([`Backend::unified`]), else the host copy and the device buffer.
pub(super) enum Store<B: Backend> {
    Host(Vec<G1AffineLimbs>),
    Device(Buffer<B>),
    Both(Vec<G1AffineLimbs>, Buffer<B>),
}

impl<B: Backend> Store<B> {
    /// `f` over the first `len` points (fewer if the store holds fewer);
    /// `None` when the device buffer has no host view.
    fn with_points<R>(&self, len: usize, f: impl FnOnce(&[G1AffineLimbs]) -> R) -> Option<R> {
        let first = |all: &[G1AffineLimbs]| f(&all[..len.min(all.len())]);
        match self {
            Store::Host(h) | Store::Both(h, _) => Some(first(h)),
            Store::Device(b) => b.host_read(first).ok(),
        }
    }

    /// The device buffer, once uploaded.
    pub(super) fn device(&self) -> Option<&Buffer<B>> {
        match self {
            Store::Host(_) => None,
            Store::Device(b) | Store::Both(_, b) => Some(b),
        }
    }
}

/// `MSM(r)` of a table by length (public: the mask and the points are).
pub(super) type MaskCache = Mutex<HashMap<usize, G1JacobianLimbs>>;

/// A registered table. Its points never change after registration.
pub(crate) struct Table<B: Backend> {
    id: u64,
    len: usize,
    /// The first index whose point is not in G1, or `len`. An MSM on a longer
    /// prefix declines.
    pub(super) first_bad: usize,
    pub(super) store: RwLock<Store<B>>,
    pub(super) masks: MaskCache,
}

impl<B: Backend> Table<B> {
    /// Whether the table still holds the caller's points (every point
    /// compared).
    fn serves(&self, bases: &[G1AffineLimbs]) -> bool {
        if bases.len() > self.len {
            return false;
        }
        let st = read(&self.store);
        st.with_points(self.len, |t| {
            t.len() >= bases.len() && points_equal(&t[..bases.len()], bases)
        })
        .unwrap_or(false)
    }

    /// Uploads the host copy to a device buffer, once (the caller holds the
    /// device lock). The host copy is kept unless the device is unified.
    pub(super) fn upload(&self, ctx: &Ctx<B>) -> Result<(), Error> {
        if matches!(*read(&self.store), Store::Host(_)) {
            let mut w = write(&self.store);
            if let Store::Host(h) = &mut *w {
                let b = ctx.upload(h)?;
                let host = std::mem::take(h);
                *w = if ctx.unified() {
                    Store::Device(b)
                } else {
                    Store::Both(host, b)
                };
                B::host().msm.uploads.fetch_add(1, Relaxed);
            }
        }
        Ok(())
    }
}

/// The registry: tables keyed by (address, length) of the registered slice.
type Registry<B> = HashMap<(usize, usize), Arc<Table<B>>>;

/// The MSM state of one backend: the registry and the counters.
pub(crate) struct MsmState<B: Backend> {
    tables: Mutex<Option<Registry<B>>>,
    next_id: AtomicU64,
    /// Tables dropped at an MSM because the caller's slice no longer matched.
    stale: AtomicU64,
    /// Table uploads.
    uploads: AtomicU64,
    /// MSMs served from a registered table.
    pub(crate) tables_served: AtomicU64,
    /// One-shot MSMs (past their G1 check).
    pub(crate) one_shot: AtomicU64,
    /// Per-call G1 checks of the one-shot path started.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) g1_checks: AtomicU64,
    /// Whether every MSM pipeline has the SIMD width the kernels assume.
    pub(crate) simd_ok: OnceLock<bool>,
    /// The thread pool of the CPU share.
    pub(crate) cpu_pool: OnceLock<Option<rayon::ThreadPool>>,
}

impl<B: Backend> MsmState<B> {
    /// No table, every counter zero.
    pub(crate) const fn new() -> Self {
        MsmState {
            tables: Mutex::new(None),
            next_id: AtomicU64::new(1),
            stale: AtomicU64::new(0),
            uploads: AtomicU64::new(0),
            tables_served: AtomicU64::new(0),
            one_shot: AtomicU64::new(0),
            #[cfg(any(test, feature = "test-support"))]
            g1_checks: AtomicU64::new(0),
            simd_ok: OnceLock::new(),
            cpu_pool: OnceLock::new(),
        }
    }
}

fn tables<B: Backend>() -> MutexGuard<'static, Option<Registry<B>>> {
    lock(&B::host().msm.tables)
}

/// The live table registered under `key` with `id`, pinned.
fn pinned<B: Backend>(key: (usize, usize), id: u64) -> Option<Arc<Table<B>>> {
    tables::<B>().as_ref().and_then(|t| t.get(&key)).filter(|e| e.id == id).cloned()
}

/// The registration of an SRS base slice. While it lives, the registrant
/// keeps the slice alive and unchanged. Dropping it frees the table.
pub struct SrsHandle<B: Backend> {
    id: u64,
    key: (usize, usize),
    _backend: PhantomData<fn() -> B>,
}

impl<B: Backend> std::fmt::Debug for SrsHandle<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrsHandle")
            .field("id", &self.id)
            .field("key", &self.key)
            .finish()
    }
}

impl<B: Backend> SrsHandle<B> {
    /// Number of registered points.
    pub fn len(&self) -> usize {
        self.key.1
    }

    /// Whether no point is registered.
    pub fn is_empty(&self) -> bool {
        self.key.1 == 0
    }

    /// Bytes this table holds in device memory: its points once uploaded,
    /// else 0.
    pub fn device_bytes(&self) -> u64 {
        pinned::<B>(self.key, self.id).map_or(0, |t| {
            read(&t.store).device().map_or(0, |b| b.len_bytes() as u64)
        })
    }

    /// Whether `bases` is the registered slice with the registered content.
    pub fn verify(&self, bases: &[G1AffineLimbs]) -> bool {
        (bases.as_ptr() as usize, bases.len()) == self.key
            && pinned::<B>(self.key, self.id).is_some_and(|t| t.serves(bases))
    }
}

impl<B: Backend> Drop for SrsHandle<B> {
    fn drop(&mut self) {
        let removed = {
            let mut t = tables::<B>();
            match t.as_mut() {
                Some(m) if m.get(&self.key).is_some_and(|e| e.id == self.id) => m.remove(&self.key),
                _ => None,
            }
        };
        // outside the registry lock; an MSM still holding the table frees it
        // when it ends
        drop(removed);
    }
}

/// Registers a host copy of an SRS base slice; the first MSM on it uploads
/// it. Registering the same slice again replaces the table. The KZG prover
/// commits from four `ParamsKZG` base vectors, so register all four.
///
/// Every point is checked to be in G1 once here. A table with a point outside
/// G1 is still registered, and MSMs on a prefix that includes it decline.
pub fn register_srs<B: Backend>(bases: &[G1AffineLimbs]) -> SrsHandle<B> {
    let key = (bases.as_ptr() as usize, bases.len());
    let id = B::host().msm.next_id.fetch_add(1, Relaxed);
    let t = Arc::new(Table {
        id,
        len: bases.len(),
        first_bad: first_not_in_g1(bases),
        store: RwLock::new(Store::Host(bases.to_vec())),
        masks: Mutex::new(HashMap::new()),
    });
    let old = tables::<B>().get_or_insert_with(HashMap::new).insert(key, t);
    // outside the registry lock, as in `SrsHandle::drop`
    drop(old);
    SrsHandle {
        id,
        key,
        _backend: PhantomData,
    }
}

/// Live registered tables.
pub fn srs_handles<B: Backend>() -> usize {
    tables::<B>().as_ref().map_or(0, |t| t.len())
}

/// Tables dropped at an MSM because the caller's slice no longer matched.
pub fn srs_stale<B: Backend>() -> u64 {
    B::host().msm.stale.load(Relaxed)
}

/// (MSMs served from a registered table, one-shot MSMs, table uploads) since
/// process start.
pub fn msm_counts<B: Backend>() -> (u64, u64, u64) {
    let m = &B::host().msm;
    (
        m.tables_served.load(Relaxed),
        m.one_shot.load(Relaxed),
        m.uploads.load(Relaxed),
    )
}

/// Per-call G1 checks started on the one-shot path since process start (not
/// a stable interface).
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn one_shot_g1_checks<B: Backend>() -> u64 {
    B::host().msm.g1_checks.load(Relaxed)
}

/// The key of the table that serves `m` scalars on the slice `key`: the exact
/// entry, else the shortest longer entry at the same address.
fn table_for<B: Backend>(t: &Registry<B>, key: (usize, usize), m: usize) -> Option<(usize, usize)> {
    if m > key.1 {
        return None;
    }
    if t.contains_key(&key) {
        return Some(key);
    }
    t.keys().filter(|k| k.0 == key.0 && k.1 > key.1).min_by_key(|k| k.1).copied()
}

/// The table that serves an MSM of `n` scalars on `bases`, pinned, after every
/// point was compared. A mismatching table is dropped (with its cached mask
/// MSMs) unless it was replaced meanwhile, and the next candidate is tried.
pub(super) fn serving_table<B: Backend>(
    bases: &[G1AffineLimbs],
    n: usize,
) -> Option<Arc<Table<B>>> {
    let key = (bases.as_ptr() as usize, bases.len());
    loop {
        let (tk, t) = {
            let g = tables::<B>();
            let t = g.as_ref()?;
            let tk = table_for(t, key, n)?;
            (tk, t.get(&tk)?.clone())
        };
        if t.serves(bases) {
            return Some(t);
        }
        let removed = {
            let mut g = tables::<B>();
            match g.as_mut() {
                Some(m) if m.get(&tk).is_some_and(|e| Arc::ptr_eq(e, &t)) => m.remove(&tk),
                _ => None,
            }
        };
        if removed.is_some() {
            B::host().msm.stale.fetch_add(1, Relaxed);
        }
        drop(removed);
        drop(t);
    }
}
