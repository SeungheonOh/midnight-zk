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

//! Registered base tables ("Registered base tables" in the module text) and
//! the MSM counters.

use std::{
    collections::HashMap,
    sync::{
        Arc, Mutex, MutexGuard, RwLock,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

use super::{
    checks::{first_not_in_g1, points_equal},
    par::{lock, read, write},
};
use crate::{
    G1AffineLimbs, G1JacobianLimbs,
    ctx::{Buffer, Ctx, Error},
};

/// Where a table's points are: the host copy until the first MSM, then the
/// device buffer.
pub(super) enum Store {
    Host(Vec<G1AffineLimbs>),
    Device(Buffer),
}

impl Store {
    /// The first `len` points (fewer if the store holds fewer).
    fn points(&self, len: usize) -> &[G1AffineLimbs] {
        let all = match self {
            Store::Host(h) => h.as_slice(),
            Store::Device(b) => b.as_slice::<G1AffineLimbs>(),
        };
        &all[..len.min(all.len())]
    }
}

/// `MSM(r)` of a table by length (public: the mask and the points are
/// public).
pub(super) type MaskCache = Mutex<HashMap<usize, G1JacobianLimbs>>;

/// A registered table. Its points never change after registration.
pub(super) struct Table {
    id: u64,
    len: usize,
    /// The first index whose point is not in G1 (off the curve, or on it
    /// outside the prime-order subgroup); `len` when every point is. An MSM
    /// on a prefix longer than this declines.
    pub(super) first_bad: usize,
    pub(super) store: RwLock<Store>,
    pub(super) masks: MaskCache,
}

impl Table {
    /// Whether the table still has the caller's points (it is the slice or a
    /// longer one): every point compared.
    fn serves(&self, bases: &[G1AffineLimbs]) -> bool {
        if bases.len() > self.len {
            return false;
        }
        let st = read(&self.store);
        let t = st.points(self.len);
        t.len() >= bases.len() && points_equal(&t[..bases.len()], bases)
    }

    /// Copies the host copy into a device buffer, once (the caller holds the
    /// device state, see the lock order).
    pub(super) fn upload(&self, ctx: &Ctx) -> Result<(), Error> {
        if matches!(*read(&self.store), Store::Host(_)) {
            let mut w = write(&self.store);
            if let Store::Host(h) = &*w {
                let b = ctx.upload(h)?;
                *w = Store::Device(b);
                UPLOADS.fetch_add(1, Relaxed);
            }
        }
        Ok(())
    }
}

/// The registry: tables keyed by (address, length) of the registered slice.
type Registry = HashMap<(usize, usize), Arc<Table>>;

static TABLES: Mutex<Option<Registry>> = Mutex::new(None);
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Tables dropped at an MSM because the caller's slice no longer matched.
static STALE: AtomicU64 = AtomicU64::new(0);
/// Table uploads.
static UPLOADS: AtomicU64 = AtomicU64::new(0);
/// MSMs served from a registered table.
pub(super) static TABLES_SERVED: AtomicU64 = AtomicU64::new(0);
/// One-shot MSMs (past their G1 check).
pub(super) static ONE_SHOT: AtomicU64 = AtomicU64::new(0);
/// Per-call G1 checks of the one-shot path started.
#[cfg(any(test, feature = "test-support"))]
pub(super) static G1_CHECKS: AtomicU64 = AtomicU64::new(0);

fn tables() -> MutexGuard<'static, Option<Registry>> {
    lock(&TABLES)
}

/// The live table registered under `key` with `id`, pinned (the registry
/// lock is released on return).
fn pinned(key: (usize, usize), id: u64) -> Option<Arc<Table>> {
    tables().as_ref().and_then(|t| t.get(&key)).filter(|e| e.id == id).cloned()
}

/// The registration of an SRS base slice. While it lives, the registrant
/// promises to keep the slice alive and unchanged; MSMs on it or on a prefix
/// of it use a device table. Dropping it frees the table.
#[derive(Debug)]
pub struct SrsHandle {
    id: u64,
    key: (usize, usize),
}

impl SrsHandle {
    /// Number of registered points.
    pub fn len(&self) -> usize {
        self.key.1
    }

    /// Whether no point is registered.
    pub fn is_empty(&self) -> bool {
        self.key.1 == 0
    }

    /// Bytes this table holds in device memory now: its points once the
    /// first MSM on it uploaded them, 0 before that and after the table was
    /// dropped as stale. The host copy of a registration that was not
    /// uploaded yet is not counted.
    pub fn device_bytes(&self) -> u64 {
        pinned(self.key, self.id).map_or(0, |t| match &*read(&t.store) {
            Store::Device(b) => b.len_bytes() as u64,
            Store::Host(_) => 0,
        })
    }

    /// Whether every registered point is in G1 (on the curve and in the
    /// prime-order subgroup), as the check at registration found; `false`
    /// also when the table was dropped as stale. Lets a registrant that read
    /// its points without a check rely on this one instead of checking them
    /// twice.
    pub fn in_g1(&self) -> bool {
        pinned(self.key, self.id).is_some_and(|t| t.first_bad >= t.len)
    }

    /// Whether `bases` is the registered slice with the registered content
    /// (every point compared).
    pub fn verify(&self, bases: &[G1AffineLimbs]) -> bool {
        (bases.as_ptr() as usize, bases.len()) == self.key
            && pinned(self.key, self.id).is_some_and(|t| t.serves(bases))
    }
}

impl Drop for SrsHandle {
    fn drop(&mut self) {
        let removed = {
            let mut t = tables();
            match t.as_mut() {
                Some(m) if m.get(&self.key).is_some_and(|e| e.id == self.id) => m.remove(&self.key),
                _ => None,
            }
        };
        // outside the registry lock (an MSM may still hold the table: it is
        // freed when that MSM ends)
        drop(removed);
    }
}

/// Registers a host copy of an SRS base slice (no device work: the first MSM
/// on the slice uploads it). Registering the same slice again replaces the
/// table.
///
/// Every point is checked to be in G1 (see "Bases outside G1" in the module
/// text): a table with a point that is not is still registered, and MSMs on
/// a prefix that includes that point decline.
pub fn register_srs(bases: &[G1AffineLimbs]) -> SrsHandle {
    let key = (bases.as_ptr() as usize, bases.len());
    let id = NEXT_ID.fetch_add(1, Relaxed);
    let t = Arc::new(Table {
        id,
        len: bases.len(),
        first_bad: first_not_in_g1(bases),
        store: RwLock::new(Store::Host(bases.to_vec())),
        masks: Mutex::new(HashMap::new()),
    });
    let old = tables().get_or_insert_with(HashMap::new).insert(key, t);
    // outside the registry lock, as in `SrsHandle::drop`
    drop(old);
    SrsHandle { id, key }
}

/// Live registered tables.
pub fn srs_handles() -> usize {
    tables().as_ref().map_or(0, |t| t.len())
}

/// Tables dropped at an MSM because the caller's slice no longer matched.
pub fn srs_stale() -> u64 {
    STALE.load(Relaxed)
}

/// (MSMs served from a registered table, one-shot MSMs, table uploads) since
/// process start.
pub fn msm_counts() -> (u64, u64, u64) {
    (
        TABLES_SERVED.load(Relaxed),
        ONE_SHOT.load(Relaxed),
        UPLOADS.load(Relaxed),
    )
}

/// Per-call G1 checks of the bases started on the one-shot path since
/// process start (feature `test-support`; not a stable interface). The check
/// of [`register_srs`] is not counted.
#[doc(hidden)]
#[cfg(any(test, feature = "test-support"))]
pub fn one_shot_g1_checks() -> u64 {
    G1_CHECKS.load(Relaxed)
}

/// The key of the table that serves an MSM of `m` scalars on the slice
/// `key`: the exact entry, else the shortest entry at the same address that
/// is longer.
fn table_for(t: &Registry, key: (usize, usize), m: usize) -> Option<(usize, usize)> {
    if m > key.1 {
        return None;
    }
    if t.contains_key(&key) {
        return Some(key);
    }
    t.keys().filter(|k| k.0 == key.0 && k.1 > key.1).min_by_key(|k| k.1).copied()
}

/// The table that serves an MSM of `n` scalars on `bases`, pinned, after
/// every point of `bases` was compared with it. A table that does not match
/// is dropped from the registry (with its cached mask MSMs), unless it was
/// replaced or dropped meanwhile, and the next candidate is tried. The
/// registry lock is held only for the lookups and the removal.
pub(super) fn serving_table(bases: &[G1AffineLimbs], n: usize) -> Option<Arc<Table>> {
    let key = (bases.as_ptr() as usize, bases.len());
    loop {
        let (tk, t) = {
            let g = tables();
            let t = g.as_ref()?;
            let tk = table_for(t, key, n)?;
            (tk, t.get(&tk)?.clone())
        };
        if t.serves(bases) {
            return Some(t);
        }
        let removed = {
            let mut g = tables();
            match g.as_mut() {
                Some(m) if m.get(&tk).is_some_and(|e| Arc::ptr_eq(e, &t)) => m.remove(&tk),
                _ => None,
            }
        };
        if removed.is_some() {
            STALE.fetch_add(1, Relaxed);
        }
        drop(removed);
        drop(t);
    }
}
