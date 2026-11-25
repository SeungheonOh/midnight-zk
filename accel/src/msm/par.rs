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

//! Host threads and locks that never panic out.

use std::{
    panic::AssertUnwindSafe,
    sync::{Mutex, MutexGuard, OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard},
};

use rayon::prelude::*;

use crate::Backend;

/// The threads of a host pass over the points or the scalars.
pub fn host_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |t| t.get()).min(16)
}

/// Persistent thread pool for short host passes (`par_parts`, `all_reduced`,
/// `par_copy`, `collect`), avoiding per-call OS thread creation overhead while
/// staying separate from `cpu_pool` (which runs `cpu_msm`).
pub fn host_pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(host_threads())
            .thread_name(|i| format!("midnight-accel-host-{i}"))
            .build()
            .ok()
    })
    .as_ref()
}

/// The thread pool of the CPU share of an MSM: its own threads, so that the
/// share never waits for a worker of the caller's pool (the prover calls MSMs
/// from rayon workers, which may all be blocked on the device lock).
pub fn cpu_pool<B: Backend>() -> Option<&'static rayon::ThreadPool> {
    B::host()
        .msm
        .cpu_pool
        .get_or_init(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(std::thread::available_parallelism().map_or(4, |t| t.get()))
                .thread_name(|i| format!("midnight-{}-msm-cpu-{i}", B::NAME))
                .build()
                .ok()
        })
        .as_ref()
}

/// `f` with a panic caught: `None` when it panicked.
pub fn catch<R>(f: impl FnOnce() -> R) -> Option<R> {
    std::panic::catch_unwind(AssertUnwindSafe(f)).ok()
}

/// `f(a, b)` over parts `[a, b)` of `[0, len)` (at most `parts` parts of at
/// least `min` items): the first on the calling thread, the others on scoped
/// threads, or on the calling thread when a thread cannot be started.
/// `None`: a part panicked. Never panics.
pub fn par_parts<R: Send>(
    len: usize,
    parts: usize,
    min: usize,
    f: impl Fn(usize, usize) -> R + Sync,
) -> Option<Vec<R>> {
    let per = len.div_ceil(parts.max(1)).max(min.max(1));
    let bounds: Vec<(usize, usize)> =
        (0..len).step_by(per).map(|a| (a, (a + per).min(len))).collect();
    if bounds.len() <= 1 {
        return catch(|| bounds.iter().map(|&(a, b)| f(a, b)).collect());
    }
    if let Some(pool) = host_pool() {
        let res: Vec<Option<R>> =
            pool.install(|| bounds.par_iter().map(|&(a, b)| catch(|| f(a, b))).collect());
        return res.into_iter().collect();
    }
    // fallback if the pool could not be created: scoped threads
    let f = &f;
    std::thread::scope(|s| {
        let started: Vec<_> = bounds[1..]
            .iter()
            .map(|&(a, b)| {
                let h = std::thread::Builder::new().spawn_scoped(s, move || f(a, b)).ok();
                (a, b, h)
            })
            .collect();
        let mut out = Vec::with_capacity(bounds.len());
        let mut ok = true;
        match catch(|| f(bounds[0].0, bounds[0].1)) {
            Some(r) => out.push(r),
            None => ok = false,
        }
        for (a, b, h) in started {
            let r = match h {
                Some(h) => h.join().ok(),
                None if ok => catch(|| f(a, b)),
                None => None,
            };
            match r {
                Some(r) => out.push(r),
                None => ok = false,
            }
        }
        ok.then_some(out)
    })
}

/// A read lock, poisoned or not: data under these locks is only ever replaced
/// whole, so it is valid after any panic.
pub fn read<T>(l: &RwLock<T>) -> RwLockReadGuard<'_, T> {
    l.read().unwrap_or_else(|e| e.into_inner())
}

/// A write lock, poisoned or not (see [`read`]).
pub fn write<T>(l: &RwLock<T>) -> RwLockWriteGuard<'_, T> {
    l.write().unwrap_or_else(|e| e.into_inner())
}

/// A mutex, poisoned or not (see [`read`]).
pub fn lock<T>(l: &Mutex<T>) -> MutexGuard<'_, T> {
    l.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_parts_covers_and_reports_panics() {
        let got = par_parts(1000, 7, 10, |a, b| (a, b)).unwrap();
        assert_eq!(got.first().map(|p| p.0), Some(0));
        assert_eq!(got.last().map(|p| p.1), Some(1000));
        assert!(got.windows(2).all(|w| w[0].1 == w[1].0));
        assert_eq!(par_parts(0, 4, 1, |a, b| (a, b)).unwrap(), vec![]);
        // a panic in a part (on the calling thread or another) is `None`
        assert!(par_parts(1000, 4, 1, |a, _| assert!(a != 0)).is_none());
        assert!(par_parts(1000, 4, 1, |a, _| assert!(a != 750)).is_none());
    }
}
