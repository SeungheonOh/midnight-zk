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

//! The device context of a backend, and the buffers the shared code hands
//! out.
//!
//! ```ignore
//! let mut s = ctx::lock::<B>()?;              // serialises device work
//! let a = s.ctx().upload(scalars)?;           // a scoped, zeroed-on-drop buffer
//! let tw = s.scratch("ntt-twiddles", bytes)?; // a per-request buffer
//! let ns = s.ctx().run(|c| {
//!     c.dispatch("ntt_radix2", n / 2, &[Arg::Buffer(&a)], &(n as u32))?;
//!     Ok(())
//! })?.1;
//! a.read(out);
//! ```
//!
//! [`lock`] returns a [`Session`], which also guards the per-request buffers,
//! so [`end_request`] never zeroes a buffer a kernel is using. [`Ctx::run`]
//! encodes dispatches into one command buffer and waits for it.
//!
//! Buffers:
//! - [`Buffer`]: owned by the caller, zeroed when dropped if it may hold a
//!   non-zero byte.
//! - Pooled ([`Ctx::alloc_pooled`]): taken from and returned to a pool of
//!   zeroed buffers, because a new buffer delays its first command buffer
//!   while the device maps it.
//! - Per-request ([`Session::scratch`], [`take_request_buffer`]): kept under a
//!   key until [`end_request_report`] zeroes them, or zeroed at the end of a
//!   hook call made outside a request ([`wipe_unbracketed`]).

use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    mem::ManuallyDrop,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::NonNull,
    sync::{
        Arc, Mutex, MutexGuard, OnceLock,
        atomic::{
            AtomicBool, AtomicU8,
            Ordering::{Relaxed, SeqCst},
        },
    },
    time::Instant,
};

use crate::{
    WipeStats,
    account::{self, HostPhase, Reason},
    backend::{Backend, Encoder, RawBuffer},
};

/// A device failure, mapped to a decline reason by [`Error::reason`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No usable device, or the context could not be made.
    NoDevice(String),
    /// A buffer could not be allocated (or is above the largest buffer).
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
// SAFETY: no bytes at all.
unsafe impl Pod for () {}
// SAFETY: integers and arrays of them have no padding and no invalid values.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}
// SAFETY: as above.
unsafe impl Pod for u64 {}
// SAFETY: as above.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// The bytes of a plain value.
pub fn bytes_of<T: Pod>(v: &T) -> &[u8] {
    // SAFETY: a Pod value is valid to view as its bytes.
    unsafe { std::slice::from_raw_parts((v as *const T).cast(), size_of::<T>()) }
}

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

// ---------------------------------------------------------------------------
// Context

/// The context of backend `B`: the backend's device with the shared buffer
/// management on top. Dereferences to the backend.
#[repr(transparent)]
pub struct Ctx<B: Backend>(B);

impl<B: Backend> Ctx<B> {
    /// The context over the backend's device.
    pub fn of(dev: &'static B) -> &'static Ctx<B> {
        // SAFETY: `Ctx<B>` is a transparent wrapper of `B`.
        unsafe { &*(dev as *const B as *const Ctx<B>) }
    }
}

impl<B: Backend> std::ops::Deref for Ctx<B> {
    type Target = B;
    fn deref(&self) -> &B {
        &self.0
    }
}

impl<B: Backend> std::fmt::Debug for Ctx<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ctx").field("backend", &B::NAME).finish_non_exhaustive()
    }
}

/// The context of `B`, made at the first call.
pub fn context<B: Backend>() -> Result<&'static Ctx<B>, Error> {
    B::context().map(Ctx::of)
}

// ---------------------------------------------------------------------------
// Byte accounting

/// Adds `bytes` to the bytes in use and raises their peak.
fn in_use_add<B: Backend>(bytes: u64) {
    let h = B::host();
    let now = h.in_use.fetch_add(bytes, Relaxed) + bytes;
    h.in_use_peak.fetch_max(now, Relaxed);
}

/// Counts a new buffer of `bytes` bytes (held and in use).
fn count_alloc<B: Backend>(bytes: u64) {
    let h = B::host();
    let now = h.bytes_now.fetch_add(bytes, Relaxed) + bytes;
    h.bytes_peak.fetch_max(now, Relaxed);
    in_use_add::<B>(bytes);
}

/// `(now, peak)` bytes held in the backend's buffers, idle pool buffers
/// included. The peak is since process start or the last [`begin_request`].
pub fn device_bytes<B: Backend>() -> (u64, u64) {
    let h = B::host();
    (h.bytes_now.load(Relaxed), h.bytes_peak.load(Relaxed))
}

/// Bytes in buffers in use now: [`device_bytes`] without idle pool buffers.
pub fn in_use_bytes<B: Backend>() -> u64 {
    B::host().in_use.load(Relaxed)
}

/// The peak of [`in_use_bytes`] since the last [`begin_request`], which resets
/// it to the bytes in use then.
pub fn request_peak_bytes<B: Backend>() -> u64 {
    B::host().in_use_peak.load(Relaxed)
}

/// Opens a request and resets the peaks of [`device_bytes`] and
/// [`request_peak_bytes`].
pub fn begin_request<B: Backend>() {
    let h = B::host();
    h.bytes_peak.store(h.bytes_now.load(Relaxed), Relaxed);
    h.in_use_peak.store(h.in_use.load(Relaxed), Relaxed);
    h.request_open.store(true, SeqCst);
}

/// Whether a request is open ([`begin_request`] was called and its
/// [`end_request_report`] has not started).
pub fn request_open<B: Backend>() -> bool {
    B::host().request_open.load(SeqCst)
}

// ---------------------------------------------------------------------------
// The pool of zeroed buffers

/// Zeroed buffers kept for reuse ([`Ctx::alloc_pooled`]).
///
/// Invariant: every buffer in the pool reads zero. A buffer enters only
/// known zero and never written, or after its whole allocation was zeroed
/// ([`pool_put`]), so a buffer that held a witness value is never handed out
/// again unzeroed. A buffer whose zeroing panicked or that failed the erasure
/// check is released instead. The pool keeps at most [`pool_limit`] bytes.
pub(crate) struct Pool<R> {
    pub(crate) free: Vec<R>,
    pub(crate) bytes: u64,
}

impl<R> Pool<R> {
    /// An empty pool.
    pub(crate) const fn new() -> Self {
        Pool {
            free: Vec::new(),
            bytes: 0,
        }
    }
}

/// [`pool_limit`] before the environment is read.
pub(crate) const POOL_LIMIT_UNREAD: u64 = u64::MAX;

/// The pool, locked; a poisoned lock is taken over (no update under it is
/// left half done).
fn pool<B: Backend>() -> MutexGuard<'static, Pool<B::Raw>> {
    B::host().pool.lock().unwrap_or_else(|e| e.into_inner())
}

/// The most bytes the pool keeps idle: `{ENV_PREFIX}_POOL_MIB` (read once),
/// else the device's memory budget over [`crate::Tuning::pool_divisor`].
pub fn pool_limit<B: Backend>() -> u64 {
    let h = B::host();
    let l = h.pool_limit.load(Relaxed);
    if l != POOL_LIMIT_UNREAD {
        return l;
    }
    // a huge value saturates below the marker, so it is read once
    let l = std::env::var(format!("{}_POOL_MIB", B::ENV_PREFIX))
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|m| m.saturating_mul(1 << 20))
        .unwrap_or_else(|| {
            B::probe().map_or(0, |i| i.memory_budget / B::TUNING.pool_divisor.max(1))
        })
        .min(POOL_LIMIT_UNREAD - 1);
    let _ = h.pool_limit.compare_exchange(POOL_LIMIT_UNREAD, l, Relaxed, Relaxed);
    h.pool_limit.load(Relaxed)
}

/// Sets [`pool_limit`] in bytes (`None`: the default again) and releases
/// idle buffers above it. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_pool_limit<B: Backend>(limit: Option<u64>) {
    B::host().pool_limit.store(
        limit.map_or(POOL_LIMIT_UNREAD, |l| l.min(POOL_LIMIT_UNREAD - 1)),
        Relaxed,
    );
    let limit = pool_limit::<B>();
    let freed: Vec<_> = {
        let mut p = pool::<B>();
        let mut out = Vec::new();
        while p.bytes > limit {
            let Some(b) = p.free.pop() else { break };
            p.bytes -= b.len() as u64;
            out.push(b);
        }
        out
    };
    release_raw::<B>(freed);
}

/// Bytes of the idle (zeroed) buffers in the pool.
pub fn pool_bytes<B: Backend>() -> u64 {
    pool::<B>().bytes
}

/// Releases idle buffers taken out of the pool (a debug build reads each one
/// back first).
fn release_raw<B: Backend>(bufs: Vec<B::Raw>) -> u64 {
    debug_assert!(
        bufs.iter().all(|b| b.read_back_zero() == ReadBack::Zero),
        "an idle buffer of the pool does not read zero"
    );
    release_unchecked::<B>(bufs)
}

/// Releases buffers taken out of the pool: out of the byte count, after the
/// backend's release hook. Returns their bytes.
fn release_unchecked<B: Backend>(bufs: Vec<B::Raw>) -> u64 {
    let bytes: u64 = bufs.iter().map(|b| b.len() as u64).sum();
    for b in &bufs {
        b.on_release();
    }
    drop(bufs);
    B::host().bytes_now.fetch_sub(bytes, Relaxed);
    bytes
}

/// Releases every idle buffer of the pool once background zeroing has ended.
/// Returns the bytes released.
pub fn release_pool<B: Backend>() -> u64 {
    settle_releases::<B>();
    let bufs = {
        let mut p = pool::<B>();
        p.bytes = 0;
        std::mem::take(&mut p.free)
    };
    let staging = {
        let mut p = B::host().staging.lock().unwrap_or_else(|e| e.into_inner());
        p.bytes = 0;
        std::mem::take(&mut p.free)
    };
    for b in staging {
        b.on_release();
    }
    release_raw::<B>(bufs)
}

/// The smallest pooled buffer of at least `bytes` and at most a quarter
/// more.
fn pool_take<B: Backend>(bytes: usize) -> Option<B::Raw> {
    let mut p = pool::<B>();
    let max = bytes.saturating_add(bytes / 4);
    let i = p
        .free
        .iter()
        .enumerate()
        .filter(|(_, b)| (bytes..=max).contains(&b.len()))
        .min_by_key(|(_, b)| b.len())?
        .0;
    let b = p.free.swap_remove(i);
    p.bytes -= b.len() as u64;
    in_use_add::<B>(b.len() as u64);
    Some(b)
}

/// Releases idle buffers of the pool, the largest first, until at least
/// `need` bytes are freed or the pool is empty. Returns the bytes freed.
fn release_idle<B: Backend>(need: u64) -> u64 {
    let freed: Vec<_> = {
        let mut p = pool::<B>();
        p.free.sort_by_key(|b| b.len());
        let mut out = Vec::new();
        let mut got = 0u64;
        while got < need {
            let Some(b) = p.free.pop() else { break };
            let len = b.len() as u64;
            p.bytes -= len;
            got += len;
            out.push(b);
        }
        out
    };
    release_raw::<B>(freed)
}

/// Puts a zeroed buffer into the pool, or releases it when the pool would
/// exceed its limit. The caller has zeroed the whole allocation (or it is
/// known zero and unwritten): the callers are [`Buffer`]'s drop after
/// [`Buffer::erase`] and the erasure report, neither of which pools a buffer
/// that failed.
fn pool_put<B: Backend>(raw: B::Raw) {
    let len = raw.len() as u64;
    B::host().in_use.fetch_sub(len, Relaxed);
    let limit = pool_limit::<B>();
    let back = {
        let mut p = pool::<B>();
        if p.bytes + len <= limit {
            p.bytes += len;
            p.free.push(raw);
            None
        } else {
            Some(raw)
        }
    };
    if let Some(b) = back {
        release_raw::<B>(vec![b]);
    }
}

/// Background zeroing ([`release_later`]): each call has a ticket, and
/// `outstanding` holds the tickets not yet complete.
pub(crate) struct Releases {
    pub(crate) next: u64,
    pub(crate) outstanding: BTreeSet<u64>,
}

impl Releases {
    /// No release yet.
    pub(crate) const fn new() -> Self {
        Releases {
            next: 0,
            outstanding: BTreeSet::new(),
        }
    }
}

/// The tickets, locked (a poisoned lock is taken over).
fn releases<B: Backend>() -> MutexGuard<'static, Releases> {
    B::host().releases.lock().unwrap_or_else(|e| e.into_inner())
}

/// Drops `bufs` (each zeroed, then pooled or released) on a background
/// thread, or on this one when no thread can be made. [`settle_releases`]
/// waits for it, so at the end of a request every such buffer is zeroed.
pub fn release_later<B: Backend>(bufs: Vec<Buffer<B>>) {
    if bufs.is_empty() {
        return;
    }
    let ticket = {
        let mut r = releases::<B>();
        let t = r.next;
        r.next += 1;
        r.outstanding.insert(t);
        t
    };
    // the buffers stay reachable from here if the thread is not made
    let slot = Arc::new(Mutex::new(Some(bufs)));
    let theirs = Arc::clone(&slot);
    let spawned = std::thread::Builder::new().name(format!("midnight-{}-release", B::NAME)).spawn(
        move || {
            BACKGROUND.set(true);
            release_now(&theirs, ticket)
        },
    );
    if spawned.is_err() {
        release_now(&slot, ticket);
    }
}

thread_local! {
    /// Whether this thread zeroes released buffers in the background.
    static BACKGROUND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The host counter of a zeroing on this thread.
fn zero_phase() -> HostPhase {
    if BACKGROUND.get() {
        HostPhase::ZeroBackground
    } else {
        HostPhase::Zero
    }
}

/// Drops the buffers of a [`release_later`] call and completes its ticket.
/// Never panics.
fn release_now<B: Backend>(slot: &Mutex<Option<Vec<Buffer<B>>>>, ticket: u64) {
    let bufs = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    for b in bufs.into_iter().flatten() {
        // `Buffer::drop` does not unwind (its zeroing catches a panic); a
        // panic from the pool or the backend's release hook after it is
        // counted
        if catch_unwind(AssertUnwindSafe(|| drop(b))).is_err() {
            B::host().erase_faults.fetch_add(1, Relaxed);
        }
    }
    releases::<B>().outstanding.remove(&ticket);
    B::host().released.notify_all();
}

/// Whether a [`release_later`] call has not completed yet.
pub fn releases_pending<B: Backend>() -> bool {
    !releases::<B>().outstanding.is_empty()
}

/// Waits until every [`release_later`] call made before this one has
/// completed.
pub fn settle_releases<B: Backend>() {
    let mut r = releases::<B>();
    let target = r.next;
    while r.outstanding.first().is_some_and(|t| *t < target) {
        r = B::host().released.wait(r).unwrap_or_else(|e| e.into_inner());
    }
}

// ---------------------------------------------------------------------------
// Host passes over memory

/// Byte runs of at least this size are processed on several threads.
const PAR_BYTES: usize = 4 << 20;

/// The threads of a parallel host pass over memory.
fn mem_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |t| t.get()).min(8)
}

/// `f(start, len)` over `[0, len)` in parts of at least [`PAR_BYTES`] (a
/// multiple of 64), on several threads; `true` when every part returned
/// `true`. A part whose thread cannot be made or panicked is redone on the
/// calling thread, so with a non-panicking `f` this covers the whole range.
fn par_ranges(len: usize, f: impl Fn(usize, usize) -> bool + Sync) -> bool {
    if len < 2 * PAR_BYTES {
        return f(0, len);
    }
    let per = len.div_ceil(mem_threads()).max(PAR_BYTES).next_multiple_of(64);
    std::thread::scope(|s| {
        let parts: Vec<_> = (0..len)
            .step_by(per)
            .map(|a| {
                let n = per.min(len - a);
                let f = &f;
                let h = std::thread::Builder::new().spawn_scoped(s, move || f(a, n)).ok();
                (a, n, h)
            })
            .collect();
        let mut all = true;
        for (a, n, h) in parts {
            let ok = match h {
                Some(h) => h.join().unwrap_or_else(|_| f(a, n)),
                None => f(a, n),
            };
            all &= ok;
        }
        all
    })
}

/// Zeroes `len` bytes of host memory at `p` on several threads. Does not
/// panic. The caller has exclusive use of the memory, which is valid for
/// writes of `len` bytes.
pub fn host_zero(p: NonNull<u8>, len: usize) {
    let base = p.as_ptr() as usize;
    par_ranges(len, |a, n| {
        // SAFETY: valid for `len` bytes (the caller's contract), exclusive,
        // and the parts are disjoint.
        unsafe { std::ptr::write_bytes((base + a) as *mut u8, 0, n) };
        true
    });
    std::sync::atomic::compiler_fence(SeqCst);
}

/// Whether `len` bytes of host memory at `p` read zero (volatile word loads).
/// Does not panic. `p` is 8-byte aligned and valid for reads of `len` bytes.
pub fn host_reads_zero(p: NonNull<u8>, len: usize) -> bool {
    let base = p.as_ptr() as usize;
    par_ranges(len, |a, n| {
        let p = (base + a) as *const u8;
        let words = n / 8;
        // SAFETY: valid for `len` bytes (the caller's contract); `a` is a
        // multiple of 64 and `p` 8-byte aligned, so the word loads are
        // aligned; the reads stay inside the part.
        unsafe {
            (0..words).all(|i| std::ptr::read_volatile(p.cast::<u64>().add(i)) == 0)
                && (words * 8..n).all(|i| std::ptr::read_volatile(p.add(i)) == 0)
        }
    })
}

// ---------------------------------------------------------------------------
// Buffers

/// A shadow: the shadow is the device buffer itself, or both hold the same
/// bytes.
const SYNCED: u8 = 0;
/// A shadow: the host wrote it; the next run that binds the buffer copies it
/// in.
const HOST_AHEAD: u8 = 1;
/// A shadow: a run may have written the device buffer since; the shadow is
/// stale until a download.
const DEVICE_AHEAD: u8 = 2;

/// The host copy of a device buffer the host cannot map: a staging buffer of
/// the backend, of the device buffer's allocation length. It holds the same
/// witness data, so it is zeroed with its buffer and pooled like it.
struct Shadow<B: Backend> {
    raw: ManuallyDrop<B::Raw>,
    /// [`SYNCED`], [`HOST_AHEAD`] or [`DEVICE_AHEAD`].
    state: AtomicU8,
    /// Serialises downloads into the shadow.
    sync: Mutex<()>,
}

/// A device buffer, zeroed when dropped.
///
/// The host reads and writes the contents through [`Buffer::host_read`] and
/// [`Buffer::host_write`]. When the backend maps the buffer into host memory
/// these access it in place. Otherwise they access a shadow, a staging buffer
/// made at the first host access: a written shadow is copied in by the next
/// [`Ctx::run`] that binds the buffer, and a buffer marked with
/// [`Command::download`] has its shadow refreshed after the run (a read
/// without that mark downloads first, in a run of its own).
pub struct Buffer<B: Backend> {
    raw: ManuallyDrop<B::Raw>,
    bytes: usize,
    /// Goes back to the pool when dropped; cleared when its zeroing failed or
    /// it failed the erasure check, so it is released instead.
    pooled: bool,
    /// May hold a non-zero byte (the buffer or its shadow): written by the
    /// host or bound to a kernel since last zeroed.
    dirty: AtomicBool,
    /// The host copy; never made for a buffer with a host mapping.
    shadow: OnceLock<Shadow<B>>,
}

impl<B: Backend> std::fmt::Debug for Buffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("bytes", &self.bytes).finish()
    }
}

impl<B: Backend> Buffer<B> {
    /// A buffer over `raw`, known zero.
    fn new(raw: B::Raw, bytes: usize, pooled: bool) -> Self {
        Buffer {
            raw: ManuallyDrop::new(raw),
            bytes,
            pooled,
            dirty: AtomicBool::new(false),
            shadow: OnceLock::new(),
        }
    }

    /// The length in bytes.
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// The backend's buffer.
    pub fn raw(&self) -> &B::Raw {
        &self.raw
    }

    /// The identifier of the allocation ([`RawBuffer::id`]).
    pub fn id(&self) -> u64 {
        self.raw.id()
    }

    /// Records that the buffer may now hold a non-zero byte.
    pub(crate) fn touch(&self) {
        self.dirty.store(true, Relaxed);
    }

    /// The shadow, made (zeroed, from the staging pool) at the first call. A
    /// buffer that may hold data starts with a stale shadow.
    fn shadow(&self) -> Result<&Shadow<B>, Error> {
        if let Some(sh) = self.shadow.get() {
            return Ok(sh);
        }
        let raw = staging_take::<B>(self.raw.len())?;
        let state = if self.dirty.load(Relaxed) {
            DEVICE_AHEAD
        } else {
            SYNCED
        };
        let made = Shadow {
            raw: ManuallyDrop::new(raw),
            state: AtomicU8::new(state),
            sync: Mutex::new(()),
        };
        if let Err(lost) = self.shadow.set(made) {
            staging_put::<B>(ManuallyDrop::into_inner(lost.raw));
        }
        Ok(self.shadow.get().expect("the shadow was just set"))
    }

    /// [`Buffer::host_view`] for writing: marks the buffer written and its
    /// shadow ahead of the device.
    fn host_view_mut(&mut self) -> Result<NonNull<u8>, Error> {
        let p = self.host_view()?;
        self.touch();
        if let Some(sh) = self.shadow.get() {
            sh.state.store(HOST_AHEAD, SeqCst);
        }
        Ok(p)
    }

    /// The host address of the contents, current with the device: the
    /// mapping, else the shadow after a download when it is stale.
    fn host_view(&self) -> Result<NonNull<u8>, Error> {
        if let Some(p) = self.raw.host_ptr() {
            return Ok(p);
        }
        let sh = self.shadow()?;
        if sh.state.load(SeqCst) == DEVICE_AHEAD {
            let _sync = sh.sync.lock().unwrap_or_else(|e| e.into_inner());
            if sh.state.load(SeqCst) == DEVICE_AHEAD {
                context::<B>()?.run(|cmd| cmd.download(self))?;
            }
        }
        sh.raw
            .host_ptr()
            .ok_or_else(|| Error::Device("a staging buffer without a host mapping".into()))
    }

    /// `f` over the contents as `T`s (the shadow on a device the host does
    /// not map; a stale shadow is downloaded first).
    #[inline]
    pub fn host_read<T: Pod, R>(&self, f: impl FnOnce(&[T]) -> R) -> Result<R, Error> {
        let p = self.host_view()?;
        // SAFETY: the view is aligned for any `Pod`, valid for `bytes` bytes,
        // and no GPU work writes it while `&self` is held outside `Ctx::run`.
        let s =
            unsafe { std::slice::from_raw_parts(p.as_ptr().cast(), self.bytes / size_of::<T>()) };
        let t = Instant::now();
        let r = f(s);
        account::add_host_since::<B>(HostPhase::Copy, t);
        Ok(r)
    }

    /// `f` over the contents as mutable `T`s; the next run that binds the
    /// buffer sees the writes (on a device the host does not map, the shadow
    /// is copied in then).
    #[inline]
    pub fn host_write<T: Pod, R>(&mut self, f: impl FnOnce(&mut [T]) -> R) -> Result<R, Error> {
        let p = self.host_view_mut()?;
        // SAFETY: as `host_read`, and `&mut self` is exclusive.
        let s = unsafe {
            std::slice::from_raw_parts_mut(p.as_ptr().cast(), self.bytes / size_of::<T>())
        };
        let t = Instant::now();
        let r = f(s);
        account::add_host_since::<B>(HostPhase::Copy, t);
        Ok(r)
    }

    /// The contents as a slice of `T` ([`Buffer::host_read`]). Panics when the
    /// host view cannot be made.
    pub fn as_slice<T: Pod>(&self) -> &[T] {
        let p = self.host_view().expect("a host view of a device buffer");
        // SAFETY: as `host_read`.
        unsafe { std::slice::from_raw_parts(p.as_ptr().cast(), self.bytes / size_of::<T>()) }
    }

    /// The contents as a mutable slice of `T` ([`Buffer::host_write`]).
    /// Panics when the host view cannot be made.
    pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
        let p = self.host_view_mut().expect("a host view of a device buffer");
        // SAFETY: as `host_write`; the slice borrows `self` mutably.
        unsafe { std::slice::from_raw_parts_mut(p.as_ptr().cast(), self.bytes / size_of::<T>()) }
    }

    /// Copies `data` to the start of the buffer. Panics if it does not fit.
    pub fn write<T: Pod>(&mut self, data: &[T]) {
        let to = &mut self.as_mut_slice::<T>()[..data.len()];
        let t = Instant::now();
        to.copy_from_slice(data);
        account::add_host_since::<B>(HostPhase::Copy, t);
    }

    /// Copies the start of the buffer into `out`. Panics if it is shorter.
    pub fn read<T: Pod>(&self, out: &mut [T]) {
        let from = &self.as_slice::<T>()[..out.len()];
        let t = Instant::now();
        out.copy_from_slice(from);
        account::add_host_since::<B>(HostPhase::Copy, t);
    }

    /// `f` over the whole allocation's bytes, which may exceed
    /// [`Buffer::len_bytes`]. Test support.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn host_read_alloc<R>(&self, f: impl FnOnce(&[u8]) -> R) -> Result<R, Error> {
        let p = self.host_view()?;
        // SAFETY: the view is valid for the allocation's length.
        Ok(f(unsafe {
            std::slice::from_raw_parts(p.as_ptr(), self.raw.len())
        }))
    }

    /// Zeroes the whole allocation (which may exceed [`Buffer::len_bytes`])
    /// and its shadow. Does not panic. A zeroing the backend reports failed is
    /// counted as an erasure fault and the buffer is released instead of
    /// pooled.
    pub fn zero(&mut self) {
        let t = Instant::now();
        // exclusive: `&mut self`
        let mut ok = self.raw.zero_all().is_ok();
        if let Some(sh) = self.shadow.get() {
            ok &= sh.raw.zero_all().is_ok();
            sh.state.store(SYNCED, SeqCst);
        }
        account::add_host_since::<B>(zero_phase(), t);
        if ok {
            self.dirty.store(false, Relaxed);
        } else {
            self.pooled = false;
            B::host().erase_faults.fetch_add(1, Relaxed);
        }
    }

    /// Zeroes the buffer if it may hold a non-zero byte. If the zeroing
    /// panics, it is redone on this thread without threads or allocation, the
    /// buffer is marked not to be pooled, and the fault is counted for
    /// [`end_request_report`].
    fn erase(&mut self) {
        if !self.dirty.load(Relaxed) {
            return;
        }
        let id = self.id();
        let r = catch_unwind(AssertUnwindSafe(|| {
            test_switch::zeroing_panic::<B>(id);
            self.zero()
        }));
        if r.is_err() {
            let shadow = self.shadow.get().map(|sh| &*sh.raw);
            for raw in std::iter::once(&*self.raw).chain(shadow) {
                match raw.host_ptr() {
                    // SAFETY: valid for `len()` bytes, exclusive (`&mut self`).
                    Some(p) => unsafe { std::ptr::write_bytes(p.as_ptr(), 0, raw.len()) },
                    None => {
                        let _ = raw.zero_all();
                    }
                }
            }
            std::sync::atomic::compiler_fence(SeqCst);
            self.dirty.store(false, Relaxed);
            self.pooled = false;
            B::host().erase_faults.fetch_add(1, Relaxed);
        }
    }

    /// The bytes [`Buffer::zero`] writes: the whole allocation.
    pub fn alloc_bytes(&self) -> usize {
        self.raw.len()
    }

    /// Reads back the buffer and its shadow: zero only when both are.
    fn read_back_zero(&self) -> ReadBack {
        let r = self.raw.read_back_zero();
        match self.shadow.get().map(|sh| sh.raw.read_back_zero()) {
            Some(ReadBack::Zero) | None => r,
            Some(ReadBack::NonZero) if r == ReadBack::Zero => ReadBack::NonZero,
            Some(ReadBack::Unreadable) => ReadBack::Unreadable,
            Some(_) => r,
        }
    }
}

impl<B: Backend> Drop for Buffer<B> {
    fn drop(&mut self) {
        self.erase();
        if let Some(sh) = self.shadow.take() {
            let raw = ManuallyDrop::into_inner(sh.raw);
            if self.pooled {
                staging_put::<B>(raw);
            } else {
                raw.on_release();
            }
        }
        // SAFETY: taken once, here; `self.raw` is not used afterwards.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        if self.pooled {
            pool_put::<B>(raw);
        } else {
            let len = raw.len() as u64;
            raw.on_release();
            drop(raw);
            let h = B::host();
            h.in_use.fetch_sub(len, Relaxed);
            h.bytes_now.fetch_sub(len, Relaxed);
        }
    }
}

/// A zeroed staging buffer of at least `len` bytes: from the staging pool,
/// else new from the backend.
fn staging_take<B: Backend>(len: usize) -> Result<B::Raw, Error> {
    let took = {
        let mut p = B::host().staging.lock().unwrap_or_else(|e| e.into_inner());
        let max = len.saturating_add(len / 4);
        let found = p
            .free
            .iter()
            .enumerate()
            .filter(|(_, b)| (len..=max).contains(&b.len()))
            .min_by_key(|(_, b)| b.len())
            .map(|(i, _)| i);
        found.map(|i| {
            let b = p.free.swap_remove(i);
            p.bytes -= b.len() as u64;
            b
        })
    };
    if let Some(b) = took {
        return Ok(b);
    }
    let (raw, zeroed) = context::<B>()?
        .alloc_staging(len)
        .ok_or_else(|| Error::Memory(format!("staging buffer of {len} bytes")))?;
    if !zeroed {
        let t = Instant::now();
        let r = raw.zero_all();
        account::add_host_since::<B>(zero_phase(), t);
        if let Err(e) = r {
            raw.on_release();
            return Err(e);
        }
    }
    Ok(raw)
}

/// Puts a zeroed staging buffer into the staging pool, or releases it above
/// [`pool_limit`].
fn staging_put<B: Backend>(raw: B::Raw) {
    let len = raw.len() as u64;
    let limit = pool_limit::<B>();
    let back = {
        let mut p = B::host().staging.lock().unwrap_or_else(|e| e.into_inner());
        if p.bytes + len <= limit {
            p.bytes += len;
            p.free.push(raw);
            None
        } else {
            Some(raw)
        }
    };
    if let Some(b) = back {
        b.on_release();
    }
}

impl<B: Backend> Ctx<B> {
    /// A new [`Buffer`] over a fresh allocation, counted; a buffer not known
    /// to be zero-filled is zeroed first.
    fn wrap(&self, raw: Option<(B::Raw, bool)>, bytes: usize) -> Result<Buffer<B>, Error> {
        let (raw, zeroed) = raw.ok_or_else(|| Error::Memory(format!("buffer of {bytes} bytes")))?;
        if !zeroed {
            let t = Instant::now();
            let r = raw.zero_all();
            account::add_host_since::<B>(zero_phase(), t);
            if let Err(e) = r {
                raw.on_release();
                return Err(e);
            }
        }
        count_alloc::<B>(raw.len() as u64);
        Ok(Buffer::new(raw, bytes, false))
    }

    /// The allocation length for `bytes`; `Err` above the device's largest
    /// buffer.
    fn check_len(&self, bytes: usize) -> Result<usize, Error> {
        let max = self.max_buffer_len();
        if bytes as u64 > max {
            return Err(Error::Memory(format!(
                "{bytes} bytes is above the largest buffer, {max}"
            )));
        }
        // no device takes a zero-length buffer
        Ok(bytes.max(16))
    }

    /// A new zero-filled buffer (never from the pool). It starts clean and
    /// may be pooled unwritten.
    ///
    /// The pool never causes a failed allocation: idle buffers are released
    /// first when this would exceed the device's memory budget, and all of
    /// them before one retry when the allocation fails.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer<B>, Error> {
        let len = self.check_len(bytes)?;
        let (used, total) = self.budget();
        let over = (used + len as u64).saturating_sub(total);
        if over > 0 && pool_bytes::<B>() > 0 {
            release_idle::<B>(over);
        }
        let mut raw = self.alloc_raw(len);
        if raw.is_none() && pool_bytes::<B>() > 0 {
            release_idle::<B>(u64::MAX);
            raw = self.alloc_raw(len);
        }
        self.wrap(raw, bytes)
    }

    /// A zero-filled buffer from the pool (at most a quarter longer), else a
    /// new one. When dropped it is zeroed and goes back to the pool, or is
    /// released if that failed.
    pub fn alloc_pooled(&self, bytes: usize) -> Result<Buffer<B>, Error> {
        let len = self.check_len(bytes)?;
        let mut took = pool_take::<B>(len);
        if took.is_none() && releases_pending::<B>() {
            // a buffer being zeroed in the background may fit, and waiting for it
            // is faster than mapping a new buffer
            settle_releases::<B>();
            took = pool_take::<B>(len);
        }
        let mut b = match took {
            Some(raw) => Buffer::new(raw, bytes, true),
            None => self.alloc(bytes)?,
        };
        b.pooled = true;
        Ok(b)
    }

    /// A buffer holding a copy of `data`; bytes past the data read zero.
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<Buffer<B>, Error> {
        let bytes = size_of_val(data);
        let mut b = self.alloc(bytes)?;
        if bytes > 0 {
            b.host_write(|d: &mut [u8]| d[..bytes].copy_from_slice(bytes_of_slice(data)))?;
        }
        Ok(b)
    }

    /// Encodes the dispatches of `f` into one command buffer, submits it and
    /// waits ([`Backend::run`]). Returns `f`'s value and the GPU time in
    /// nanoseconds.
    ///
    /// A buffer the host wrote through its shadow is copied in before its
    /// first dispatch; a buffer marked with [`Command::download`] has its
    /// shadow refreshed after the run.
    pub fn run<'a, R>(
        &'a self,
        f: impl FnOnce(&mut Command<'_, 'a, B>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error> {
        let t = Instant::now();
        let ran = self.0.run(|enc| {
            let mut cmd = Command {
                enc,
                down: Vec::new(),
            };
            let r = f(&mut cmd)?;
            cmd.encode_downloads()?;
            Ok((r, cmd.down))
        });
        account::add_host_since::<B>(HostPhase::Run, t);
        account::add_host::<B>(HostPhase::Runs, 1);
        let ((r, down), ns) = ran?;
        account::add_host::<B>(HostPhase::RunGpu, ns);
        for (_, sh) in down {
            sh.state.store(SYNCED, SeqCst);
        }
        Ok((r, ns))
    }
}

/// The bytes of a slice of plain values.
fn bytes_of_slice<T: Pod>(v: &[T]) -> &[u8] {
    // SAFETY: Pod values are valid to view as their bytes.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), size_of_val(v)) }
}

// ---------------------------------------------------------------------------
// Dispatch

/// A buffer argument of a kernel, bound at the index of its position in the
/// list; the kernel's parameters follow the buffers.
#[derive(Debug)]
pub enum Arg<'a, B: Backend> {
    /// A buffer from its start.
    Buffer(&'a Buffer<B>),
    /// A buffer from a byte offset.
    BufferAt(&'a Buffer<B>, usize),
}

impl<B: Backend> Clone for Arg<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: Backend> Copy for Arg<'_, B> {}

impl<'a, B: Backend> Arg<'a, B> {
    /// The buffer.
    pub fn buffer(&self) -> &'a Buffer<B> {
        match *self {
            Arg::Buffer(b) | Arg::BufferAt(b, _) => b,
        }
    }

    /// The backend's buffer.
    pub fn raw(&self) -> &'a B::Raw {
        &self.buffer().raw
    }

    /// The byte offset of the binding.
    pub fn offset(&self) -> usize {
        match *self {
            Arg::Buffer(_) => 0,
            Arg::BufferAt(_, off) => off,
        }
    }
}

/// The command buffer being encoded in [`Ctx::run`]: each dispatch sees the
/// writes of the earlier ones.
pub struct Command<'c, 'a, B: Backend> {
    enc: &'c mut B::Cmd<'a>,
    /// The buffers whose shadows are refreshed at the end of the run.
    down: Vec<(&'a B::Raw, &'a Shadow<B>)>,
}

impl<B: Backend> std::fmt::Debug for Command<'_, '_, B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Command").finish_non_exhaustive()
    }
}

impl<'a, B: Backend> Command<'_, 'a, B> {
    /// Marks the buffers written, checks their offsets, and copies in the
    /// shadows the host wrote.
    fn bind(&mut self, args: &[Arg<'a, B>]) -> Result<(), Error> {
        for (i, a) in args.iter().enumerate() {
            let b = a.buffer();
            b.touch();
            if a.offset() > b.bytes {
                return Err(Error::Device(format!(
                    "argument {i}: offset {} past the buffer",
                    a.offset()
                )));
            }
            if let Some(sh) = b.shadow.get() {
                if sh.state.load(SeqCst) == HOST_AHEAD {
                    self.enc.copy(&sh.raw, &b.raw, b.raw.len())?;
                }
                sh.state.store(DEVICE_AHEAD, SeqCst);
            }
        }
        Ok(())
    }

    /// Marks `buf` to be read by the host after the run: on a device the host
    /// does not map, its shadow is refreshed at the end of the run. Nothing on
    /// a mapped buffer.
    pub fn download(&mut self, buf: &'a Buffer<B>) -> Result<(), Error> {
        if buf.raw.host_ptr().is_some() {
            return Ok(());
        }
        let sh = buf.shadow()?;
        buf.touch();
        if !self.down.iter().any(|d| std::ptr::eq(d.1, sh)) {
            self.down.push((&buf.raw, sh));
        }
        Ok(())
    }

    /// Encodes the copies of the marked buffers into their shadows. A shadow
    /// the host wrote and no dispatch bound stays as it is.
    fn encode_downloads(&mut self) -> Result<(), Error> {
        let mut i = 0;
        while i < self.down.len() {
            let (raw, sh) = self.down[i];
            if sh.state.load(SeqCst) == HOST_AHEAD {
                self.down.swap_remove(i);
                continue;
            }
            // a shadow from the staging pool may be longer than its buffer
            self.enc.copy(raw, &sh.raw, raw.len())?;
            i += 1;
        }
        Ok(())
    }

    /// `threads` threads in workgroups of up to 256; the last group may be
    /// partial, so kernels check their index.
    pub fn dispatch<P: Pod>(
        &mut self,
        kernel: &str,
        threads: usize,
        args: &[Arg<'a, B>],
        params: &P,
    ) -> Result<(), Error> {
        self.bind(args)?;
        self.enc.dispatch(kernel, threads, args, bytes_of(params))
    }

    /// `threads` threads in groups of `group` threads, with shared memory of
    /// `shared[i]` bytes in block `i`.
    pub fn dispatch_with<P: Pod>(
        &mut self,
        kernel: &str,
        threads: usize,
        group: usize,
        args: &[Arg<'a, B>],
        params: &P,
        shared: &[usize],
    ) -> Result<(), Error> {
        self.bind(args)?;
        self.enc.dispatch_with(kernel, threads, group, args, bytes_of(params), shared)
    }

    /// `groups` whole groups of `group` threads, with shared memory as
    /// [`Command::dispatch_with`].
    pub fn dispatch_groups<P: Pod>(
        &mut self,
        kernel: &str,
        groups: usize,
        group: usize,
        args: &[Arg<'a, B>],
        params: &P,
        shared: &[usize],
    ) -> Result<(), Error> {
        self.bind(args)?;
        self.enc.dispatch_groups(kernel, groups, group, args, bytes_of(params), shared)
    }
}

// ---------------------------------------------------------------------------
// Sessions and per-request buffers

/// The per-request buffers of [`Session::scratch`], by key.
pub(crate) struct State<B: Backend> {
    scratch: HashMap<&'static str, Buffer<B>>,
}

impl<B: Backend> Default for State<B> {
    fn default() -> Self {
        State {
            scratch: HashMap::new(),
        }
    }
}

/// Exclusive use of the device: the context and the per-request buffers.
pub struct Session<B: Backend> {
    ctx: &'static Ctx<B>,
    state: MutexGuard<'static, Option<State<B>>>,
}

impl<B: Backend> std::fmt::Debug for Session<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

/// The device lock, taken over when poisoned.
fn state<B: Backend>() -> MutexGuard<'static, Option<State<B>>> {
    B::host().state.lock().unwrap_or_else(|e| e.into_inner())
}

/// Takes the device, making the context if needed.
pub fn lock<B: Backend>() -> Result<Session<B>, Error> {
    let ctx = context::<B>()?;
    let mut state = state::<B>();
    state.get_or_insert_with(State::default);
    Ok(Session { ctx, state })
}

impl<B: Backend> Session<B> {
    /// The context.
    pub fn ctx(&self) -> &'static Ctx<B> {
        self.ctx
    }

    /// The per-request buffer `key` of at least `bytes`: the existing one
    /// with its contents if long enough, else a zero-filled one (the shorter
    /// one goes back to the pool).
    pub fn scratch(&mut self, key: &'static str, bytes: usize) -> Result<&mut Buffer<B>, Error> {
        let ctx = self.ctx;
        let st = self.state.get_or_insert_with(State::default);
        if st.scratch.get(key).is_some_and(|b| b.bytes < bytes) {
            st.scratch.remove(key);
        }
        Ok(match st.scratch.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let b = e.insert(ctx.alloc_pooled(bytes)?);
                B::host().scratch_any.store(true, SeqCst);
                b
            }
        })
    }

    /// The per-request buffer `key`, if it exists.
    pub fn get_scratch(&self, key: &'static str) -> Option<&Buffer<B>> {
        self.state.as_ref()?.scratch.get(key)
    }

    /// Several per-request buffers at once (for binding them together).
    pub fn scratch_many<const N: usize>(&self, keys: [&'static str; N]) -> Option<[&Buffer<B>; N]> {
        let st = self.state.as_ref()?;
        let found: Vec<&Buffer<B>> =
            keys.iter().map(|k| st.scratch.get(k)).collect::<Option<_>>()?;
        found.try_into().ok()
    }

    /// Bytes held in per-request buffers. Test support.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn scratch_bytes(&self) -> u64 {
        self.state
            .as_ref()
            .map_or(0, |s| s.scratch.values().map(|b| b.raw.len() as u64).sum())
    }
}

/// Per-request buffers given back by [`take_request_buffer`] callers, by key.
pub(crate) struct Lendable<B: Backend> {
    /// The request of the buffers ([`end_request_report`] increments it).
    pub(crate) epoch: u64,
    pub(crate) free: Vec<(&'static str, Buffer<B>)>,
}

impl<B: Backend> Lendable<B> {
    /// No buffer, the first request.
    pub(crate) const fn new() -> Self {
        Lendable {
            epoch: 0,
            free: Vec::new(),
        }
    }
}

/// The idle lendable buffers, locked.
fn lendable<B: Backend>() -> MutexGuard<'static, Lendable<B>> {
    B::host().lendable.lock().unwrap_or_else(|e| e.into_inner())
}

/// A per-request buffer under `key` taken without the device, so a call can
/// fill it on the CPU while another call holds the device. It is the idle
/// buffer an earlier call of the request gave back, with its contents, else a
/// zero-filled one. [`end_request_report`] zeroes the idle ones; one given
/// back after its request ended is zeroed then.
pub fn take_request_buffer<B: Backend>(
    key: &'static str,
    bytes: usize,
) -> Result<RequestBuffer<B>, Error> {
    let (epoch, found) = {
        let mut l = lendable::<B>();
        let epoch = l.epoch;
        let found = l
            .free
            .iter()
            .enumerate()
            .filter(|(_, (k, b))| *k == key && b.bytes >= bytes)
            .min_by_key(|(_, (_, b))| b.bytes)
            .map(|(i, _)| i)
            .map(|i| l.free.swap_remove(i).1);
        (epoch, found)
    };
    let buf = match found {
        Some(b) => b,
        None => context::<B>()?.alloc_pooled(bytes)?,
    };
    Ok(RequestBuffer { key, epoch, buf })
}

/// A per-request buffer checked out by [`take_request_buffer`]. Dropped
/// instead of given back, it is zeroed and pooled.
#[derive(Debug)]
pub struct RequestBuffer<B: Backend> {
    key: &'static str,
    epoch: u64,
    buf: Buffer<B>,
}

impl<B: Backend> RequestBuffer<B> {
    /// The buffer.
    pub fn buf(&self) -> &Buffer<B> {
        &self.buf
    }

    /// The buffer, mutably.
    pub fn buf_mut(&mut self) -> &mut Buffer<B> {
        &mut self.buf
    }

    /// Gives the buffer back to its request; after the request ended it is
    /// zeroed and pooled instead.
    pub fn give_back(self) {
        let RequestBuffer { key, epoch, buf } = self;
        let mut l = lendable::<B>();
        if l.epoch == epoch {
            l.free.push((key, buf));
        } else {
            drop(l);
            drop(buf);
        }
    }
}

// ---------------------------------------------------------------------------
// Test switches

/// The test switches of one backend ([`crate::HostState`]).
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub(crate) struct Switches {
    /// [`set_plant_nonzero`].
    pub(crate) plant_nonzero: AtomicBool,
    /// The id of the buffer of the last planted byte.
    pub(crate) last_plant: std::sync::atomic::AtomicU64,
    /// The id of the buffer whose next zeroing panics (0: none).
    pub(crate) zeroing_panic: std::sync::atomic::AtomicU64,
}

#[cfg(any(test, feature = "test-support"))]
impl Switches {
    /// Every switch off.
    pub(crate) const fn new() -> Self {
        Switches {
            plant_nonzero: AtomicBool::new(false),
            last_plant: std::sync::atomic::AtomicU64::new(0),
            zeroing_panic: std::sync::atomic::AtomicU64::new(0),
        }
    }
}

/// Test switches. In other builds they are no-ops, so no code outside the
/// crates can force an erasure failure.
#[cfg(any(test, feature = "test-support"))]
mod test_switch {
    use std::sync::atomic::Ordering::Relaxed;

    use crate::backend::{Backend, RawBuffer};

    /// After the zeroing of region `i`: plants a non-zero byte into the first
    /// region of a checked wipe when [`super::set_plant_nonzero`] is on.
    pub(super) fn plant_after_zeroing<B: Backend>(check: bool, i: usize, raw: &B::Raw) {
        if check && i == 0 && B::host().switches.plant_nonzero.load(Relaxed) {
            plant_raw::<B>(raw);
        }
    }

    /// Panics once, in the zeroing of the buffer `id`, after
    /// [`super::set_zeroing_panic`].
    pub(super) fn zeroing_panic<B: Backend>(id: u64) {
        let s = &B::host().switches.zeroing_panic;
        if id != 0 && s.compare_exchange(id, 0, Relaxed, Relaxed).is_ok() {
            panic!("midnight-{} test switch: a panic in a zeroing", B::NAME);
        }
    }

    /// Writes one non-zero byte into the middle of `raw` behind the shared
    /// code's back (`dirty` not set, [`RawBuffer::write_test_byte`]), and
    /// records its id. The caller has exclusive use of the buffer.
    pub(super) fn plant_raw<B: Backend>(raw: &B::Raw) {
        if raw.write_test_byte() {
            B::host().switches.last_plant.store(raw.id(), Relaxed);
        }
    }
}

/// The test switches of a normal build: no-ops.
#[cfg(not(any(test, feature = "test-support")))]
mod test_switch {
    use crate::backend::Backend;

    #[inline(always)]
    pub(super) fn plant_after_zeroing<B: Backend>(_: bool, _: usize, _: &B::Raw) {}

    #[inline(always)]
    pub(super) fn zeroing_panic<B: Backend>(_: u64) {}
}

/// Makes every later [`end_request_report`]`(true)` plant a non-zero byte in
/// its first region after zeroing, so the read-back fails. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_plant_nonzero<B: Backend>(on: bool) {
    B::host().switches.plant_nonzero.store(on, Relaxed);
}

/// The id of the buffer of the last planted byte. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn last_planted<B: Backend>() -> u64 {
    B::host().switches.last_plant.load(Relaxed)
}

/// Makes the next zeroing of the dropped buffer `id` panic once; 0 stops it.
/// Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_zeroing_panic<B: Backend>(id: u64) {
    B::host().switches.zeroing_panic.store(id, Relaxed);
}

/// Writes a non-zero byte into an idle pool buffer, breaking the pool's
/// invariant; returns its id. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn plant_in_idle_pool<B: Backend>() -> Option<u64> {
    let p = pool::<B>();
    let b = p.free.first()?;
    test_switch::plant_raw::<B>(b);
    Some(b.id())
}

/// The ids of the idle buffers of the pool and of the staging pool (the
/// regions a checked report reads back besides the per-request buffers).
/// Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn idle_pool_ids<B: Backend>() -> Vec<u64> {
    let mut ids: Vec<u64> = pool::<B>().free.iter().map(|b| b.id()).collect();
    let staging = B::host().staging.lock().unwrap_or_else(|e| e.into_inner());
    ids.extend(staging.free.iter().map(|b| b.id()));
    ids
}

/// Whether every idle pool buffer reads zero. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn idle_pool_reads_zero<B: Backend>() -> bool {
    pool::<B>().free.iter().all(|b| b.read_back_zero() == ReadBack::Zero)
}

/// The per-request buffers now: `(key, allocation bytes, reads zero, id)`.
/// Takes the device. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn request_buffers<B: Backend>() -> Vec<(&'static str, usize, bool, u64)> {
    let row = |k: &'static str, b: &Buffer<B>| {
        (
            k,
            b.alloc_bytes(),
            b.read_back_zero() == ReadBack::Zero,
            b.id(),
        )
    };
    let state = state::<B>();
    let mut out: Vec<_> = state
        .as_ref()
        .map(|st| st.scratch.iter().map(|(k, b)| row(k, b)).collect())
        .unwrap_or_default();
    out.extend(lendable::<B>().free.iter().map(|(k, b)| row(k, b)));
    out
}

// ---------------------------------------------------------------------------
// Request wipes

/// The result of reading a region back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadBack {
    /// Every byte reads zero.
    Zero,
    /// A byte reads non-zero.
    NonZero,
    /// The host cannot read it back.
    Unreadable,
}

/// Reads back a region's whole allocation and counts the result in `w`.
fn read_back<R: RawBuffer>(raw: &R, w: &mut WipeStats) -> ReadBack {
    count_read_back(raw.read_back_zero(), w)
}

/// Counts the read-back `r` of a region in `w`.
fn count_read_back(r: ReadBack, w: &mut WipeStats) -> ReadBack {
    w.checked_regions += 1;
    match r {
        ReadBack::Unreadable => w.check_failures += 1,
        ReadBack::NonZero => w.nonzero_regions += 1,
        ReadBack::Zero => {}
    }
    r
}

/// Check mode: reads back every idle pool buffer. One that reads non-zero is
/// zeroed and read back again; it, and one that cannot be read back, is
/// taken out of the pool and released.
fn check_idle_pool<B: Backend>(w: &mut WipeStats) {
    let bad = check_pool(&mut pool::<B>(), w);
    // not `release_raw`: its debug read-back would fail on a buffer that is
    // not readable
    release_unchecked::<B>(bad);
    let bad = check_pool(
        &mut B::host().staging.lock().unwrap_or_else(|e| e.into_inner()),
        w,
    );
    for b in bad {
        b.on_release();
    }
}

/// Reads back every idle buffer of `p`; returns the ones taken out.
fn check_pool<R: RawBuffer>(p: &mut Pool<R>, w: &mut WipeStats) -> Vec<R> {
    let mut bad = Vec::new();
    let mut i = 0;
    while i < p.free.len() {
        let raw = &p.free[i];
        let len = raw.len() as u64;
        w.regions += 1;
        w.bytes += len;
        w.pool_regions += 1;
        w.pool_bytes += len;
        let fault = match read_back(raw, w) {
            ReadBack::Zero => false,
            ReadBack::Unreadable => true,
            ReadBack::NonZero => {
                // idle under the pool's lock: exclusive
                let _ = raw.zero_all();
                let _again = raw.read_back_zero();
                true
            }
        };
        if fault {
            let b = p.free.swap_remove(i);
            p.bytes -= len;
            bad.push(b);
        } else {
            i += 1;
        }
    }
    bad
}

/// Zeroes every per-request buffer (`state` is the device lock) after
/// background zeroing; with `check`, reads back the idle pool and each region.
/// A region that reads back non-zero or unreadable is zeroed again and
/// released, never pooled or kept; the others go back to the pool.
fn wipe_locked<B: Backend>(state: &mut Option<State<B>>, check: bool, w: &mut WipeStats) {
    settle_releases::<B>();
    if check {
        check_idle_pool::<B>(w);
    }
    let lent: Vec<Buffer<B>> = {
        let mut l = lendable::<B>();
        l.epoch += 1;
        l.free.drain(..).map(|(_, b)| b).collect()
    };
    let scratch: Vec<Buffer<B>> = state
        .as_mut()
        .map(|st| st.scratch.drain().map(|(_, b)| b).collect())
        .unwrap_or_default();
    B::host().scratch_any.store(false, SeqCst);
    for (i, mut b) in scratch.into_iter().chain(lent).enumerate() {
        b.zero();
        w.regions += 1;
        w.bytes += b.alloc_bytes() as u64;
        // exclusive: the region is owned here
        test_switch::plant_after_zeroing::<B>(check, i, &b.raw);
        if check && count_read_back(b.read_back_zero(), w) != ReadBack::Zero {
            // zeroed again unconditionally, read back again, released
            b.zero();
            let _again = b.read_back_zero();
            b.pooled = false;
        }
        // zero (and not dirty): pooled, or released when it failed
        drop(b);
    }
}

/// Zeroes every per-request buffer after any device work and background
/// zeroing, and closes the request.
///
/// With `check`, reads back every idle pool buffer and each per-request
/// buffer after zeroing. A region that reads back non-zero or cannot be read
/// is zeroed again and released; the report is then `Err`, as it is when a
/// zeroing panicked since the last report.
pub fn end_request_report<B: Backend>(check: bool) -> Result<WipeStats, WipeStats> {
    let mut w = WipeStats {
        checked: check,
        ..WipeStats::default()
    };
    let h = B::host();
    let mut state = state::<B>();
    h.request_open.store(false, SeqCst);
    wipe_locked(&mut state, check, &mut w);
    drop(state);
    w.erasure_failures = h.erase_faults.swap(0, Relaxed);
    if w.nonzero_regions == 0 && w.check_failures == 0 && w.erasure_failures == 0 {
        Ok(w)
    } else {
        Err(w)
    }
}

/// [`end_request_report`] without the read-back. `true` when done.
pub fn end_request<B: Backend>() -> bool {
    end_request_report::<B>(false).is_ok()
}

/// The wipe of a hook call made outside a request: as [`end_request`]
/// without closing a request or reading back. It takes the device, so it
/// never zeroes a buffer another call is using; a [`take_request_buffer`]
/// buffer is not idle until given back. With nothing to wipe it does not
/// take the device.
pub fn wipe_unbracketed<B: Backend>() {
    let idle_lent = !lendable::<B>().free.is_empty();
    if !(B::host().scratch_any.load(SeqCst) || idle_lent || releases_pending::<B>()) {
        return;
    }
    let mut state = state::<B>();
    let mut w = WipeStats::default();
    wipe_locked(&mut state, false, &mut w);
}

/// `(free, total)`: `total` is the device's memory budget, `free` is that
/// minus the bytes allocated plus the idle pool buffers, which count as free
/// because an allocation that would not fit releases them first.
pub fn mem_info<B: Backend>() -> Option<(usize, usize)> {
    let ctx = context::<B>().ok()?;
    let (used, total) = ctx.budget();
    let idle = pool_bytes::<B>();
    Some((
        (total + idle).saturating_sub(used).min(total) as usize,
        total as usize,
    ))
}

#[cfg(test)]
mod tests {
    //! The shadows of a backend whose device memory the host does not map,
    //! on a backend that keeps "device" buffers in host memory behind a
    //! missing host pointer.

    use std::{
        alloc::{Layout, alloc_zeroed, dealloc},
        sync::atomic::AtomicU64,
    };

    use super::*;
    use crate::{DeviceInfo, Encoder, HostState, Thresholds, Tuning};

    /// Memory of a [`Mock`] buffer; a device buffer has no host pointer.
    struct MockBuf {
        mem: NonNull<u8>,
        len: usize,
        host: bool,
        id: u64,
    }

    // SAFETY: plain memory, accessed under the shared code's rules.
    unsafe impl Send for MockBuf {}
    // SAFETY: as above.
    unsafe impl Sync for MockBuf {}

    static NEXT: AtomicU64 = AtomicU64::new(1);

    impl MockBuf {
        fn new(len: usize, host: bool) -> Self {
            let layout = Layout::from_size_align(len, 64).unwrap();
            // SAFETY: a non-zero size.
            let mem = NonNull::new(unsafe { alloc_zeroed(layout) }).unwrap();
            MockBuf {
                mem,
                len,
                host,
                id: NEXT.fetch_add(1, Relaxed),
            }
        }

        /// The memory as bytes.
        ///
        /// # Safety
        /// The buffer outlives the slice, and no other reference to the memory
        /// is live.
        unsafe fn bytes<'m>(&self) -> &'m mut [u8] {
            // SAFETY: valid for `len` bytes; exclusive by the caller's contract.
            unsafe { std::slice::from_raw_parts_mut(self.mem.as_ptr(), self.len) }
        }
    }

    impl Drop for MockBuf {
        fn drop(&mut self) {
            // SAFETY: allocated in `new` with this layout.
            unsafe {
                dealloc(
                    self.mem.as_ptr(),
                    Layout::from_size_align(self.len, 64).unwrap(),
                )
            }
        }
    }

    impl RawBuffer for MockBuf {
        fn len(&self) -> usize {
            self.len
        }

        fn id(&self) -> u64 {
            self.id
        }

        fn host_ptr(&self) -> Option<NonNull<u8>> {
            self.host.then_some(self.mem)
        }

        fn zero_all(&self) -> Result<(), Error> {
            // SAFETY: the shared code has exclusive use while it zeroes.
            unsafe { self.bytes() }.fill(0);
            Ok(())
        }

        fn read_back_zero(&self) -> ReadBack {
            // SAFETY: the shared code has exclusive use while it reads back.
            if unsafe { self.bytes() }.iter().all(|b| *b == 0) {
                ReadBack::Zero
            } else {
                ReadBack::NonZero
            }
        }
    }

    /// A backend with discrete memory and one kernel, `double`: every `u64`
    /// of its buffer times two, run at encoding.
    struct Mock;

    struct MockCmd;

    impl<'a> Encoder<'a, Mock> for MockCmd {
        fn dispatch(
            &mut self,
            kernel: &str,
            _: usize,
            bufs: &[Arg<'a, Mock>],
            _: &[u8],
        ) -> Result<(), Error> {
            assert_eq!(kernel, "double");
            // SAFETY: the kernel runs at encoding; nothing else uses the buffer.
            for w in unsafe { bufs[0].raw().bytes() }.chunks_mut(8) {
                let v = u64::from_le_bytes(w.try_into().unwrap());
                w.copy_from_slice(&(2 * v).to_le_bytes());
            }
            Ok(())
        }

        fn dispatch_with(
            &mut self,
            _: &str,
            _: usize,
            _: usize,
            _: &[Arg<'a, Mock>],
            _: &[u8],
            _: &[usize],
        ) -> Result<(), Error> {
            unreachable!()
        }

        fn dispatch_groups(
            &mut self,
            _: &str,
            _: usize,
            _: usize,
            _: &[Arg<'a, Mock>],
            _: &[u8],
            _: &[usize],
        ) -> Result<(), Error> {
            unreachable!()
        }

        fn copy(&mut self, src: &'a MockBuf, dst: &'a MockBuf, bytes: usize) -> Result<(), Error> {
            // SAFETY: two distinct buffers, used by nothing else meanwhile.
            unsafe { dst.bytes()[..bytes].copy_from_slice(&src.bytes()[..bytes]) };
            Ok(())
        }
    }

    static HOST: HostState<Mock> = HostState::new();

    impl Backend for Mock {
        const NAME: &'static str = "mock";
        const ENV_PREFIX: &'static str = "MIDNIGHT_ACCEL_MOCK";
        type Raw = MockBuf;
        type Cmd<'a> = MockCmd;
        const TUNING: Tuning = Tuning {
            thresholds: Thresholds::all(1),
            msm_windows: &[(13, 11, 1)],
            msm_shares: &[(13, 64, 64)],
            msm_chunk_adds: 1 << 25,
            msm_sort_groups: 8,
            msm_ones_groups: 32,
            msm_hot_bucket_limit: 1 << 16,
            msm_sparse_div: 5,
            msm_sparse_fixed: 1 << 12,
            ntt_run_budget: 1 << 30,
            quotient_target_nanos: 1 << 28,
            quotient_threads: 1 << 16,
            pool_divisor: 4,
        };

        fn host() -> &'static HostState<Self> {
            &HOST
        }

        fn probe() -> Option<&'static DeviceInfo> {
            static INFO: OnceLock<DeviceInfo> = OnceLock::new();
            Some(INFO.get_or_init(|| DeviceInfo {
                name: "mock".into(),
                unified_memory: false,
                memory_budget: 1 << 30,
                max_buffer_length: 1 << 30,
                max_threads_per_group: 256,
            }))
        }

        fn context() -> Result<&'static Self, Error> {
            Ok(&Mock)
        }

        fn unified(&self) -> bool {
            false
        }

        fn budget(&self) -> (u64, u64) {
            (0, 1 << 30)
        }

        fn max_buffer_len(&self) -> u64 {
            1 << 30
        }

        fn alloc_raw(&self, len: usize) -> Option<(MockBuf, bool)> {
            Some((MockBuf::new(len, false), true))
        }

        fn alloc_staging(&self, len: usize) -> Option<(MockBuf, bool)> {
            Some((MockBuf::new(len, true), true))
        }

        fn run<'a, R>(
            &'a self,
            f: impl FnOnce(&mut Self::Cmd<'a>) -> Result<R, Error>,
        ) -> Result<(R, u64), Error> {
            f(&mut MockCmd).map(|r| (r, 0))
        }

        fn max_group(&self, _: &str) -> Result<usize, Error> {
            Ok(256)
        }

        fn subgroup_width(&self, _: &str) -> Result<usize, Error> {
            Ok(32)
        }
    }

    /// Writes reach the device at the next run that binds the buffer, reads
    /// see the device's writes (downloaded in the run when marked, else before
    /// the read), and a partial host write keeps the device's other bytes.
    #[test]
    fn shadows_follow_the_device() {
        let c = context::<Mock>().unwrap();
        let data: Vec<u64> = (1..=8).collect();
        let mut b = c.upload(&data).unwrap();
        let double =
            |b: &Buffer<Mock>| c.run(|cmd| cmd.dispatch("double", 8, &[Arg::Buffer(b)], &()));
        c.run(|cmd| {
            cmd.dispatch("double", 8, &[Arg::Buffer(&b)], &())?;
            cmd.download(&b)
        })
        .unwrap();
        let got = |b: &Buffer<Mock>| b.host_read(|v: &[u64]| v.to_vec()).unwrap();
        assert_eq!(got(&b), data.iter().map(|v| 2 * v).collect::<Vec<_>>());
        double(&b).unwrap();
        assert_eq!(got(&b), data.iter().map(|v| 4 * v).collect::<Vec<_>>());
        double(&b).unwrap();
        b.host_write(|v: &mut [u64]| v[0] = 100).unwrap();
        double(&b).unwrap();
        let mut want: Vec<u64> = data.iter().map(|v| 16 * v).collect();
        want[0] = 200;
        assert_eq!(got(&b), want);
        // a marked buffer the host wrote and no dispatch bound keeps the write
        b.host_write(|v: &mut [u64]| v[1] = 7).unwrap();
        c.run(|cmd| cmd.download(&b)).unwrap();
        assert_eq!(got(&b)[1], 7);
    }

    /// A shadow from the staging pool may be longer than its buffer: a
    /// download copies the buffer's bytes only.
    #[test]
    fn a_longer_shadow_downloads_its_buffer() {
        let c = context::<Mock>().unwrap();
        let mut b = c.alloc_pooled(5000).unwrap();
        b.host_write(|v: &mut [u8]| v.fill(1)).unwrap();
        drop(b);
        let data: Vec<u64> = (1..=512).collect();
        let b = c.upload(&data).unwrap();
        c.run(|cmd| {
            cmd.dispatch("double", 512, &[Arg::Buffer(&b)], &())?;
            cmd.download(&b)
        })
        .unwrap();
        let got = b.host_read(|v: &[u64]| v.to_vec()).unwrap();
        assert_eq!(got, data.iter().map(|v| 2 * v).collect::<Vec<_>>());
    }

    /// A buffer and its shadow are zeroed together and both pools stay zero.
    #[test]
    fn shadows_are_zeroed_with_their_buffer() {
        let c = context::<Mock>().unwrap();
        let mut b = c.alloc_pooled(4096).unwrap();
        b.host_write(|v: &mut [u8]| v.fill(0x5a)).unwrap();
        c.run(|cmd| cmd.dispatch("double", 1, &[Arg::Buffer(&b)], &())).unwrap();
        assert_eq!(b.read_back_zero(), ReadBack::NonZero);
        drop(b);
        let w = end_request_report::<Mock>(true).expect("every idle buffer reads zero");
        assert!(w.pool_regions >= 2, "{w:?}");
        let b = c.alloc_pooled(4096).unwrap();
        assert!(b.host_read(|v: &[u8]| v.iter().all(|x| *x == 0)).unwrap());
    }
}
