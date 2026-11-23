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

//! The Metal device context (macOS and iOS only).
//!
//! One process-wide context, made lazily at the first use ([`context`]): the
//! system default `MTLDevice`, one command queue, the shader library (the
//! embedded metallib, else the embedded source compiled at that moment, see
//! `build.rs`) and a cache of compute pipeline states by kernel name.
//!
//! # Use from a kernel module
//!
//! ```ignore
//! let mut s = ctx::lock()?;                  // serialises device work
//! let a = s.ctx().upload(scalars)?;           // a scoped, zeroed-on-drop buffer
//! let tw = s.scratch("ntt-twiddles", bytes)?; // a per-request buffer
//! let ns = s.ctx().run(|c| {
//!     c.dispatch("ntt_radix2", n / 2, &[Arg::Buffer(&a), Arg::value(&n)])?;
//!     Ok(())
//! })?.1;
//! a.read(out);
//! ```
//!
//! [`lock`] returns a [`Session`]: device work, the per-request buffers and
//! [`end_request`] are serialised by it, so [`end_request`] never zeroes a
//! buffer a kernel is using. A [`Ctx::run`] encodes any number of dispatches
//! into one command buffer (one serial compute encoder: each dispatch sees the
//! writes of the previous ones), commits it, waits for it and returns the GPU
//! time. A command buffer that ends in the error state is an
//! [`Error::Device`], never a panic.
//!
//! # Memory on unified-memory Apple GPUs
//!
//! Apple Silicon has one physical memory shared by the CPU and the GPU. Every
//! buffer here is `MTLResourceStorageModeShared`: its `contents()` pointer is
//! host-visible and coherent with the GPU at command-buffer boundaries, so a
//! "transfer" is a plain `memcpy` into or out of the buffer, with no DMA and
//! no staging copy. Reading results after [`Ctx::run`] returns needs no blit
//! and no synchronisation call (`didModifyRange` is a managed-mode concept).
//!
//! Wrapping a caller's slice without a copy
//! (`newBufferWithBytesNoCopy:length:options:deallocator:`) needs a
//! page-aligned start address and a length that is a multiple of the page
//! size (16 KiB on Apple Silicon). Slices from the prover are ordinary heap
//! allocations, so they are generally copied ([`Ctx::upload`],
//! [`Buffer::write`]); at memory bandwidth the copy is small next to the
//! kernels. A copy also keeps the hook contract simple: the caller's slice is
//! written only after the device result is complete, so a decline leaves it
//! unchanged.
//!
//! [`device_bytes`] counts the bytes this process holds in Metal buffers (now
//! and peak since [`begin_request`]), the idle buffers of the pool included;
//! [`in_use_bytes`] leaves them out, and [`request_peak_bytes`] is its peak
//! since [`begin_request`]. Metal's own figure for the device,
//! `currentAllocatedSize`, also includes the library, pipelines and driver
//! allocations.
//!
//! # Buffers
//!
//! - [`Buffer`]: owned by the caller, zeroed when dropped if it may hold a
//!   non-zero byte (written by the host or bound to a kernel since it was made
//!   or zeroed). Large zeroing runs on several threads.
//! - Pooled buffers ([`Ctx::alloc_pooled`]): a zeroed buffer from the pool, or
//!   a new one; when dropped it is zeroed and goes back to the pool instead of
//!   being released. [`release_later`] drops buffers on a background thread;
//!   [`end_request_report`] waits for it.
//! - Per-request buffers ([`Session::scratch`], and [`take_request_buffer`] for
//!   a call that fills its buffer without holding the device): kept by the
//!   context under a key while a request is open ([`begin_request`]) until
//!   [`end_request_report`], which zeroes each one, with its check reads it
//!   back from the host mapping (and every idle buffer of the pool too), and
//!   puts it back into the pool; a region, or an idle buffer of the pool, that
//!   reads back non-zero (or cannot be read back) is zeroed again, read back
//!   again and released, never pooled, and the report is `Err`. A kernel uses
//!   them for scratch it reuses across the calls of one request. A hook call
//!   made outside a request zeroes them before it returns (`crate::HookScope`,
//!   [`wipe_unbracketed`]).
//!
//! # Residency and first use
//!
//! The start of a command buffer waits while Metal makes its buffers
//! resident: for a new buffer, and for one not used for about two seconds.
//! On an M4 Max each costs 13 to 30 ms for the 448 MiB of the quotient's
//! cosets at `k = 20`. The pool of zeroed buffers avoids the first, the
//! queue's residency set (see [`Ctx`]) the second.

use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    ffi::c_void,
    marker::PhantomData,
    panic::{AssertUnwindSafe, catch_unwind},
    ptr::NonNull,
    sync::{
        Arc, Condvar, Mutex, MutexGuard, OnceLock,
        atomic::{
            AtomicBool, AtomicU64,
            Ordering::{Relaxed, SeqCst},
        },
    },
};

#[cfg(midnight_metal_metallib)]
use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSObjectProtocol, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResource, MTLResourceOptions, MTLSize,
    MTLStorageMode,
};

use crate::{MetalInfo, account::Reason};

// MTLCreateSystemDefaultDevice returns no device in a process that does not
// link CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// The Metal Shading Language version of the shaders (set by `build.rs`).
pub const MSL_VERSION: &str = env!("MIDNIGHT_METAL_MSL_VERSION");

#[cfg(midnight_metal_metallib)]
static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/midnight_metal.metallib"));
static SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/midnight_metal.metal"));

/// A device failure, mapped to a decline reason by [`Error::reason`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// No usable device, or the context could not be made.
    NoDevice(String),
    /// A buffer could not be allocated (or is above `maxBufferLength`).
    Memory(String),
    /// A pipeline, encoder or command buffer failed.
    Device(String),
    /// A scalar argument of a device building block is not below the
    /// modulus (the device field arithmetic takes reduced inputs only).
    Unreduced(String),
}

impl Error {
    /// The decline reason of this error: `no-device`, `memory`,
    /// `device-error` or `unreduced`.
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

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// ---------------------------------------------------------------------------
// Detection

struct Probe {
    device: Device,
    info: MetalInfo,
}

static PROBE: OnceLock<Option<Probe>> = OnceLock::new();

fn probe_device() -> Option<&'static Probe> {
    PROBE
        .get_or_init(|| {
            let device = MTLCreateSystemDefaultDevice()?;
            let tg = device.maxThreadsPerThreadgroup();
            let apple_family = (1..=10u32)
                .rev()
                .find(|f| device.supportsFamily(MTLGPUFamily(1000 + *f as isize)))
                .unwrap_or(0);
            let info = MetalInfo {
                name: device.name().to_string(),
                registry_id: device.registryID(),
                unified_memory: device.hasUnifiedMemory(),
                recommended_max_working_set: device.recommendedMaxWorkingSetSize(),
                max_buffer_length: device.maxBufferLength() as u64,
                max_threads_per_threadgroup: tg.width as u64,
                max_threadgroup_memory_length: device.maxThreadgroupMemoryLength() as u64,
                apple_family,
                metal3: device.supportsFamily(MTLGPUFamily::Metal3),
                msl_version: MSL_VERSION,
            };
            Some(Probe { device, info })
        })
        .as_ref()
}

/// The device's properties, without making the context (no library is
/// loaded). `None`: no Metal device.
pub fn probe() -> Option<&'static MetalInfo> {
    probe_device().map(|p| &p.info)
}

// ---------------------------------------------------------------------------
// Context

/// The process-wide device context.
pub struct Ctx {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    library_source: String,
    pipelines: Mutex<HashMap<String, Pipeline>>,
    /// The queue's residency set (see [`Resident`]); `None` where Metal has
    /// none (before macOS 15 and iOS 18).
    resident: Option<Mutex<Resident>>,
}

/// Buffers of at least this many bytes are in the queue's residency set
/// while they live.
const RESIDENT_MIN: usize = 1 << 20;

/// The queue's residency set.
///
/// Metal makes a buffer resident for the GPU when a command buffer uses it,
/// and lets it go once it has not been used for about two seconds, even
/// while the GPU works on other buffers. Its next use then waits while it is
/// made resident again: measured on an M4 Max, two 448 MiB buffers unused
/// for 2 to 5 s while the GPU ran other work waited 13 to 19 ms from
/// `commit` to `GPUStartTime` (unused for 1 s: 0.3 ms). A proof takes about
/// 2 s, so the quotient's buffers, the transform buffers and the MSM tables
/// would be made resident again at every proof. A buffer in the queue's
/// residency set stays resident (in the same measurement: no wait). [`Ctx`]
/// adds every buffer of at least [`RESIDENT_MIN`] bytes when it is made and
/// removes it when it is released; an idle buffer of the pool stays in it.
/// The set also makes a new buffer resident when it is added, so its first
/// command buffer does not wait for that either.
struct Resident {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

// SAFETY: a residency set may be used from any thread; every change goes
// through the mutex around it.
unsafe impl Send for Resident {}

impl Ctx {
    fn residency(&self) -> Option<MutexGuard<'_, Resident>> {
        self.resident.as_ref().map(|r| r.lock().unwrap_or_else(|e| e.into_inner()))
    }

    /// Adds a new buffer to the residency set (when it is large enough).
    fn keep(&self, b: &ProtocolObject<dyn MTLBuffer>) {
        if b.length() >= RESIDENT_MIN
            && let Some(r) = self.residency()
        {
            r.set.addAllocation(ProtocolObject::from_ref(b));
            r.set.commit();
            r.set.requestResidency();
        }
    }
}

/// Removes a buffer that is being released from the residency set (when it
/// is large enough to be in it). A buffer is released by its last owner,
/// after the GPU work that used it.
fn unkeep(b: &ProtocolObject<dyn MTLBuffer>) {
    if b.length() >= RESIDENT_MIN
        && let Some(Ok(c)) = CTX.get()
        && let Some(r) = c.residency()
    {
        r.set.removeAllocation(ProtocolObject::from_ref(b));
        r.set.commit();
    }
}

impl std::fmt::Debug for Ctx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ctx")
            .field("device", &self.device.name().to_string())
            .field("library_source", &self.library_source)
            .finish_non_exhaustive()
    }
}

static CTX: OnceLock<Result<Ctx, String>> = OnceLock::new();

/// The context, made at the first call. The error of a failed attempt (no
/// device, a library that does not load) is kept: later calls return it
/// without trying again.
pub fn context() -> Result<&'static Ctx, Error> {
    CTX.get_or_init(make_context).as_ref().map_err(|e| Error::NoDevice(e.clone()))
}

/// Why the context could not be made, if it was tried and failed.
pub fn context_error() -> Option<&'static str> {
    match CTX.get() {
        Some(Err(e)) => Some(e),
        _ => None,
    }
}

fn make_context() -> Result<Ctx, String> {
    let probe = probe_device().ok_or("no Metal device")?;
    let device = probe.device.clone();
    let queue = device.newCommandQueue().ok_or("newCommandQueue returned nil")?;
    let (library, library_source) = load_library(&device)?;
    // the residency set of the large buffers (macOS 15+, iOS 18+)
    let resident = if device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
        let desc = MTLResidencySetDescriptor::new();
        device.newResidencySetWithDescriptor_error(&desc).ok().map(|set| {
            queue.addResidencySet(&set);
            Mutex::new(Resident { set })
        })
    } else {
        None
    };
    Ok(Ctx {
        device,
        queue,
        library,
        library_source,
        pipelines: Mutex::new(HashMap::new()),
        resident,
    })
}

/// The shader library and where it came from ([`Ctx::library_source`]): the
/// embedded metallib, else the embedded source compiled now.
fn load_library(
    device: &Device,
) -> Result<(Retained<ProtocolObject<dyn MTLLibrary>>, String), String> {
    #[cfg(midnight_metal_metallib)]
    let how = if std::env::var("MIDNIGHT_METAL_SHADER_SOURCE").as_deref() == Ok("1") {
        "source (MIDNIGHT_METAL_SHADER_SOURCE=1)".to_string()
    } else {
        let data = DispatchData::from_static_bytes(METALLIB);
        match device.newLibraryWithData_error(&data) {
            Ok(lib) => return Ok((lib, "metallib".into())),
            Err(e) => {
                let why = e.localizedDescription().to_string();
                eprintln!(
                    "midnight-metal: the embedded metallib did not load ({why}); \
                     compiling the embedded shader source"
                );
                format!("source (metallib refused: {why})")
            }
        }
    };
    #[cfg(not(midnight_metal_metallib))]
    let how = "source (no metallib was built)".to_string();
    let options = MTLCompileOptions::new();
    options.setLanguageVersion(language_version()?);
    let lib = device
        .newLibraryWithSource_options_error(&NSString::from_str(SOURCE), Some(&options))
        .map_err(|e| format!("shader compile failed: {}", e.localizedDescription()))?;
    Ok((lib, how))
}

/// [`MSL_VERSION`] as `MTLCompileOptions.languageVersion`.
fn language_version() -> Result<MTLLanguageVersion, String> {
    let (major, minor) = MSL_VERSION
        .split_once('.')
        .and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?)))
        .ok_or_else(|| format!("bad MSL version {MSL_VERSION}"))?;
    Ok(MTLLanguageVersion((major << 16) + minor))
}

// ---------------------------------------------------------------------------
// Byte accounting

static BYTES_NOW: AtomicU64 = AtomicU64::new(0);
static BYTES_PEAK: AtomicU64 = AtomicU64::new(0);
/// Bytes in buffers that are not idle in the pool.
static IN_USE: AtomicU64 = AtomicU64::new(0);
/// The peak of [`IN_USE`] since process start or the last [`begin_request`].
static IN_USE_PEAK: AtomicU64 = AtomicU64::new(0);

/// Adds `bytes` to [`IN_USE`] (a new buffer, or one taken from the pool) and
/// raises its peak.
fn in_use_add(bytes: u64) {
    let now = IN_USE.fetch_add(bytes, Relaxed) + bytes;
    IN_USE_PEAK.fetch_max(now, Relaxed);
}

/// Counts a new buffer of `bytes` bytes (held and in use).
fn count_alloc(bytes: u64) {
    let now = BYTES_NOW.fetch_add(bytes, Relaxed) + bytes;
    BYTES_PEAK.fetch_max(now, Relaxed);
    in_use_add(bytes);
}

/// `(now, peak)` bytes this process holds in Metal buffers of this crate,
/// the idle buffers of the pool included ([`pool_bytes`]). The peak is since
/// process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    (BYTES_NOW.load(Relaxed), BYTES_PEAK.load(Relaxed))
}

/// Bytes in buffers in use now: [`device_bytes`] without the idle buffers of
/// the pool.
pub fn in_use_bytes() -> u64 {
    IN_USE.load(Relaxed)
}

/// The peak of [`in_use_bytes`] since the last [`begin_request`] (or process
/// start): every buffer handed out (new or taken from the pool, per-request,
/// scoped or per-call), the resident SRS tables and proving-key copies, and
/// not the idle buffers of the pool. A buffer a request takes from the pool
/// counts from the moment it is taken. [`begin_request`] resets it to the
/// bytes in use then (what earlier requests left resident).
pub fn request_peak_bytes() -> u64 {
    IN_USE_PEAK.load(Relaxed)
}

/// Whether a request is open: between [`begin_request`] and
/// [`end_request_report`] / [`end_request`].
static REQUEST_OPEN: AtomicBool = AtomicBool::new(false);

/// Opens a request (per-request buffers are then kept for reuse across the
/// calls of the request until [`end_request_report`]) and resets the peaks of
/// [`device_bytes`] and [`request_peak_bytes`] to the current values.
pub fn begin_request() {
    BYTES_PEAK.store(BYTES_NOW.load(Relaxed), Relaxed);
    IN_USE_PEAK.store(IN_USE.load(Relaxed), Relaxed);
    REQUEST_OPEN.store(true, SeqCst);
}

/// Whether a request is open ([`begin_request`] was called and its
/// [`end_request_report`] has not started yet).
pub fn request_open() -> bool {
    REQUEST_OPEN.load(SeqCst)
}

// ---------------------------------------------------------------------------
// The pool of zeroed buffers

/// Zeroed buffers kept for reuse ([`Ctx::alloc_pooled`]).
///
/// A new shared buffer costs more than its allocation: the first command
/// buffer that uses it waits while its pages are made resident for the GPU
/// (18 to 22 ms for 448 MiB, measured on an M4 Max as the time from `commit`
/// to `GPUStartTime`, with no residency set; about 4 ms with it, which makes
/// a new buffer resident when it is added). A buffer that is resident starts
/// within 0.1 ms. So the large per-request and per-call buffers come from
/// this pool and go back to it, and stay in the residency set meanwhile.
///
/// Every buffer in the pool reads zero (the pool's invariant): a buffer goes
/// back only zero-filled by Metal and never written, or after its zeroing
/// (on the CPU, of its whole allocation; see [`pool_put`]), so a buffer that
/// held a witness value is never handed out again, nor reported erased,
/// before it was zeroed. A buffer whose zeroing panicked, or that read back
/// non-zero in the erasure check, is released instead. The check mode of
/// [`end_request_report`] reads back every idle buffer, and [`release_pool`]
/// (in a debug build) asserts the invariant before it frees them. The pool
/// keeps at most [`pool_limit`] bytes (a buffer that would exceed it is
/// released instead); [`release_pool`] releases every idle buffer, and a new
/// allocation that would not fit releases idle buffers first ([`Ctx::alloc`]).
struct Pool {
    free: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    bytes: u64,
}

// SAFETY: Metal buffers may be used from any thread; the pool's buffers are
// idle (no host reference to their contents, no GPU work).
unsafe impl Send for Pool {}

/// The pool of zeroed buffers.
static POOL: Mutex<Pool> = Mutex::new(Pool {
    free: Vec::new(),
    bytes: 0,
});
/// [`pool_limit`]: [`POOL_LIMIT_UNREAD`] until read; every stored limit is
/// below it.
static POOL_LIMIT: AtomicU64 = AtomicU64::new(POOL_LIMIT_UNREAD);
/// [`POOL_LIMIT`] before the environment is read.
const POOL_LIMIT_UNREAD: u64 = u64::MAX;

/// The pool, locked. Like every lock of this crate, a poisoned one is taken
/// over: no update under it is left half done by a panic.
fn pool() -> MutexGuard<'static, Pool> {
    POOL.lock().unwrap_or_else(|e| e.into_inner())
}

/// The most bytes the pool keeps idle: `MIDNIGHT_METAL_POOL_MIB` (read
/// once), else a quarter of the device's `recommendedMaxWorkingSetSize` (0
/// without a device); the tests can set it (`set_pool_limit`, feature
/// `test-support`).
pub fn pool_limit() -> u64 {
    let l = POOL_LIMIT.load(Relaxed);
    if l != POOL_LIMIT_UNREAD {
        return l;
    }
    // a huge value saturates below the marker, so it is read once
    let l = std::env::var("MIDNIGHT_METAL_POOL_MIB")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .map(|m| m.saturating_mul(1 << 20))
        .unwrap_or_else(|| probe().map_or(0, |i| i.recommended_max_working_set / 4))
        .min(POOL_LIMIT_UNREAD - 1);
    let _ = POOL_LIMIT.compare_exchange(POOL_LIMIT_UNREAD, l, Relaxed, Relaxed);
    POOL_LIMIT.load(Relaxed)
}

/// Sets [`pool_limit`] in bytes (`None`: the default again) and releases
/// idle buffers above it. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_pool_limit(limit: Option<u64>) {
    POOL_LIMIT.store(
        limit.map_or(POOL_LIMIT_UNREAD, |l| l.min(POOL_LIMIT_UNREAD - 1)),
        Relaxed,
    );
    let limit = pool_limit();
    let freed: Vec<_> = {
        let mut p = pool();
        let mut out = Vec::new();
        while p.bytes > limit {
            let Some(b) = p.free.pop() else { break };
            p.bytes -= b.length() as u64;
            out.push(b);
        }
        out
    };
    release_raw(freed);
}

/// Bytes of the idle (zeroed) buffers in the pool.
pub fn pool_bytes() -> u64 {
    pool().bytes
}

/// Releases idle buffers taken out of the pool (zero, by the pool's
/// invariant; a debug build reads each one back first).
fn release_raw(bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>) -> u64 {
    debug_assert!(
        bufs.iter().all(|b| raw_reads_zero(b)),
        "an idle buffer of the pool does not read zero"
    );
    release_unchecked(bufs)
}

/// Releases buffers taken out of the pool: out of the residency set and out
/// of the byte count. Returns their bytes.
fn release_unchecked(bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>) -> u64 {
    let bytes: u64 = bufs.iter().map(|b| b.length() as u64).sum();
    for b in &bufs {
        unkeep(b);
    }
    drop(bufs);
    BYTES_NOW.fetch_sub(bytes, Relaxed);
    bytes
}

/// Releases every idle buffer of the pool (after the zeroing still running
/// in the background has ended, see [`release_later`]). Returns the bytes
/// released. Every buffer it frees reads zero: that is the pool's invariant
/// (see [`Pool`]), which a debug build asserts by reading each one back.
pub fn release_pool() -> u64 {
    settle_releases();
    let bufs = {
        let mut p = pool();
        p.bytes = 0;
        std::mem::take(&mut p.free)
    };
    release_raw(bufs)
}

/// A zeroed buffer of the pool of at least `bytes` bytes and at most a
/// quarter more (the smallest such), if there is one.
fn pool_take(bytes: usize) -> Option<Retained<ProtocolObject<dyn MTLBuffer>>> {
    let mut p = pool();
    let max = bytes.saturating_add(bytes / 4);
    let i = p
        .free
        .iter()
        .enumerate()
        .filter(|(_, b)| (bytes..=max).contains(&b.length()))
        .min_by_key(|(_, b)| b.length())?
        .0;
    let b = p.free.swap_remove(i);
    p.bytes -= b.length() as u64;
    in_use_add(b.length() as u64);
    Some(b)
}

/// Releases idle buffers of the pool, the largest first, until at least
/// `need` bytes are freed or the pool is empty. Returns the bytes freed.
fn release_idle(need: u64) -> u64 {
    let freed: Vec<_> = {
        let mut p = pool();
        p.free.sort_by_key(|b| b.length());
        let mut out = Vec::new();
        let mut got = 0u64;
        while got < need {
            let Some(b) = p.free.pop() else { break };
            let len = b.length() as u64;
            p.bytes -= len;
            got += len;
            out.push(b);
        }
        out
    };
    release_raw(freed)
}

/// Puts a zeroed buffer into the pool, or releases it when the pool would
/// exceed its limit. The caller has zeroed the buffer's whole allocation
/// (or got it zero-filled from Metal and never wrote it): the only callers
/// are [`Buffer`]'s drop, after [`Buffer::erase`], which zeroes every buffer
/// that may hold a non-zero byte (its `dirty` flag, set by every host write
/// and every kernel binding) and never pools one whose zeroing failed, and
/// the erasure report, which never pools a region that read back non-zero.
fn pool_put(raw: Retained<ProtocolObject<dyn MTLBuffer>>) {
    let len = raw.length() as u64;
    IN_USE.fetch_sub(len, Relaxed);
    let limit = pool_limit();
    let back = {
        let mut p = pool();
        if p.bytes + len <= limit {
            p.bytes += len;
            p.free.push(raw);
            None
        } else {
            Some(raw)
        }
    };
    if let Some(b) = back {
        release_raw(vec![b]);
    }
}

/// The zeroing and release running on background threads
/// ([`release_later`]): each call has a ticket; `outstanding` holds the
/// tickets whose buffers are not all zeroed and pooled or released yet.
struct Releases {
    next: u64,
    outstanding: BTreeSet<u64>,
}

/// The tickets of [`release_later`].
static RELEASES: Mutex<Releases> = Mutex::new(Releases {
    next: 0,
    outstanding: BTreeSet::new(),
});
/// Notified whenever a ticket of [`RELEASES`] completes.
static RELEASED: Condvar = Condvar::new();

/// The tickets, locked (a poisoned lock is taken over).
fn releases() -> MutexGuard<'static, Releases> {
    RELEASES.lock().unwrap_or_else(|e| e.into_inner())
}

/// Zeroings that panicked (each one was completed on the calling thread
/// instead, and its buffer was released, not pooled) since the last
/// [`end_request_report`], which reports them as erasure failures.
static ERASE_FAULTS: AtomicU64 = AtomicU64::new(0);

/// Drops `bufs` (each zeroed, then pooled or released) on a background
/// thread, so that a caller does not wait for the zeroing. When no thread
/// can be made, they are dropped on the calling thread before this returns.
/// [`settle_releases`] (and so [`end_request`], [`end_request_report`] and
/// [`release_pool`]) waits for every release made before it was called: at
/// the end of a request every such buffer has been zeroed.
pub fn release_later(bufs: Vec<Buffer>) {
    if bufs.is_empty() {
        return;
    }
    let ticket = {
        let mut r = releases();
        let t = r.next;
        r.next += 1;
        r.outstanding.insert(t);
        t
    };
    // the buffers stay reachable from here if the thread is not made
    let slot = Arc::new(Mutex::new(Some(bufs)));
    let theirs = Arc::clone(&slot);
    let spawned = std::thread::Builder::new()
        .name("midnight-metal-release".into())
        .spawn(move || release_now(&theirs, ticket));
    if spawned.is_err() {
        release_now(&slot, ticket);
    }
}

/// Drops the buffers of a [`release_later`] call and completes its ticket.
/// Never panics: a zeroing that panics is completed on this thread and
/// counted ([`Buffer::erase`]).
fn release_now(slot: &Mutex<Option<Vec<Buffer>>>, ticket: u64) {
    let bufs = slot.lock().unwrap_or_else(|e| e.into_inner()).take();
    for b in bufs.into_iter().flatten() {
        // `Buffer::drop` does not unwind (its zeroing catches a panic); a
        // panic from the pool or residency bookkeeping after it is counted
        if catch_unwind(AssertUnwindSafe(|| drop(b))).is_err() {
            ERASE_FAULTS.fetch_add(1, Relaxed);
        }
    }
    releases().outstanding.remove(&ticket);
    RELEASED.notify_all();
}

/// Whether a [`release_later`] call has not completed yet.
pub(crate) fn releases_pending() -> bool {
    !releases().outstanding.is_empty()
}

/// Waits until every [`release_later`] call made before this call has
/// completed (its buffers zeroed and pooled or released), whatever other
/// thread waits for them at the same time.
pub fn settle_releases() {
    let mut r = releases();
    let target = r.next;
    while r.outstanding.first().is_some_and(|t| *t < target) {
        r = RELEASED.wait(r).unwrap_or_else(|e| e.into_inner());
    }
}

// ---------------------------------------------------------------------------
// Host copies

/// Copies `src` into `dst` (equal lengths), on several threads when long (a
/// single thread does not reach the memory bandwidth of the machine). Never
/// panics: a part whose thread cannot be made is copied on the calling
/// thread, so a copy into a caller's slice never stops half way.
pub(crate) fn par_copy(dst: &mut [crate::Scalar], src: &[crate::Scalar]) {
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
// Buffers

/// Plain data that may be copied into and out of a buffer byte for byte:
/// every bit pattern is a valid value and there is no padding.
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

/// A shared-storage Metal buffer, zeroed when dropped.
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    bytes: usize,
    /// Goes back to the pool when dropped (after its zeroing); cleared when
    /// its zeroing panicked or it failed the erasure check, so that it is
    /// released instead.
    pooled: bool,
    /// May hold a non-zero byte: written by the host or bound to a kernel
    /// since it was made or last zeroed.
    dirty: AtomicBool,
}

// SAFETY: Metal buffers may be used from any thread; host access to the
// contents goes through `&self` / `&mut self`, and GPU access happens only
// inside `Ctx::run`, which borrows the buffer and waits for the GPU.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer").field("bytes", &self.bytes).finish()
    }
}

/// Byte runs of at least this size are zeroed, copied or checked on several
/// threads: one core does not reach the memory bandwidth.
const PAR_BYTES: usize = 4 << 20;

/// The threads of a parallel host pass over memory.
fn mem_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |t| t.get()).min(8)
}

/// `f(start, len)` over `[0, len)` in parts of at least [`PAR_BYTES`] (a
/// multiple of 64), on several threads; `true` when every part returned
/// `true`.
///
/// Every part is done exactly once to completion: a part whose thread cannot
/// be made, or whose thread panicked, is done (again) on the calling thread.
/// So with an `f` that does not panic (zeroing, reading back) this does not
/// panic and covers the whole range.
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

/// Writes zero over the whole allocation of `raw` (on several threads when
/// it is large). Does not panic.
///
/// The caller has exclusive use of the contents: no host reference to them
/// and no GPU work on the buffer.
fn raw_zero(raw: &ProtocolObject<dyn MTLBuffer>) {
    let base = raw.contents().as_ptr() as usize;
    par_ranges(raw.length(), |a, n| {
        // SAFETY: `contents` is valid for `length()` bytes, the caller has
        // exclusive use of them, and the parts are disjoint.
        unsafe { std::ptr::write_bytes((base + a) as *mut u8, 0, n) };
        true
    });
    std::sync::atomic::compiler_fence(SeqCst);
}

/// Whether every byte of `raw`'s allocation reads zero, read word by word
/// with volatile loads from the host mapping (on several threads when it is
/// large). Does not panic.
fn raw_reads_zero(raw: &ProtocolObject<dyn MTLBuffer>) -> bool {
    let base = raw.contents().as_ptr() as usize;
    par_ranges(raw.length(), |a, n| {
        let p = (base + a) as *const u8;
        let words = n / 8;
        // SAFETY: `contents` is valid for the allocation's bytes; `a` is a
        // multiple of 64 and `contents` page aligned, so `p` is 8-byte
        // aligned; the reads stay inside the part.
        unsafe {
            (0..words).all(|i| std::ptr::read_volatile(p.cast::<u64>().add(i)) == 0)
                && (words * 8..n).all(|i| std::ptr::read_volatile(p.add(i)) == 0)
        }
    })
}

/// Debug builds: samples a new buffer of `newBufferWithLength:options:`,
/// which Apple documents as zero-filled, and asserts the samples read zero.
fn debug_assert_zero_filled(raw: &ProtocolObject<dyn MTLBuffer>) {
    if cfg!(debug_assertions) {
        let len = raw.length();
        let p = raw.contents().as_ptr() as *const u8;
        // the first and last bytes, and about 64 bytes spread between
        let step = (len / 64).max(1);
        let ok = (0..len).step_by(step).chain([len - 1]).all(|i| {
            // SAFETY: `i < len`, inside the allocation; nothing else uses
            // the new buffer yet.
            unsafe { std::ptr::read_volatile(p.add(i)) == 0 }
        });
        debug_assert!(ok, "a new Metal buffer of {len} bytes is not zero-filled");
    }
}

impl Buffer {
    /// The length in bytes.
    pub fn len_bytes(&self) -> usize {
        self.bytes
    }

    /// The host address of the contents.
    fn ptr(&self) -> *mut u8 {
        self.raw.contents().as_ptr().cast()
    }

    /// Records that the buffer may now hold a non-zero byte.
    fn touch(&self) {
        self.dirty.store(true, Relaxed);
    }

    /// The contents as a slice of `T` (`len_bytes / size_of::<T>()`
    /// elements).
    pub fn as_slice<T: Pod>(&self) -> &[T] {
        // SAFETY: `contents` is page aligned, valid for `bytes` bytes, and no
        // GPU work writes it while `&self` is held outside `Ctx::run`.
        unsafe { std::slice::from_raw_parts(self.ptr().cast(), self.bytes / size_of::<T>()) }
    }

    /// The contents as a mutable slice of `T`.
    pub fn as_mut_slice<T: Pod>(&mut self) -> &mut [T] {
        self.touch();
        // SAFETY: as `as_slice`, and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr().cast(), self.bytes / size_of::<T>()) }
    }

    /// Copies `data` to the start of the buffer. Panics if it does not fit.
    pub fn write<T: Pod>(&mut self, data: &[T]) {
        self.as_mut_slice::<T>()[..data.len()].copy_from_slice(data);
    }

    /// Copies the start of the buffer into `out`. Panics if the buffer is
    /// shorter.
    pub fn read<T: Pod>(&self, out: &mut [T]) {
        out.copy_from_slice(&self.as_slice::<T>()[..out.len()]);
    }

    /// Writes zero over the whole buffer (its whole allocation, which may be
    /// longer than [`Buffer::len_bytes`]), on several threads when it is
    /// large. Does not panic, and always zeroes every byte: a part whose
    /// thread cannot be made or fails is zeroed on the calling thread.
    pub fn zero(&mut self) {
        // exclusive: `&mut self`
        raw_zero(&self.raw);
        self.dirty.store(false, Relaxed);
    }

    /// Zeroes the buffer if it may hold a non-zero byte. If the zeroing
    /// panics (it does not: this is the last line of defence), the whole
    /// allocation is zeroed again on this thread with no threads and no
    /// allocation, the buffer is marked not to be pooled, and the fault is
    /// counted for [`end_request_report`].
    fn erase(&mut self) {
        if !self.dirty.load(Relaxed) {
            return;
        }
        let r = catch_unwind(AssertUnwindSafe(|| {
            test_switch::zeroing_panic(self.ptr() as usize);
            self.zero()
        }));
        if r.is_err() {
            // SAFETY: valid for `raw.length()` bytes, exclusive (`&mut self`).
            unsafe { std::ptr::write_bytes(self.ptr(), 0, self.raw.length()) };
            std::sync::atomic::compiler_fence(SeqCst);
            self.dirty.store(false, Relaxed);
            self.pooled = false;
            ERASE_FAULTS.fetch_add(1, Relaxed);
        }
    }

    /// The bytes [`Buffer::zero`] writes: the whole allocation.
    pub(crate) fn alloc_bytes(&self) -> usize {
        self.raw.length()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.erase();
        if self.pooled {
            pool_put(self.raw.clone());
        } else {
            unkeep(&self.raw);
            IN_USE.fetch_sub(self.raw.length() as u64, Relaxed);
            BYTES_NOW.fetch_sub(self.raw.length() as u64, Relaxed);
        }
    }
}

impl Ctx {
    /// The library's origin: `metallib`, or `source` (with the reason when
    /// the embedded metallib was not used).
    pub fn library_source(&self) -> &str {
        &self.library_source
    }

    /// A new buffer as a [`Buffer`]: counted, and in the residency set when
    /// large. `None` from Metal is [`Error::Memory`].
    fn wrap(
        &self,
        raw: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        bytes: usize,
        dirty: bool,
    ) -> Result<Buffer, Error> {
        let raw = raw.ok_or_else(|| Error::Memory(format!("buffer of {bytes} bytes")))?;
        count_alloc(raw.length() as u64);
        self.keep(&raw);
        Ok(Buffer {
            raw,
            bytes,
            pooled: false,
            dirty: AtomicBool::new(dirty),
        })
    }

    /// The allocation length of a buffer of `bytes` bytes: `Err` above
    /// `maxBufferLength`.
    fn check_len(&self, bytes: usize) -> Result<usize, Error> {
        let max = self.device.maxBufferLength();
        if bytes > max {
            return Err(Error::Memory(format!(
                "{bytes} bytes is above maxBufferLength {max}"
            )));
        }
        // Metal refuses a zero-length buffer
        Ok(bytes.max(16))
    }

    /// A zero-filled shared buffer of `bytes` bytes (a new one, never from
    /// the pool).
    ///
    /// Zero-filled: `newBufferWithLength:options:` returns a buffer whose
    /// contents are zero (Apple's documentation of the method: "a new buffer
    /// with zero-filled contents"). The buffer is therefore made clean
    /// (`dirty` false: dropped unwritten, it is not zeroed again), and
    /// [`Ctx::alloc_pooled`] may put it into the pool of zeroed buffers
    /// unwritten. A debug build samples the new buffer and asserts it.
    ///
    /// The pool is never the cause of a failed allocation: when this
    /// allocation would take the process above `recommendedMaxWorkingSetSize`
    /// (`currentAllocatedSize` plus the new length), idle buffers of the pool
    /// are released first, the largest first, until it fits; and when Metal
    /// still returns no buffer, every idle buffer is released and the
    /// allocation is tried once more.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer, Error> {
        let len = self.check_len(bytes)?;
        let total = self.device.recommendedMaxWorkingSetSize();
        let used = self.device.currentAllocatedSize() as u64;
        let over = (used + len as u64).saturating_sub(total);
        if over > 0 && pool_bytes() > 0 {
            release_idle(over);
        }
        let make = || {
            self.device
                .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
        };
        let mut raw = make();
        if raw.is_none() && pool_bytes() > 0 {
            release_idle(u64::MAX);
            raw = make();
        }
        if let Some(r) = raw.as_deref() {
            debug_assert_zero_filled(r);
        }
        self.wrap(raw, bytes, false)
    }

    /// A zero-filled shared buffer of `bytes` bytes from the pool of zeroed
    /// buffers (one at most a quarter longer), else a new one ([`Ctx::alloc`],
    /// zero-filled by Metal); it goes back to the pool when dropped, after its
    /// zeroing, or is released when that zeroing panicked or it failed the
    /// erasure check (see the pool's text in this module). For large buffers
    /// made again and again: one that the GPU used before starts without the
    /// mapping delay of a new one.
    pub fn alloc_pooled(&self, bytes: usize) -> Result<Buffer, Error> {
        let len = self.check_len(bytes)?;
        let mut took = pool_take(len);
        if took.is_none() && releases_pending() {
            // a buffer being zeroed in the background may be the one: its
            // zeroing ends sooner than a new buffer is mapped for the GPU
            settle_releases();
            took = pool_take(len);
        }
        let mut b = match took {
            Some(raw) => Buffer {
                raw,
                bytes,
                pooled: true,
                dirty: AtomicBool::new(false),
            },
            None => self.alloc(bytes)?,
        };
        b.pooled = true;
        Ok(b)
    }

    /// A shared buffer holding a copy of `data`: a zero-filled buffer of the
    /// allocation's length ([`Ctx::alloc`]; at least 16 bytes) into which
    /// exactly `size_of_val(data)` bytes are copied, so the bytes past the
    /// data read zero.
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<Buffer, Error> {
        let bytes = size_of_val(data);
        let b = self.alloc(bytes)?;
        if bytes > 0 {
            b.touch();
            // SAFETY: `data` is valid for `bytes` bytes, the new buffer for
            // at least `bytes` bytes (its allocation is `max(bytes, 16)`),
            // nothing else uses it yet, and the two do not overlap.
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), b.ptr(), bytes) };
        }
        Ok(b)
    }

    /// The compute pipeline of a kernel, compiled at its first use and
    /// cached.
    pub fn pipeline(&self, kernel: &str) -> Result<Pipeline, Error> {
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(kernel) {
            return Ok(p.clone());
        }
        let function = self
            .library
            .newFunctionWithName(&NSString::from_str(kernel))
            .ok_or_else(|| Error::Device(format!("no kernel `{kernel}` in the library")))?;
        let p = self.device.newComputePipelineStateWithFunction_error(&function).map_err(|e| {
            Error::Device(format!("pipeline `{kernel}`: {}", e.localizedDescription()))
        })?;
        cache.insert(kernel.to_string(), p.clone());
        Ok(p)
    }

    /// Encodes the dispatches of `f` into one command buffer, commits it and
    /// waits for it. Returns `f`'s value and the GPU time in nanoseconds. A
    /// command buffer that ends in error is [`Error::Device`]; an error from
    /// `f` is returned after the encoded work has completed, so no buffer is
    /// still in use when this returns.
    pub fn run<'a, R>(
        &'a self,
        f: impl FnOnce(&mut Command<'a>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error> {
        objc2::rc::autoreleasepool(|_| {
            let cb = self
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Device("commandBuffer returned nil".into()))?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| Error::Device("computeCommandEncoder returned nil".into()))?;
            let mut cmd = Command {
                ctx: self,
                enc,
                _args: PhantomData,
            };
            let value = f(&mut cmd);
            cmd.enc.endEncoding();
            cb.commit();
            cb.waitUntilCompleted();
            let status = cb.status();
            if status != MTLCommandBufferStatus::Completed {
                let why = cb
                    .error()
                    .map(|e| format!("{} (code {})", e.localizedDescription(), e.code()))
                    .unwrap_or_else(|| format!("status {}", status.0));
                return Err(Error::Device(format!("command buffer failed: {why}")));
            }
            let ns = ((cb.GPUEndTime() - cb.GPUStartTime()) * 1e9).max(0.0) as u64;
            Ok((value?, ns))
        })
    }
}

// ---------------------------------------------------------------------------
// Dispatch

/// A kernel argument, bound at the index of its position in the list.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a> {
    /// A buffer from its start.
    Buffer(&'a Buffer),
    /// A buffer from a byte offset.
    BufferAt(&'a Buffer, usize),
    /// Bytes copied into the command (`setBytes`, at most 4 KiB): a
    /// `constant` value or small struct.
    Bytes(&'a [u8]),
}

impl<'a> Arg<'a> {
    /// A plain value as [`Arg::Bytes`].
    pub fn value<T: Pod>(v: &'a T) -> Self {
        // SAFETY: a Pod value is valid to view as its bytes.
        Arg::Bytes(unsafe { std::slice::from_raw_parts((v as *const T).cast(), size_of::<T>()) })
    }
}

/// The command buffer being encoded inside [`Ctx::run`]: one serial compute
/// encoder, so each dispatch sees the writes of the earlier ones.
pub struct Command<'a> {
    ctx: &'a Ctx,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _args: PhantomData<&'a Buffer>,
}

impl std::fmt::Debug for Command<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Command").finish_non_exhaustive()
    }
}

impl<'a> Command<'a> {
    /// Sets the pipeline of `kernel`, its arguments and its threadgroup
    /// memory; returns the pipeline.
    fn bind(&self, kernel: &str, args: &[Arg<'a>], tg_mem: &[usize]) -> Result<Pipeline, Error> {
        let p = self.ctx.pipeline(kernel)?;
        check_threadgroup_memory(&self.ctx.device, kernel, &p, tg_mem)?;
        self.enc.setComputePipelineState(&p);
        for (i, a) in args.iter().enumerate() {
            match *a {
                Arg::Buffer(b) => {
                    b.touch();
                    // SAFETY: the buffer outlives the command (`'a`), and
                    // `run` waits for the GPU before returning.
                    unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.raw), 0, i) }
                }
                Arg::BufferAt(b, off) => {
                    b.touch();
                    if off > b.bytes {
                        return Err(Error::Device(format!(
                            "argument {i}: offset {off} past the buffer"
                        )));
                    }
                    // SAFETY: as above, and the offset is inside the buffer.
                    unsafe { self.enc.setBuffer_offset_atIndex(Some(&b.raw), off, i) }
                }
                Arg::Bytes(v) => {
                    if v.is_empty() || v.len() > 4096 {
                        return Err(Error::Device(format!(
                            "argument {i}: {} bytes inline",
                            v.len()
                        )));
                    }
                    // SAFETY: valid for `v.len()` bytes; Metal copies them
                    // and never writes through the pointer.
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::from(v).cast::<c_void>(),
                            v.len(),
                            i,
                        )
                    }
                }
            }
        }
        for (i, len) in tg_mem.iter().enumerate() {
            // SAFETY: the total is within the device limit
            // (`check_threadgroup_memory`).
            unsafe { self.enc.setThreadgroupMemoryLength_atIndex(*len, i) };
        }
        Ok(p)
    }

    /// `threads` threads in one dimension, in threadgroups of the pipeline's
    /// largest size up to 256 (the last group may be partial: Apple GPUs take
    /// non-uniform threadgroups). The kernel still checks its index against
    /// its length.
    pub fn dispatch(
        &mut self,
        kernel: &str,
        threads: usize,
        args: &[Arg<'a>],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, &[])?;
        let tg = p.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        self.threads(threads, tg);
        Ok(())
    }

    /// `threads` threads in groups of `group` threads, with threadgroup memory
    /// of `tg_mem[i]` bytes at `[[threadgroup(i)]]`.
    pub fn dispatch_with(
        &mut self,
        kernel: &str,
        threads: usize,
        group: usize,
        args: &[Arg<'a>],
        tg_mem: &[usize],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, tg_mem)?;
        check_group(kernel, &p, group)?;
        self.threads(threads, group);
        Ok(())
    }

    /// `groups` whole threadgroups of `group` threads, with threadgroup
    /// memory as [`Command::dispatch_with`].
    pub fn dispatch_groups(
        &mut self,
        kernel: &str,
        groups: usize,
        group: usize,
        args: &[Arg<'a>],
        tg_mem: &[usize],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, tg_mem)?;
        check_group(kernel, &p, group)?;
        if groups > 0 {
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(size1(groups), size1(group));
        }
        Ok(())
    }

    /// Dispatches `threads` threads in groups of `group` (none when zero).
    fn threads(&self, threads: usize, group: usize) {
        if threads > 0 {
            self.enc.dispatchThreads_threadsPerThreadgroup(size1(threads), size1(group));
        }
    }
}

/// A threadgroup of `group` threads must be non-empty and within the
/// pipeline's `maxTotalThreadsPerThreadgroup`.
fn check_group(kernel: &str, p: &Pipeline, group: usize) -> Result<(), Error> {
    let max = p.maxTotalThreadsPerThreadgroup();
    if group == 0 || group > max {
        return Err(Error::Device(format!(
            "`{kernel}`: threadgroup of {group} threads (pipeline max {max})"
        )));
    }
    Ok(())
}

/// The threadgroup memory of a dispatch of `kernel`, `tg_mem` with the
/// pipeline's static threadgroup memory, must be within the device's
/// `maxThreadgroupMemoryLength` (32 KiB on an M4 Max, a property of each
/// device): above it the dispatch is not encoded and the call declines
/// (`device-error`) instead of running a kernel whose threadgroup arrays the
/// GPU does not hold.
fn check_threadgroup_memory(
    device: &Device,
    kernel: &str,
    p: &Pipeline,
    tg_mem: &[usize],
) -> Result<(), Error> {
    let need = tg_mem.iter().sum::<usize>() + p.staticThreadgroupMemoryLength();
    let max = device.maxThreadgroupMemoryLength();
    if need > max {
        return Err(Error::Device(format!(
            "`{kernel}`: {need} bytes of threadgroup memory (device max {max})"
        )));
    }
    Ok(())
}

/// A one-dimensional `MTLSize`.
fn size1(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

// ---------------------------------------------------------------------------
// Sessions and per-request buffers

/// The per-request buffers of [`Session::scratch`], by key.
#[derive(Default)]
struct State {
    scratch: HashMap<&'static str, Buffer>,
}

/// The device lock and the per-request buffers it guards.
static STATE: Mutex<Option<State>> = Mutex::new(None);
/// A per-request buffer of [`Session::scratch`] may exist (set when one is
/// made, cleared under [`STATE`] when they are all zeroed): lets a call
/// with nothing to wipe skip taking the device ([`wipe_unbracketed`]).
static SCRATCH_ANY: AtomicBool = AtomicBool::new(false);

/// Exclusive use of the device: the context and the per-request buffers.
pub struct Session {
    ctx: &'static Ctx,
    state: MutexGuard<'static, Option<State>>,
}

impl std::fmt::Debug for Session {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session").finish_non_exhaustive()
    }
}

/// Takes the device (waiting for the current holder) after making the
/// context if needed.
pub fn lock() -> Result<Session, Error> {
    let ctx = context()?;
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    state.get_or_insert_with(State::default);
    Ok(Session { ctx, state })
}

impl Session {
    /// The context.
    pub fn ctx(&self) -> &'static Ctx {
        self.ctx
    }

    /// The per-request buffer `key`, of at least `bytes` bytes: the existing
    /// one with its contents if it is long enough, else a zero-filled one
    /// from the pool of zeroed buffers or a new one (the shorter one is
    /// zeroed and goes back to the pool, see [`Ctx::alloc_pooled`]).
    pub fn scratch(&mut self, key: &'static str, bytes: usize) -> Result<&mut Buffer, Error> {
        let ctx = self.ctx;
        let st = self.state.get_or_insert_with(State::default);
        if st.scratch.get(key).is_some_and(|b| b.bytes < bytes) {
            st.scratch.remove(key);
        }
        Ok(match st.scratch.entry(key) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let b = e.insert(ctx.alloc_pooled(bytes)?);
                SCRATCH_ANY.store(true, SeqCst);
                b
            }
        })
    }

    /// The per-request buffer `key`, if it exists.
    pub fn get_scratch(&self, key: &'static str) -> Option<&Buffer> {
        self.state.as_ref()?.scratch.get(key)
    }

    /// Several per-request buffers at once (for binding them together).
    pub fn scratch_many<const N: usize>(&self, keys: [&'static str; N]) -> Option<[&Buffer; N]> {
        let st = self.state.as_ref()?;
        let found: Vec<&Buffer> = keys.iter().map(|k| st.scratch.get(k)).collect::<Option<_>>()?;
        found.try_into().ok()
    }

    /// Bytes held in per-request buffers. Test support.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn scratch_bytes(&self) -> u64 {
        self.state.as_ref().map_or(0, |s| {
            s.scratch.values().map(|b| b.raw.length() as u64).sum()
        })
    }
}

/// Per-request buffers checked out by a call for its own use
/// ([`take_request_buffer`]), by key: the ones given back and idle.
struct Lendable {
    /// The request of the buffers ([`end_request_report`] increments it).
    epoch: u64,
    free: Vec<(&'static str, Buffer)>,
}

/// The idle buffers of [`take_request_buffer`].
static LENDABLE: Mutex<Lendable> = Mutex::new(Lendable {
    epoch: 0,
    free: Vec::new(),
});

/// The idle lendable buffers, locked (a poisoned lock is taken over).
fn lendable() -> MutexGuard<'static, Lendable> {
    LENDABLE.lock().unwrap_or_else(|e| e.into_inner())
}

/// A per-request buffer under `key` taken without the device: the call that
/// takes it uses it alone, for example to fill it on the CPU while another
/// call holds the device, and gives it back with [`RequestBuffer::give_back`]
/// (dropped instead, it is zeroed and goes back to the pool). It is the idle
/// one of at least `bytes` bytes that an earlier call of the request gave
/// back, with its contents, else a zero-filled one from the pool or a new
/// one. [`end_request_report`] zeroes, checks and counts the idle ones as it
/// does [`Session::scratch`]; one given back after its request ended is
/// zeroed then.
pub fn take_request_buffer(key: &'static str, bytes: usize) -> Result<RequestBuffer, Error> {
    let (epoch, found) = {
        let mut l = lendable();
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
        None => context()?.alloc_pooled(bytes)?,
    };
    Ok(RequestBuffer { key, epoch, buf })
}

/// A per-request buffer checked out by [`take_request_buffer`]. Dropped
/// instead of given back, its buffer is zeroed and goes back to the pool.
#[derive(Debug)]
pub struct RequestBuffer {
    key: &'static str,
    epoch: u64,
    buf: Buffer,
}

impl RequestBuffer {
    /// The buffer.
    pub fn buf(&self) -> &Buffer {
        &self.buf
    }

    /// The buffer, mutably.
    pub fn buf_mut(&mut self) -> &mut Buffer {
        &mut self.buf
    }

    /// Gives the buffer back to its request (its contents stay until
    /// [`end_request_report`] zeroes it); after the end of its request it is
    /// zeroed and goes back to the pool.
    pub fn give_back(self) {
        let RequestBuffer { key, epoch, buf } = self;
        let mut l = lendable();
        if l.epoch == epoch {
            l.free.push((key, buf));
        } else {
            drop(l);
            drop(buf);
        }
    }
}

/// Test switches (feature `test-support`, or this crate's unit tests). In
/// other builds every switch is a no-op that the compiler removes, so no
/// code outside the crate can force an erasure failure.
#[cfg(any(test, feature = "test-support"))]
mod test_switch {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed};

    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLBuffer;

    /// [`super::set_plant_nonzero`].
    pub(super) static PLANT_NONZERO: AtomicBool = AtomicBool::new(false);
    /// The contents address of the region of the last planted byte.
    pub(super) static LAST_PLANT: AtomicUsize = AtomicUsize::new(0);
    /// The contents address of the buffer whose next zeroing panics (0:
    /// none).
    pub(super) static ZEROING_PANIC: AtomicUsize = AtomicUsize::new(0);

    /// After the zeroing of region `i` of a wipe: plants a non-zero byte into
    /// the first region of a checked wipe when [`super::set_plant_nonzero`]
    /// is on.
    pub(super) fn plant_after_zeroing(check: bool, i: usize, raw: &ProtocolObject<dyn MTLBuffer>) {
        if check && i == 0 && PLANT_NONZERO.load(Relaxed) {
            plant_raw(raw);
        }
    }

    /// Panics once, in the zeroing of the buffer at `addr`, after
    /// [`super::set_zeroing_panic`].
    pub(super) fn zeroing_panic(addr: usize) {
        if addr != 0 && ZEROING_PANIC.compare_exchange(addr, 0, Relaxed, Relaxed).is_ok() {
            panic!("midnight-metal test switch: a panic in a zeroing");
        }
    }

    /// Writes one non-zero byte into the middle of `raw`'s allocation through
    /// the raw host pointer (the buffer's `dirty` flag is not set, as for a
    /// write the crate did not see) and records its address. The caller has
    /// exclusive use of the buffer.
    pub(super) fn plant_raw(raw: &ProtocolObject<dyn MTLBuffer>) {
        let p = raw.contents().as_ptr().cast::<u8>();
        // SAFETY: the middle byte is inside the allocation; the caller has
        // exclusive use of the buffer.
        unsafe { std::ptr::write_volatile(p.add(raw.length() / 2), 0xa5) };
        LAST_PLANT.store(p as usize, Relaxed);
    }
}

/// The test switches of a normal build: no-ops.
#[cfg(not(any(test, feature = "test-support")))]
mod test_switch {
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLBuffer;

    #[inline(always)]
    pub(super) fn plant_after_zeroing(_: bool, _: usize, _: &ProtocolObject<dyn MTLBuffer>) {}

    #[inline(always)]
    pub(super) fn zeroing_panic(_: usize) {}
}

/// Makes every later [`end_request_report`]`(true)` write a non-zero byte
/// into its first per-request region after the zeroing, through the raw
/// host pointer (as a write the crate did not track would), so that the
/// read-back fails (`true`), or stops it (`false`). A test switch: only with
/// the feature `test-support` (or in this crate's unit tests).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_plant_nonzero(on: bool) {
    test_switch::PLANT_NONZERO.store(on, Relaxed);
}

/// The contents address of the region the last planted byte went into
/// ([`set_plant_nonzero`], [`plant_in_idle_pool`]). Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn last_planted() -> usize {
    test_switch::LAST_PLANT.load(Relaxed)
}

/// Makes the next zeroing of the dropped buffer whose contents are at
/// `addr` panic once (inside `Buffer::erase`, which completes the zeroing
/// on the calling thread, does not pool the buffer and counts the fault);
/// 0 stops it. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_zeroing_panic(addr: usize) {
    test_switch::ZEROING_PANIC.store(addr, Relaxed);
}

/// Writes a non-zero byte into an idle buffer of the pool through the raw
/// host pointer (breaking the pool's invariant as a stray write would);
/// returns its contents address, `None` when the pool is empty. Test
/// support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn plant_in_idle_pool() -> Option<usize> {
    let p = pool();
    let b = p.free.first()?;
    test_switch::plant_raw(b);
    Some(b.contents().as_ptr() as usize)
}

/// The contents addresses of the idle buffers of the pool. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn idle_pool_addresses() -> Vec<usize> {
    pool().free.iter().map(|b| b.contents().as_ptr() as usize).collect()
}

/// Whether every idle buffer of the pool reads zero over its whole
/// allocation. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn idle_pool_reads_zero() -> bool {
    pool().free.iter().all(|b| raw_reads_zero(b))
}

/// The per-request buffers that exist now, [`Session::scratch`] ones then
/// idle [`take_request_buffer`] ones: `(key, allocation bytes, reads zero,
/// contents address)`. Takes the device (waits for device work). Test
/// support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn request_buffers() -> Vec<(&'static str, usize, bool, usize)> {
    let row = |k: &'static str, b: &Buffer| {
        (k, b.alloc_bytes(), raw_reads_zero(&b.raw), b.ptr() as usize)
    };
    let state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let mut out: Vec<_> = state
        .as_ref()
        .map(|st| st.scratch.iter().map(|(k, b)| row(k, b)).collect())
        .unwrap_or_default();
    out.extend(lendable().free.iter().map(|(k, b)| row(k, b)));
    out
}

/// The result of reading one region back from the host mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadBack {
    /// Every byte reads zero.
    Zero,
    /// A byte reads non-zero.
    NonZero,
    /// Not in shared storage: the host cannot read it back.
    Unreadable,
}

/// Reads back the whole allocation of a region (check mode) and counts the
/// result in `w`.
fn read_back(raw: &ProtocolObject<dyn MTLBuffer>, w: &mut crate::WipeStats) -> ReadBack {
    w.checked_regions += 1;
    if raw.storageMode() != MTLStorageMode::Shared {
        w.check_failures += 1;
        ReadBack::Unreadable
    } else if !raw_reads_zero(raw) {
        w.nonzero_regions += 1;
        ReadBack::NonZero
    } else {
        ReadBack::Zero
    }
}

/// Check mode of the erasure report: reads back every idle buffer of the
/// pool (its whole allocation) and counts it as a region. A buffer that
/// reads back non-zero is zeroed again and read back again; it, and a
/// buffer not in shared storage (which cannot be read back), is taken out
/// of the pool and released.
fn check_idle_pool(w: &mut crate::WipeStats) {
    let bad: Vec<_> = {
        let mut p = pool();
        let mut bad = Vec::new();
        let mut i = 0;
        while i < p.free.len() {
            let raw = &p.free[i];
            let len = raw.length() as u64;
            w.regions += 1;
            w.bytes += len;
            w.pool_regions += 1;
            w.pool_bytes += len;
            let fault = match read_back(raw, w) {
                ReadBack::Zero => false,
                ReadBack::Unreadable => true,
                ReadBack::NonZero => {
                    // idle under the pool's lock: exclusive
                    raw_zero(raw);
                    let _again = raw_reads_zero(raw);
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
    };
    // not through `release_raw`: a buffer not in shared storage cannot be
    // read back, so its debug read-back would fail too
    release_unchecked(bad);
}

/// Zeroes every per-request buffer (the device is held: `state` is its
/// lock), after the zeroing still running in the background; with `check`
/// first reads back every idle buffer of the pool, then each per-request
/// buffer after its zeroing. A region that reads back non-zero (or cannot
/// be read back) is zeroed again, read back again and released: never put
/// back into the pool, never kept as scratch. Every other region goes back
/// into the pool, zeroed.
fn wipe_locked(state: &mut Option<State>, check: bool, w: &mut crate::WipeStats) {
    settle_releases();
    if check {
        check_idle_pool(w);
    }
    let lent: Vec<Buffer> = {
        let mut l = lendable();
        l.epoch += 1;
        l.free.drain(..).map(|(_, b)| b).collect()
    };
    let scratch: Vec<Buffer> = state
        .as_mut()
        .map(|st| st.scratch.drain().map(|(_, b)| b).collect())
        .unwrap_or_default();
    SCRATCH_ANY.store(false, SeqCst);
    for (i, mut b) in scratch.into_iter().chain(lent).enumerate() {
        b.zero();
        w.regions += 1;
        w.bytes += b.alloc_bytes() as u64;
        // exclusive: the region is owned here
        test_switch::plant_after_zeroing(check, i, &b.raw);
        if check && read_back(&b.raw, w) != ReadBack::Zero {
            // zeroed again unconditionally, read back again, released
            b.zero();
            let _again = raw_reads_zero(&b.raw);
            b.pooled = false;
        }
        // zero (and not dirty): pooled, or released when it failed
        drop(b);
    }
}

/// Zeroes every per-request buffer after any device work in progress (it
/// takes the device) and after the zeroing still running in the background
/// ([`release_later`]), and closes the request ([`request_open`]).
///
/// With `check` it first reads back every idle buffer of the pool of zeroed
/// buffers (the scoped buffers that held witness values go back there after
/// their zeroing), then reads back each per-request buffer after its
/// zeroing, from the host mapping, and compares it with zero. A region is a
/// buffer's whole allocation: its bytes are the bytes zeroed and read back.
/// On unified memory a shared buffer is the memory the GPU used, so the
/// read-back sees what the device would read.
///
/// Every region that reads back zero goes back into the pool (or is
/// released, see [`Ctx::alloc_pooled`]). A region that reads back non-zero,
/// or cannot be read back from the host, is zeroed again, read back again
/// and released: it never goes back into the pool and is never kept as
/// scratch. `Err` then, with the region counted (`nonzero_regions` or
/// `check_failures`); `Err` also when a zeroing panicked since the last
/// report (`erasure_failures`: the zeroing was completed on the calling
/// thread and the buffer released, not pooled). There is no quarantine.
pub fn end_request_report(check: bool) -> Result<crate::WipeStats, crate::WipeStats> {
    let mut w = crate::WipeStats {
        checked: check,
        ..crate::WipeStats::default()
    };
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    REQUEST_OPEN.store(false, SeqCst);
    wipe_locked(&mut state, check, &mut w);
    drop(state);
    w.erasure_failures = ERASE_FAULTS.swap(0, Relaxed);
    if w.nonzero_regions == 0 && w.check_failures == 0 && w.erasure_failures == 0 {
        Ok(w)
    } else {
        Err(w)
    }
}

/// [`end_request_report`] without the read-back. `true` when done.
pub fn end_request() -> bool {
    end_request_report(false).is_ok()
}

/// The wipe of a hook call made outside a request (`crate::HookScope`):
/// waits for the zeroing running in the background, then zeroes every
/// per-request buffer, as [`end_request`] does without closing a request,
/// without the read-back and without taking the erasure-fault count. It
/// takes the device, so it never zeroes a buffer that another call is
/// using: a call holds the device ([`Session`]) while it uses
/// [`Session::scratch`] buffers, and a [`take_request_buffer`] buffer is
/// not in the idle list until given back. When there is nothing to wipe it
/// returns without taking the device.
pub(crate) fn wipe_unbracketed() {
    let idle_lent = !lendable().free.is_empty();
    if !(SCRATCH_ANY.load(SeqCst) || idle_lent || releases_pending()) {
        return;
    }
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let mut w = crate::WipeStats::default();
    wipe_locked(&mut state, false, &mut w);
}

/// `(free, total)`: `total` is `recommendedMaxWorkingSetSize`, `free` is
/// that minus `currentAllocatedSize`, plus the idle buffers of the pool
/// ([`pool_bytes`]). The idle buffers count as free because they are free
/// for any allocation, not only for one they fit: a pooled buffer is reused
/// for a length within `[len, 1.25 len]`, and a new allocation that would
/// otherwise take the process above `total` (or that Metal refuses) first
/// releases idle buffers ([`Ctx::alloc`]), so the pool never causes an
/// allocation to fail. See [`crate::mem_info`].
pub fn mem_info() -> Option<(usize, usize)> {
    let ctx = context().ok()?;
    let total = ctx.device.recommendedMaxWorkingSetSize() as usize;
    let used = ctx.device.currentAllocatedSize();
    let idle = pool_bytes() as usize;
    Some(((total + idle).saturating_sub(used).min(total), total))
}
