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

//! The Metal backend of `midnight-accel`: the default `MTLDevice`, one command
//! queue, the shader library and a pipeline cache, made at first use
//! ([`context`]); and this crate's names for the shared device context of
//! `midnight_accel::ctx`, with [`Metal`] as the backend.
//!
//! ```ignore
//! let mut s = ctx::lock()?;                  // serialises device work
//! let a = s.ctx().upload(scalars)?;           // a scoped, zeroed-on-drop buffer
//! let tw = s.scratch("ntt-twiddles", bytes)?; // a per-request buffer
//! let ns = s.ctx().run(|c| {
//!     c.dispatch("ntt_radix2", n / 2, &[Arg::Buffer(&a)], &(n as u32))?;
//!     Ok(())
//! })?.1;
//! a.read(out);
//! ```
//!
//! [`Metal::run`] encodes dispatches into one command buffer with one serial
//! encoder and waits for it; a command buffer in the error state is
//! [`Error::Device`]. A kernel's buffers are bound at `[[buffer(i)]]` in list
//! order and its parameter block at the next index.
//!
//! Every buffer is `MTLResourceStorageModeShared`, so host access is a plain
//! copy and the backend is unified. Caller slices are copied rather than
//! wrapped (wrapping needs page-aligned slices), which also means the
//! caller's slice is written only once the device result is complete.

use std::{
    collections::HashMap,
    ffi::c_void,
    marker::PhantomData,
    ptr::NonNull,
    sync::{Mutex, MutexGuard, OnceLock},
};

#[cfg(midnight_metal_metallib)]
use dispatch2::DispatchData;
use midnight_accel::{Backend, DeviceInfo, Encoder, HostState, RawBuffer, ctx as acc};
pub use midnight_accel::ctx::{Error, Pod, ReadBack, bytes_of};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSObjectProtocol, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResidencySet, MTLResidencySetDescriptor, MTLResource, MTLResourceOptions, MTLSize,
    MTLStorageMode,
};

use crate::MetalInfo;

// MTLCreateSystemDefaultDevice returns no device unless the process links
// CoreGraphics.
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// The Metal Shading Language version of the shaders (set by `build.rs`).
pub const MSL_VERSION: &str = env!("MIDNIGHT_METAL_MSL_VERSION");

#[cfg(midnight_metal_metallib)]
static METALLIB: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/midnight_metal.metallib"));
static SOURCE: &str = include_str!(concat!(env!("OUT_DIR"), "/midnight_metal.metal"));

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

// ---------------------------------------------------------------------------
// The shared context under this crate's names

/// The device context: [`Metal`] with the shared buffer management.
pub type Ctx = acc::Ctx<Metal>;
/// A shared-storage Metal buffer, zeroed when dropped.
pub type Buffer = acc::Buffer<Metal>;
/// Exclusive use of the device: the context and the per-request buffers.
pub type Session = acc::Session<Metal>;
/// A per-request buffer checked out by [`take_request_buffer`].
pub type RequestBuffer = acc::RequestBuffer<Metal>;
/// The command buffer being encoded in [`Ctx::run`].
pub type Command<'c, 'a> = acc::Command<'c, 'a, Metal>;
/// A buffer argument of a kernel.
pub type Arg<'a> = acc::Arg<'a, Metal>;

/// The context, made at the first call. A failure is kept: later calls
/// return it without trying again.
pub fn context() -> Result<&'static Ctx, Error> {
    acc::context::<Metal>()
}

/// Takes the device, making the context if needed.
pub fn lock() -> Result<Session, Error> {
    acc::lock::<Metal>()
}

/// `(now, peak)` bytes held in this crate's Metal buffers, idle pool buffers
/// included. The peak is since process start or the last [`begin_request`].
pub fn device_bytes() -> (u64, u64) {
    acc::device_bytes::<Metal>()
}

/// Bytes in buffers in use now: [`device_bytes`] without idle pool buffers.
pub fn in_use_bytes() -> u64 {
    acc::in_use_bytes::<Metal>()
}

/// The peak of [`in_use_bytes`] since the last [`begin_request`], which resets
/// it to the bytes in use then.
pub fn request_peak_bytes() -> u64 {
    acc::request_peak_bytes::<Metal>()
}

/// Opens a request and resets the peaks of [`device_bytes`] and
/// [`request_peak_bytes`].
pub fn begin_request() {
    acc::begin_request::<Metal>()
}

/// Whether a request is open ([`begin_request`] was called and its
/// [`end_request_report`] has not started).
pub fn request_open() -> bool {
    acc::request_open::<Metal>()
}

/// The most bytes the pool keeps idle: `MIDNIGHT_METAL_POOL_MIB` (read once),
/// else a quarter of `recommendedMaxWorkingSetSize`.
pub fn pool_limit() -> u64 {
    acc::pool_limit::<Metal>()
}

/// Sets [`pool_limit`] in bytes (`None`: the default again) and releases
/// idle buffers above it. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_pool_limit(limit: Option<u64>) {
    acc::set_pool_limit::<Metal>(limit)
}

/// Bytes of the idle (zeroed) buffers in the pool.
pub fn pool_bytes() -> u64 {
    acc::pool_bytes::<Metal>()
}

/// Releases every idle buffer of the pool once background zeroing has ended.
/// Returns the bytes released.
pub fn release_pool() -> u64 {
    acc::release_pool::<Metal>()
}

/// Drops `bufs` (each zeroed, then pooled or released) on a background
/// thread; [`settle_releases`] waits for it.
pub fn release_later(bufs: Vec<Buffer>) {
    acc::release_later(bufs)
}

/// Whether a [`release_later`] call has not completed yet.
#[cfg(test)]
pub(crate) fn releases_pending() -> bool {
    acc::releases_pending::<Metal>()
}

/// Waits until every [`release_later`] call made before this one has
/// completed.
pub fn settle_releases() {
    acc::settle_releases::<Metal>()
}

/// A per-request buffer under `key` taken without the device
/// ([`midnight_accel::ctx::take_request_buffer`]).
pub fn take_request_buffer(key: &'static str, bytes: usize) -> Result<RequestBuffer, Error> {
    acc::take_request_buffer::<Metal>(key, bytes)
}

/// Copies `src` into `dst` on several threads when long.
pub(crate) use midnight_accel::ctx::par_copy;

/// Makes every later [`end_request_report`]`(true)` plant a non-zero byte in
/// its first region after zeroing, so the read-back fails. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_plant_nonzero(on: bool) {
    acc::set_plant_nonzero::<Metal>(on)
}

/// The id of the buffer of the last planted byte. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn last_planted() -> u64 {
    acc::last_planted::<Metal>()
}

/// Makes the next zeroing of the dropped buffer `id` panic once; 0 stops it.
/// Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_zeroing_panic(id: u64) {
    acc::set_zeroing_panic::<Metal>(id)
}

/// Writes a non-zero byte into an idle pool buffer; returns its id. Test
/// support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn plant_in_idle_pool() -> Option<u64> {
    acc::plant_in_idle_pool::<Metal>()
}

/// The ids of the idle buffers of the pool. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn idle_pool_ids() -> Vec<u64> {
    acc::idle_pool_ids::<Metal>()
}

/// Whether every idle pool buffer reads zero. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn idle_pool_reads_zero() -> bool {
    acc::idle_pool_reads_zero::<Metal>()
}

/// The per-request buffers now: `(key, allocation bytes, reads zero, id)`.
/// Takes the device. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn request_buffers() -> Vec<(&'static str, usize, bool, u64)> {
    acc::request_buffers::<Metal>()
}

/// Zeroes every per-request buffer after any device work and background
/// zeroing, and closes the request; with `check`, reads every region back.
pub fn end_request_report(check: bool) -> Result<crate::WipeStats, crate::WipeStats> {
    acc::end_request_report::<Metal>(check)
}

/// [`end_request_report`] without the read-back. `true` when done.
pub fn end_request() -> bool {
    acc::end_request::<Metal>()
}

/// The wipe of a hook call made outside a request.
pub(crate) fn wipe_unbracketed() {
    acc::wipe_unbracketed::<Metal>()
}

/// `(free, total)`: `total` is `recommendedMaxWorkingSetSize`, `free` is that
/// minus `currentAllocatedSize` plus the idle pool buffers, which count as
/// free because an allocation that would not fit releases them first.
pub fn mem_info() -> Option<(usize, usize)> {
    acc::mem_info::<Metal>()
}

// ---------------------------------------------------------------------------
// Detection

struct Probe {
    device: Device,
    info: MetalInfo,
    neutral: DeviceInfo,
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
                apple_family,
                metal3: device.supportsFamily(MTLGPUFamily::Metal3),
                msl_version: MSL_VERSION,
            };
            let neutral = DeviceInfo {
                name: info.name.clone(),
                // shared storage: every buffer is host memory
                unified_memory: true,
                memory_budget: info.recommended_max_working_set,
                max_buffer_length: info.max_buffer_length,
                max_threads_per_group: info.max_threads_per_threadgroup,
            };
            Some(Probe {
                device,
                info,
                neutral,
            })
        })
        .as_ref()
}

/// The device's properties, without loading the library. `None`: no Metal
/// device.
pub fn probe() -> Option<&'static MetalInfo> {
    probe_device().map(|p| &p.info)
}

// ---------------------------------------------------------------------------
// Context

/// The Metal backend: the process-wide device, queue, shader library and
/// pipeline cache.
pub struct Metal {
    device: Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
    library_source: String,
    pipelines: Mutex<HashMap<String, Pipeline>>,
    /// The queue's residency set; `None` before macOS 15.
    resident: Option<Mutex<Resident>>,
}

/// Buffers of at least this many bytes stay in the queue's residency set.
const RESIDENT_MIN: usize = 1 << 20;

/// The queue's residency set.
///
/// Metal evicts a buffer unused for about two seconds even while the GPU is
/// busy, and its next use then waits while it is made resident again. A proof
/// takes about that long, so large buffers would be re-made resident every
/// proof. [`Metal`] adds every buffer of at least [`RESIDENT_MIN`] bytes when
/// it is made (idle pool buffers included) and removes it when released.
struct Resident {
    set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

// SAFETY: a residency set may be used from any thread; every change goes
// through the mutex around it.
unsafe impl Send for Resident {}

impl Metal {
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

    /// The library's origin: `metallib`, or `source` with the reason.
    pub fn library_source(&self) -> &str {
        &self.library_source
    }

    /// The compute pipeline of a kernel, compiled at first use and cached.
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
}

/// Removes a buffer being released from the residency set. A buffer is
/// released by its last owner, after the GPU work that used it.
fn unkeep(b: &ProtocolObject<dyn MTLBuffer>) {
    if b.length() >= RESIDENT_MIN
        && let Some(Ok(c)) = CTX.get()
        && let Some(r) = c.residency()
    {
        r.set.removeAllocation(ProtocolObject::from_ref(b));
        r.set.commit();
    }
}

impl std::fmt::Debug for Metal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Metal")
            .field("device", &self.device.name().to_string())
            .field("library_source", &self.library_source)
            .finish_non_exhaustive()
    }
}

static CTX: OnceLock<Result<Metal, String>> = OnceLock::new();

/// Why the context could not be made, if it was tried and failed.
pub fn context_error() -> Option<&'static str> {
    match CTX.get() {
        Some(Err(e)) => Some(e),
        _ => None,
    }
}

fn make_context() -> Result<Metal, String> {
    let probe = probe_device().ok_or("no Metal device")?;
    let device = probe.device.clone();
    let queue = device.newCommandQueue().ok_or("newCommandQueue returned nil")?;
    let (library, library_source) = load_library(&device)?;
    // the residency set of the large buffers (macOS 15+)
    let resident = if device.respondsToSelector(objc2::sel!(newResidencySetWithDescriptor:error:)) {
        let desc = MTLResidencySetDescriptor::new();
        device.newResidencySetWithDescriptor_error(&desc).ok().map(|set| {
            queue.addResidencySet(&set);
            Mutex::new(Resident { set })
        })
    } else {
        None
    };
    Ok(Metal {
        device,
        queue,
        library,
        library_source,
        pipelines: Mutex::new(HashMap::new()),
        resident,
    })
}

/// The shader library and its origin: the embedded metallib, else the
/// embedded source compiled now.
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
// Buffers

/// A shared-storage `MTLBuffer`.
pub struct MetalBuffer(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: Metal buffers may be used from any thread; host access to the
// contents goes through the shared code's `Buffer` (`&self` / `&mut self`),
// and GPU access happens only inside `Metal::run`, which waits for the GPU.
unsafe impl Send for MetalBuffer {}
// SAFETY: as above.
unsafe impl Sync for MetalBuffer {}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer").field("length", &self.0.length()).finish()
    }
}

impl RawBuffer for MetalBuffer {
    fn len(&self) -> usize {
        self.0.length()
    }

    fn id(&self) -> u64 {
        self.0.contents().as_ptr() as u64
    }

    fn host_ptr(&self) -> Option<NonNull<u8>> {
        Some(self.0.contents().cast())
    }

    fn read_back_zero(&self) -> ReadBack {
        if self.0.storageMode() != MTLStorageMode::Shared {
            ReadBack::Unreadable
        } else if acc::host_reads_zero(self.0.contents().cast(), self.0.length()) {
            ReadBack::Zero
        } else {
            ReadBack::NonZero
        }
    }

    fn on_release(&self) {
        unkeep(&self.0);
    }
}

/// Debug builds: asserts that samples of a new buffer read zero (Metal
/// documents new buffers as zero-filled).
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

// ---------------------------------------------------------------------------
// Dispatch

/// The command buffer being encoded in [`Metal::run`]: one serial compute
/// encoder, so each dispatch sees the writes of the earlier ones.
pub struct MetalCommand<'a> {
    ctx: &'a Metal,
    enc: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    _args: PhantomData<&'a Buffer>,
}

impl std::fmt::Debug for MetalCommand<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalCommand").finish_non_exhaustive()
    }
}

impl<'a> MetalCommand<'a> {
    /// Sets the pipeline, the buffers, the parameters (at the index after the
    /// buffers, none when empty) and the threadgroup memory of `kernel`.
    fn bind(
        &self,
        kernel: &str,
        args: &[Arg<'a>],
        params: &[u8],
        tg_mem: &[usize],
    ) -> Result<Pipeline, Error> {
        let p = self.ctx.pipeline(kernel)?;
        self.enc.setComputePipelineState(&p);
        for (i, a) in args.iter().enumerate() {
            // SAFETY: the buffer outlives the command (`'a`), the shared code
            // checked the offset against it, and `run` waits for the GPU
            // before returning.
            unsafe { self.enc.setBuffer_offset_atIndex(Some(&a.raw().0), a.offset(), i) }
        }
        if !params.is_empty() {
            if params.len() > 4096 {
                return Err(Error::Device(format!(
                    "`{kernel}`: {} bytes of parameters",
                    params.len()
                )));
            }
            // SAFETY: valid for `params.len()` bytes; Metal copies them and
            // never writes through the pointer.
            unsafe {
                self.enc.setBytes_length_atIndex(
                    NonNull::from(params).cast::<c_void>(),
                    params.len(),
                    args.len(),
                )
            }
        }
        for (i, len) in tg_mem.iter().enumerate() {
            // SAFETY: the length is checked by Metal against the device limit
            // when the pipeline is dispatched.
            unsafe { self.enc.setThreadgroupMemoryLength_atIndex(*len, i) };
        }
        Ok(p)
    }

    /// Dispatches `threads` threads in groups of `group` (none when zero).
    fn threads(&self, threads: usize, group: usize) {
        if threads > 0 {
            self.enc.dispatchThreads_threadsPerThreadgroup(size1(threads), size1(group));
        }
    }
}

impl<'a> Encoder<'a, Metal> for MetalCommand<'a> {
    fn dispatch(
        &mut self,
        kernel: &str,
        threads: usize,
        args: &[Arg<'a>],
        params: &[u8],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, params, &[])?;
        let tg = p.maxTotalThreadsPerThreadgroup().clamp(1, 256);
        self.threads(threads, tg);
        Ok(())
    }

    fn dispatch_with(
        &mut self,
        kernel: &str,
        threads: usize,
        group: usize,
        args: &[Arg<'a>],
        params: &[u8],
        tg_mem: &[usize],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, params, tg_mem)?;
        check_group(kernel, &p, group)?;
        self.threads(threads, group);
        Ok(())
    }

    fn dispatch_groups(
        &mut self,
        kernel: &str,
        groups: usize,
        group: usize,
        args: &[Arg<'a>],
        params: &[u8],
        tg_mem: &[usize],
    ) -> Result<(), Error> {
        let p = self.bind(kernel, args, params, tg_mem)?;
        check_group(kernel, &p, group)?;
        if groups > 0 {
            self.enc.dispatchThreadgroups_threadsPerThreadgroup(size1(groups), size1(group));
        }
        Ok(())
    }
}

/// Checks a threadgroup size against the pipeline's limit.
fn check_group(kernel: &str, p: &Pipeline, group: usize) -> Result<(), Error> {
    let max = p.maxTotalThreadsPerThreadgroup();
    if group == 0 || group > max {
        return Err(Error::Device(format!(
            "`{kernel}`: threadgroup of {group} threads (pipeline max {max})"
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
// The backend

/// The host state of the Metal backend.
static HOST: HostState<Metal> = HostState::new();

impl Backend for Metal {
    const NAME: &'static str = "metal";
    const ENV_PREFIX: &'static str = "MIDNIGHT_METAL";
    type Raw = MetalBuffer;
    type Cmd<'a> = MetalCommand<'a>;

    fn host() -> &'static HostState<Self> {
        &HOST
    }

    fn probe() -> Option<&'static DeviceInfo> {
        probe_device().map(|p| &p.neutral)
    }

    fn context() -> Result<&'static Self, Error> {
        CTX.get_or_init(make_context).as_ref().map_err(|e| Error::NoDevice(e.clone()))
    }

    fn unified(&self) -> bool {
        true
    }

    fn budget(&self) -> (u64, u64) {
        (
            self.device.currentAllocatedSize() as u64,
            self.device.recommendedMaxWorkingSetSize(),
        )
    }

    fn max_buffer_len(&self) -> u64 {
        self.device.maxBufferLength() as u64
    }

    fn alloc_raw(&self, len: usize) -> Option<(MetalBuffer, bool)> {
        let raw = self
            .device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)?;
        debug_assert_zero_filled(&raw);
        self.keep(&raw);
        // Metal documents new buffers as zero-filled
        Some((MetalBuffer(raw), true))
    }

    fn run<'a, R>(
        &'a self,
        f: impl FnOnce(&mut MetalCommand<'a>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error> {
        objc2::rc::autoreleasepool(|_| {
            let cb = self
                .queue
                .commandBuffer()
                .ok_or_else(|| Error::Device("commandBuffer returned nil".into()))?;
            let enc = cb
                .computeCommandEncoder()
                .ok_or_else(|| Error::Device("computeCommandEncoder returned nil".into()))?;
            let mut cmd = MetalCommand {
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

    fn max_group(&self, kernel: &str) -> Result<usize, Error> {
        Ok(self.pipeline(kernel)?.maxTotalThreadsPerThreadgroup())
    }

    fn subgroup_width(&self, kernel: &str) -> Result<usize, Error> {
        Ok(self.pipeline(kernel)?.threadExecutionWidth())
    }
}
