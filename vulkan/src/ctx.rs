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

//! The Vulkan backend of `midnight-accel`: the device of
//! [`mod@crate::device`] and its arithmetic variant, made at first use
//! ([`context`]).
//!
//! A kernel's push-constant block holds the device addresses of its buffers
//! in list order (each with its argument's byte offset), then its parameter
//! block, then, for the elementwise NTT kernels, the dispatch's element count.
//! The quotient numerator's parameter block does not fit in push constants:
//! it is copied into a host-visible buffer of the run, whose address ends the
//! block. The workgroup width and the NTT pass's shared memory are
//! specialization constants. `msm_ones` runs in subgroups of 32 where the
//! device allows, else of the device's fixed width, and reduces Metal's runs
//! of 32 across narrower subgroups through shared memory. On Qualcomm Adreno
//! (`vendor_id == 0x5143`), `msm_ones` reduces entirely through shared memory
//! (`ONES_LANES = 1`), and `msm_accumulate`, `msm_integrate`, and `msm_ones`
//! are capped to 128-thread workgroups.
//!
//! On a discrete device, device buffers are unmapped: the shared code reaches
//! them through staging shadows ([`Backend::alloc_staging`],
//! [`Encoder::copy`]), and they are zeroed and read back on the GPU.

use std::{
    cell::Cell,
    ptr::NonNull,
    sync::{Mutex, OnceLock},
};

use midnight_accel::{Backend, DeviceInfo, Encoder, HostState, RawBuffer, Tuning, ctx as acc};

use crate::device::{self, Commands, Context, Grid, Options, Params, PipelineOptions, Variant};

/// The arithmetic variant: `64` or `32`.
pub const ARITH_ENV: &str = "MIDNIGHT_VULKAN_ARITH";
/// `1`: also take a CPU implementation (lavapipe). Test only.
pub const ALLOW_CPU_ENV: &str = "MIDNIGHT_VULKAN_ALLOW_CPU";
/// The device to use, by enumeration index or by text its name contains.
pub const DEVICE_ENV: &str = "MIDNIGHT_VULKAN_DEVICE";
/// `1`: treat the device as discrete. Test only.
pub const DISCRETE_ENV: &str = "MIDNIGHT_VULKAN_DISCRETE";

midnight_accel::export_ctx!(Vulkan, "MIDNIGHT_VULKAN");

/// The Vulkan backend: the process-wide device, its arithmetic variant and
/// the buffers of large parameter blocks.
pub struct Vulkan {
    dev: Context,
    variant: Variant,
    neutral: DeviceInfo,
    /// The constants of the device's class.
    tuning: &'static Tuning,
    /// Idle host-visible buffers for parameter blocks read by address.
    slabs: Mutex<Vec<device::Buffer>>,
}

impl std::fmt::Debug for Vulkan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vulkan")
            .field("device", &self.dev.info().name)
            .field("variant", &self.variant)
            .field("unified", &self.dev.info().unified_memory)
            .finish_non_exhaustive()
    }
}

impl Vulkan {
    /// The device.
    pub fn device(&self) -> &Context {
        &self.dev
    }

    /// The arithmetic variant of every pipeline.
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// A buffer for parameter blocks of at least `bytes` bytes: an idle one,
    /// else a new one.
    fn take_slab(&self, bytes: usize) -> Result<device::Buffer, Error> {
        let mut idle = self.slabs.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(i) = idle.iter().position(|b| b.len() >= bytes) {
            return Ok(idle.swap_remove(i));
        }
        drop(idle);
        self.dev.staging(bytes.max(SLAB_BYTES)).map_err(err)
    }

    /// Returns the buffers of parameter blocks of a completed run.
    fn put_slabs(&self, slabs: Vec<Slab>) {
        let mut idle = self.slabs.lock().unwrap_or_else(|e| e.into_inner());
        for s in slabs {
            if idle.len() < 4 {
                idle.push(s.buf);
            }
        }
    }
}

static CTX: OnceLock<Result<Vulkan, String>> = OnceLock::new();

/// The backend, if it was made.
fn backend() -> Option<&'static Vulkan> {
    CTX.get().and_then(|r| r.as_ref().ok())
}

/// The backend, made at the first call; `None` when it could not be made.
fn probe_device() -> Option<&'static Vulkan> {
    CTX.get_or_init(make_context).as_ref().ok()
}

/// The device's properties. `None`: no usable device. Makes the device (not
/// the pipelines) at the first call.
pub fn probe() -> Option<&'static device::VulkanInfo> {
    probe_device().map(|v| v.dev.info())
}

/// Why the context could not be made, if it was tried and failed.
pub fn context_error() -> Option<&'static str> {
    match CTX.get() {
        Some(Err(e)) => Some(e),
        _ => None,
    }
}

/// Whether the environment variable `name` is `1`.
fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

/// The arithmetic variant: [`ARITH_ENV`], else A where the device has
/// `shaderInt64` and `shaderInt16`.
fn choose_variant(info: &device::VulkanInfo) -> Result<Variant, String> {
    let has_64 = info.shader_int64 && info.shader_int16;
    match std::env::var(ARITH_ENV).as_deref() {
        Ok("64") if has_64 => Ok(Variant::Int64),
        Ok("64") => Err(format!("{ARITH_ENV}=64 on a device without shaderInt64/shaderInt16")),
        Ok("32") => Ok(Variant::Int32),
        Ok(v) => Err(format!("{ARITH_ENV}={v}: expected 64 or 32")),
        Err(_) if has_64 => Ok(Variant::Int64),
        Err(_) => Ok(Variant::Int32),
    }
}

fn make_context() -> Result<Vulkan, String> {
    let options = Options {
        allow_cpu: env_on(ALLOW_CPU_ENV),
        discrete: env_on(DISCRETE_ENV),
        select: std::env::var(DEVICE_ENV).ok().filter(|s| !s.trim().is_empty()),
    };
    let dev = Context::open(options).map_err(|e| e.to_string())?;
    let variant = choose_variant(dev.info())?;
    let info = dev.info();
    let neutral = DeviceInfo {
        name: info.name.clone(),
        unified_memory: info.unified_memory,
        memory_budget: dev.budget().1,
        max_buffer_length: info.max_buffer_len(),
        max_threads_per_group: max_group(info) as u64,
    };
    let tuning = crate::tuning::for_device(info);
    Ok(Vulkan {
        dev,
        variant,
        neutral,
        tuning,
        slabs: Mutex::new(Vec::new()),
    })
}

/// The most invocations of one workgroup.
fn max_group(info: &device::VulkanInfo) -> usize {
    info.max_compute_work_group_invocations.min(info.max_compute_work_group_size[0]) as usize
}

/// A device error as the shared code's.
fn err(e: device::Error) -> Error {
    match e {
        device::Error::OutOfMemory => Error::Memory(e.to_string()),
        device::Error::Loader(_) | device::Error::NoDevice | device::Error::NoSpirv => {
            Error::NoDevice(e.to_string())
        }
        _ => Error::Device(e.to_string()),
    }
}

/// A Vulkan buffer: mapped for the host on a unified device and for staging,
/// device-local and unmapped otherwise.
pub struct VulkanBuffer(device::Buffer);

impl std::fmt::Debug for VulkanBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("VulkanBuffer").field(&self.0).finish()
    }
}

impl VulkanBuffer {
    /// The device buffer.
    pub fn buffer(&self) -> &device::Buffer {
        &self.0
    }
}

/// The device of a buffer that has no mapping (one exists: it made it).
fn device_of_buffers() -> Result<&'static Context, Error> {
    backend()
        .map(|v| &v.dev)
        .ok_or_else(|| Error::NoDevice("no Vulkan context".into()))
}

impl RawBuffer for VulkanBuffer {
    fn len(&self) -> usize {
        self.0.len()
    }

    fn id(&self) -> u64 {
        self.0.device_address()
    }

    fn host_ptr(&self) -> Option<NonNull<u8>> {
        self.0.mapped_ptr()
    }

    fn host_cached(&self) -> bool {
        self.0.host_cached()
    }

    fn zero_all(&self) -> Result<(), Error> {
        if let Some(p) = self.0.mapped_ptr() {
            acc::host_zero(p, self.0.len());
            return Ok(());
        }
        device_of_buffers()?.fill(&self.0, 0, None, 0).map_err(err)
    }

    fn read_back_zero(&self) -> ReadBack {
        if let Some(p) = self.0.mapped_ptr() {
            return if acc::host_reads_zero(p, self.0.len()) {
                ReadBack::Zero
            } else {
                ReadBack::NonZero
            };
        }
        let Ok(dev) = device_of_buffers() else {
            return ReadBack::Unreadable;
        };
        let zero = dev.read_chunks(&self.0, |part| {
            acc::host_reads_zero(NonNull::from(part).cast(), part.len())
        });
        match zero {
            Ok(true) => ReadBack::Zero,
            Ok(false) => ReadBack::NonZero,
            Err(_) => ReadBack::Unreadable,
        }
    }

    fn write_test_byte(&self) -> bool {
        if let Some(p) = self.0.mapped_ptr() {
            // SAFETY: the middle byte is inside the mapping; the caller has
            // exclusive use of the buffer.
            unsafe { std::ptr::write_volatile(p.as_ptr().add(self.0.len() / 2), 0xa5) };
            return true;
        }
        let at = (self.0.len() / 2) & !3;
        device_of_buffers().is_ok_and(|d| d.fill(&self.0, at, Some(4), 0xa5a5_a5a5).is_ok())
    }
}

/// The bytes of a buffer of parameter blocks.
const SLAB_BYTES: usize = 4096;
/// The alignment of a parameter block in its buffer (the kernels'
/// `buffer_reference_align`).
const SLAB_ALIGN: usize = 16;
/// The workgroup width of [`Encoder::dispatch`] (at most).
const DISPATCH_GROUP: usize = 256;
/// The invocations of the runs the MSM kernels reduce (Metal's SIMD-group
/// width).
const MSM_RUN: u32 = 32;

/// The subgroup width `msm_ones` runs with: [`MSM_RUN`] where the device lets
/// a pipeline require it, else the device's fixed width when it is a power
/// of two. `None`: neither.
fn ones_subgroup(info: &device::VulkanInfo) -> Option<u32> {
    let forced = ONES_WIDTH.load(std::sync::atomic::Ordering::Relaxed);
    if forced != 0 {
        return info.takes_subgroup(forced).then_some(forced);
    }
    if info.takes_subgroup(MSM_RUN) {
        return Some(MSM_RUN);
    }
    let fixed = info.subgroup_size;
    (fixed.is_power_of_two() && info.takes_subgroup(fixed)).then_some(fixed)
}

/// The subgroup width `msm_ones` runs with instead of [`ones_subgroup`]'s
/// choice (0: that choice). Test support.
static ONES_WIDTH: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Makes `msm_ones` run in subgroups of `width` invocations, which the
/// device must let a pipeline require: the path of a device whose subgroups
/// are wider than 32; `None`: the default. Returns whether the device takes
/// the width. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_ones_width(width: Option<u32>) -> bool {
    let ok = width.is_none_or(|w| probe().is_some_and(|i| i.takes_subgroup(w)));
    if ok {
        ONES_WIDTH.store(width.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    }
    ok
}

/// The subgroup lanes `msm_ones` reduces with shuffles, below its subgroup
/// width (0: the subgroup width up to [`MSM_RUN`]). Test support.
static ONES_LANES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Makes `msm_ones` reduce with shuffles over `lanes` invocations (a power
/// of two up to its subgroup width and 32) and the rest through shared
/// memory, the path of a device with narrow subgroups; `None`: the default.
/// Returns whether the device's width takes `lanes`. Test support.
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub fn set_ones_lanes(lanes: Option<u32>) -> bool {
    let width = probe().and_then(ones_subgroup).unwrap_or(0);
    let ok = lanes.is_none_or(|l| l.is_power_of_two() && l <= width.min(MSM_RUN));
    if ok {
        ONES_LANES.store(lanes.unwrap_or(0), std::sync::atomic::Ordering::Relaxed);
    }
    ok
}

/// What a kernel takes besides its buffer addresses and parameter block in
/// the push-constant block.
#[derive(Clone, Copy, Debug, Default)]
struct Shape {
    /// The dispatch's element count follows the parameter block (the
    /// elementwise NTT kernels, which Metal bounds by the grid).
    count: bool,
    /// The parameter block is read from a buffer whose address ends the
    /// push-constant block.
    params_by_address: bool,
    /// `msm_ones`: runs in pinned subgroups ([`ones_subgroup`]), with the
    /// lanes it reduces by shuffles in specialization constant 1.
    ones: bool,
    /// Shared-memory block 0 is specialization constant 1, in words.
    shared_words: bool,
}

/// The [`Shape`] of `kernel`.
fn shape(kernel: &str) -> Shape {
    match kernel {
        "ntt_pass" => Shape {
            shared_words: true,
            ..Shape::default()
        },
        "ntt_scale" | "ntt_scale_copy" | "ntt_distribute_zeta" | "ntt_mul_periodic" => Shape {
            count: true,
            ..Shape::default()
        },
        "quotient_numerator" => Shape {
            params_by_address: true,
            ..Shape::default()
        },
        "msm_ones" => Shape {
            ones: true,
            ..Shape::default()
        },
        _ => Shape::default(),
    }
}

/// A buffer of parameter blocks used by one run, filled up to `used`.
struct Slab {
    buf: device::Buffer,
    used: usize,
}

/// The command buffer being recorded in [`Vulkan::run`].
pub struct VulkanCommand<'a> {
    vk: &'a Vulkan,
    cmds: Commands<'a>,
    slabs: Vec<Slab>,
}

impl std::fmt::Debug for VulkanCommand<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VulkanCommand").finish_non_exhaustive()
    }
}

/// The words of a parameter block.
fn words(kernel: &str, params: &[u8]) -> Result<Vec<u32>, Error> {
    if !params.len().is_multiple_of(4) {
        return Err(Error::Device(format!(
            "`{kernel}`: a parameter block of {} bytes",
            params.len()
        )));
    }
    Ok(params.as_chunks::<4>().0.iter().map(|w| u32::from_ne_bytes(*w)).collect())
}

impl<'a> VulkanCommand<'a> {
    /// The device address of a copy of `params` in a buffer of the run.
    fn park(&mut self, params: &[u8]) -> Result<u64, Error> {
        let fits = |s: &Slab| s.used.next_multiple_of(SLAB_ALIGN) + params.len() <= s.buf.len();
        if !self.slabs.last().is_some_and(fits) {
            let buf = self.vk.take_slab(params.len())?;
            self.slabs.push(Slab { buf, used: 0 });
        }
        let s = self.slabs.last_mut().expect("a slab was just pushed");
        let at = s.used.next_multiple_of(SLAB_ALIGN);
        let p = s
            .buf
            .mapped_ptr()
            .ok_or_else(|| Error::Device("an unmapped parameter buffer".into()))?;
        // SAFETY: `at + params.len()` is inside the mapping; no device work
        // of this run has been submitted yet, and the buffer is not used by
        // any other run.
        unsafe { std::ptr::copy_nonoverlapping(params.as_ptr(), p.as_ptr().add(at), params.len()) };
        s.used = at + params.len();
        Ok(s.buf.device_address() + at as u64)
    }

    /// Records `kernel` over `grid` in workgroups of `group`.
    fn encode(
        &mut self,
        kernel: &str,
        grid: Grid,
        group: usize,
        args: &[Arg<'a>],
        params: &[u8],
        shared: &[usize],
    ) -> Result<(), Error> {
        let vk = self.vk;
        let info = vk.dev.info();
        let sh = shape(kernel);
        let max = max_group(info);
        if group == 0 || group > max {
            return Err(Error::Device(format!(
                "`{kernel}`: workgroup of {group} threads (device max {max})"
            )));
        }
        let mut spec = [0u32; device::SPEC_CONSTANTS];
        match (sh.shared_words, shared) {
            (_, []) => {}
            (true, [bytes]) => {
                if *bytes > info.max_compute_shared_memory_size as usize || bytes % 4 != 0 {
                    return Err(Error::Device(format!(
                        "`{kernel}`: {bytes} bytes of shared memory (device max {})",
                        info.max_compute_shared_memory_size
                    )));
                }
                spec[0] = (*bytes / 4) as u32;
            }
            _ => {
                return Err(Error::Device(format!(
                    "`{kernel}` takes no shared-memory blocks {shared:?}"
                )));
            }
        }
        let mut subgroup = None;
        if sh.ones {
            let width = ones_subgroup(info).ok_or_else(|| {
                Error::Device(format!("`{kernel}`: no power-of-two subgroup width"))
            })?;
            // Qualcomm Adreno (vendor 0x5143): `subgroupShuffleDown` over the
            // 48 limbs of `G1Xyzz` in `msm_ones` corrupts partial sums across
            // lanes. Fall back to the shared-memory reduction (`ONES_LANES = 1`),
            // which stages 64 `G1Xyzz` records (12 KiB) per pass.
            let lanes = match ONES_LANES.load(std::sync::atomic::Ordering::Relaxed) {
                0 if info.vendor_id == 0x5143 => 1,
                0 => width.min(MSM_RUN),
                l => l,
            };
            if !lanes.is_power_of_two()
                || lanes > width.min(MSM_RUN)
                || !group.is_multiple_of(width.max(MSM_RUN) as usize)
            {
                return Err(Error::Device(format!(
                    "`{kernel}`: workgroup of {group} in subgroups of {width}, {lanes} lanes"
                )));
            }
            subgroup = Some(width);
            spec[0] = lanes;
        }
        let options = PipelineOptions {
            workgroup: group as u32,
            subgroup,
            spec,
        };
        let pipeline = vk.dev.pipeline(kernel, vk.variant, options).map_err(err)?;

        let mut push = Params::new();
        for a in args {
            push = push.addr_at(a.raw().buffer(), a.offset());
        }
        if sh.params_by_address {
            let at = self.park(params)?;
            push = push.u64s(&[at]);
        } else {
            push = push.words(&words(kernel, params)?);
        }
        if sh.count {
            let n = match grid {
                Grid::Threads(n) => u32::try_from(n).ok(),
                Grid::Groups(_) => None,
            };
            let n = n.ok_or_else(|| Error::Device(format!("`{kernel}`: element count")))?;
            push = push.u32(n);
        }
        self.cmds.dispatch_shared(&pipeline, &push, grid).map_err(err)
    }
}

impl<'a> Encoder<'a, Vulkan> for VulkanCommand<'a> {
    fn copy(
        &mut self,
        src: &'a VulkanBuffer,
        dst: &'a VulkanBuffer,
        bytes: usize,
    ) -> Result<(), Error> {
        self.cmds.copy(&src.0, &dst.0, bytes).map_err(err)
    }

    fn dispatch(
        &mut self,
        kernel: &str,
        threads: usize,
        args: &[Arg<'a>],
        params: &[u8],
    ) -> Result<(), Error> {
        // `quotient_numerator` keeps its hot program scratch slots in a fixed
        // 64-thread workgroup shared-memory tile (16 KiB) for zero-DRAM-traffic
        // intermediates and full-wave occupancy on Adreno & Apple GPUs.
        let cap = if kernel == "quotient_numerator" {
            64
        } else {
            DISPATCH_GROUP
        };
        let group = cap.min(max_group(self.vk.dev.info()));
        self.encode(
            kernel,
            Grid::Threads(threads as u64),
            group,
            args,
            params,
            &[],
        )
    }

    fn dispatch_with(
        &mut self,
        kernel: &str,
        threads: usize,
        group: usize,
        args: &[Arg<'a>],
        params: &[u8],
        shared: &[usize],
    ) -> Result<(), Error> {
        self.encode(
            kernel,
            Grid::Threads(threads as u64),
            group,
            args,
            params,
            shared,
        )
    }

    fn dispatch_groups(
        &mut self,
        kernel: &str,
        groups: usize,
        group: usize,
        args: &[Arg<'a>],
        params: &[u8],
        shared: &[usize],
    ) -> Result<(), Error> {
        self.encode(
            kernel,
            Grid::Groups(groups as u64),
            group,
            args,
            params,
            shared,
        )
    }
}

thread_local! {
    /// Whether this thread is recording a [`Vulkan::run`].
    static IN_RUN: Cell<bool> = const { Cell::new(false) };
}

/// Marks this thread as recording a run until dropped.
struct Recording;

impl Recording {
    /// `None` when this thread already records one: the main command buffer
    /// is locked by it.
    fn start() -> Option<Recording> {
        (!IN_RUN.replace(true)).then_some(Recording)
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        IN_RUN.set(false);
    }
}

/// The host state of the Vulkan backend.
static HOST: HostState<Vulkan> = HostState::new();

impl Backend for Vulkan {
    const NAME: &'static str = "vulkan";
    const ENV_PREFIX: &'static str = "MIDNIGHT_VULKAN";
    const MSM_LENGTH_BINS: bool = true;
    type Raw = VulkanBuffer;
    type Cmd<'a> = VulkanCommand<'a>;
    /// The constants of the device's class ([`crate::tuning::for_device`]),
    /// the default ones when no device is used.
    fn tuning() -> &'static Tuning {
        static OFF: OnceLock<bool> = OnceLock::new();
        if *OFF.get_or_init(|| std::env::var("MIDNIGHT_VULKAN").as_deref() == Ok("off")) {
            return &crate::tuning::TUNING;
        }
        probe_device().map_or(&crate::tuning::TUNING, |v| v.tuning)
    }

    fn host() -> &'static HostState<Self> {
        &HOST
    }

    fn probe() -> Option<&'static DeviceInfo> {
        probe_device().map(|v| &v.neutral)
    }

    fn context() -> Result<&'static Self, Error> {
        match CTX.get_or_init(make_context) {
            Ok(v) if v.dev.failed() => Err(Error::NoDevice("the Vulkan device was lost".into())),
            Ok(v) => Ok(v),
            Err(e) => Err(Error::NoDevice(e.clone())),
        }
    }

    fn unified(&self) -> bool {
        self.dev.info().unified_memory
    }

    fn budget(&self) -> (u64, u64) {
        self.dev.budget()
    }

    fn max_buffer_len(&self) -> u64 {
        self.dev.info().max_buffer_len()
    }

    fn alloc_raw(&self, len: usize) -> Option<(VulkanBuffer, bool)> {
        // new Vulkan memory is not known to be zero
        self.dev.alloc(len).ok().map(|b| (VulkanBuffer(b), false))
    }

    fn alloc_staging(&self, len: usize) -> Option<(VulkanBuffer, bool)> {
        self.dev.staging(len).ok().map(|b| (VulkanBuffer(b), false))
    }

    fn run<'a, R>(
        &'a self,
        f: impl FnOnce(&mut VulkanCommand<'a>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error> {
        let Some(_recording) = Recording::start() else {
            return Err(Error::Device(
                "a command buffer was started while recording another".into(),
            ));
        };
        let cmds = self.dev.commands().map_err(err)?;
        let mut enc = VulkanCommand {
            vk: self,
            cmds,
            slabs: Vec::new(),
        };
        let value = f(&mut enc);
        // what was recorded runs even when `f` failed: its copies keep the
        // shadows' states true
        let VulkanCommand { cmds, slabs, .. } = enc;
        let time = cmds.submit();
        self.put_slabs(slabs);
        let time = time.map_err(err)?;
        Ok((value?, time.as_nanos().min(u64::MAX as u128) as u64))
    }

    fn max_group(&self, kernel: &str) -> Result<usize, Error> {
        if !device::kernels().any(|k| k == kernel) {
            return Err(Error::Device(format!("no kernel `{kernel}`")));
        }
        let info = self.dev.info();
        let mut limit = max_group(info);
        // `msm_accumulate` (15 KiB) and `msm_integrate` (12 KiB) stage per-thread
        // `G1Xyzz` state in fixed 64-thread bank-conflict-free shared-memory tiles.
        if matches!(kernel, "msm_accumulate" | "msm_integrate") {
            limit = limit.min(64);
        }
        // Qualcomm Adreno (vendor 0x5143): cap high-register G1 kernels at 128
        // invocations (2 waves of 64), and cap radix-sort workgroups at 256 (4
        // waves of 64) to cut `group_exclusive_sum` barrier depth and reduce
        // shared-memory atomic collisions in `msm_sort_lower`.
        if info.vendor_id == 0x5143 {
            if matches!(kernel, "msm_accumulate_fallback" | "msm_ones") {
                limit = limit.min(128);
            } else if matches!(
                kernel,
                "msm_sort_count" | "msm_sort_scan" | "msm_sort_scatter" | "msm_sort_lower"
            ) {
                limit = limit.min(256);
            }
        }
        Ok(limit)
    }

    fn subgroup_width(&self, kernel: &str) -> Result<usize, Error> {
        if !device::kernels().any(|k| k == kernel) {
            return Err(Error::Device(format!("no kernel `{kernel}`")));
        }
        // The MSM kernels' contract is Metal's SIMD-groups of 32: `msm_ones`
        // reduces runs of 32 invocations whatever its subgroup width, and no
        // other MSM kernel uses subgroups. So the width of the MSM kernels is
        // that of the runs, wherever `msm_ones` has a subgroup width.
        let info = self.dev.info();
        let msm = kernel.starts_with("msm_") && !kernel.starts_with("msm_test");
        Ok(if msm && ones_subgroup(info).is_some() {
            MSM_RUN as usize
        } else {
            info.subgroup_size as usize
        })
    }
}
