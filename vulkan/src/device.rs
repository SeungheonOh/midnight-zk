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

//! The Vulkan device under [`crate::ctx::Vulkan`]: one compute queue, the
//! embedded pipelines, buffers, and command buffers.
//!
//! A kernel binds no descriptor: it reads one push-constant block
//! ([`Params`]), and every pipeline shares that one-range layout. The
//! commands of one [`Commands`] run in order, each seeing the memory writes of
//! those recorded before it.
//!
//! On a unified device every buffer is mapped for the host; otherwise device
//! buffers are device-local and the host reaches them through staging copies.
//! New memory is not known to be zero.
//!
//! [`Context::commands`] holds the main command buffer's lock until it is
//! submitted, and every submission holds the queue lock until its fence.
//! [`Context::fill`] and [`Context::read_chunks`] use a command buffer of
//! their own, so they can run from any thread while a [`Commands`] is being
//! recorded. `VK_ERROR_DEVICE_LOST` latches the context as failed: every later
//! call returns [`Error::Failed`].

use std::{
    collections::HashMap,
    ffi::{CStr, c_char},
    fmt,
    io::Cursor,
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use ash::vk;

/// The environment variable naming the Vulkan loader library to load
/// instead of the system's (`libvulkan.so.1`, `libvulkan.dylib`,
/// `vulkan-1.dll`).
pub const LOADER_ENV: &str = "MIDNIGHT_VULKAN_LOADER";
/// The bytes of push constants a kernel's params block may take: the
/// minimum `maxPushConstantsSize` Vulkan guarantees.
pub const MAX_PARAMS_BYTES: u32 = 128;
/// The specialization constants after the workgroup width (ids `1..=3`).
pub const SPEC_CONSTANTS: usize = 3;
/// The workgroup width of [`PipelineOptions::default`].
pub const DEFAULT_WORKGROUP: u32 = 256;
/// The largest staging buffer of one read-back step.
const READ_CHUNK: usize = 32 << 20;

/// An embedded kernel: its SPIR-V for both arithmetic variants.
pub(crate) struct Shader {
    pub(crate) name: &'static str,
    pub(crate) int64: &'static [u8],
    pub(crate) int32: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/shaders.rs"));

/// The names of the embedded kernels.
pub fn kernels() -> impl Iterator<Item = &'static str> {
    SHADERS.iter().map(|s| s.name)
}

/// The arithmetic variant of a kernel (shaders/include/common.glsl).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// Variant A: 32-bit limbs with 64-bit accumulators; needs
    /// `shaderInt64`.
    Int64,
    /// Variant B: 32-bit words only (`umulExtended`, `uaddCarry`,
    /// `usubBorrow`); the SPIR-V declares no `Int64` capability.
    Int32,
}

/// A failure of the device or of a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The Vulkan loader library could not be loaded.
    Loader(String),
    /// No Vulkan 1.3 GPU with a compute queue and `bufferDeviceAddress`.
    NoDevice,
    /// The crate was built without SPIR-V (the shader tools were missing).
    NoSpirv,
    /// The device lacks a feature, extension or limit the call needs.
    Unsupported(String),
    /// Host or device memory is exhausted, or no memory type fits.
    OutOfMemory,
    /// `VK_ERROR_DEVICE_LOST`; the context is now failed.
    DeviceLost,
    /// The context failed earlier (device lost).
    Failed,
    /// Any other Vulkan error.
    Vulkan {
        /// The Vulkan function.
        call: &'static str,
        /// Its result.
        result: vk::Result,
    },
    /// An argument the context cannot serve.
    Invalid(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Loader(e) => write!(f, "the Vulkan loader could not be loaded: {e}"),
            Error::NoDevice => write!(
                f,
                "no Vulkan 1.3 GPU with a compute queue and bufferDeviceAddress"
            ),
            Error::NoSpirv => write!(
                f,
                "midnight-vulkan was built without SPIR-V: install glslang (or shaderc) and \
                 SPIRV-Tools and rebuild"
            ),
            Error::Unsupported(what) => write!(f, "unsupported by the device: {what}"),
            Error::OutOfMemory => write!(f, "out of memory"),
            Error::DeviceLost => write!(f, "the Vulkan device was lost"),
            Error::Failed => write!(f, "the Vulkan context failed earlier"),
            Error::Vulkan { call, result } => write!(f, "{call} failed: {result}"),
            Error::Invalid(what) => write!(f, "invalid call: {what}"),
        }
    }
}

impl std::error::Error for Error {}

/// A result of this module.
pub type Result<T> = std::result::Result<T, Error>;

/// The device as its properties describe it.
#[derive(Clone, Debug, PartialEq)]
pub struct VulkanInfo {
    /// `deviceName`.
    pub name: String,
    /// `vendorID`.
    pub vendor_id: u32,
    /// `deviceID`.
    pub device_id: u32,
    /// `deviceType`.
    pub device_type: vk::PhysicalDeviceType,
    /// `apiVersion` as (major, minor, patch).
    pub api_version: (u32, u32, u32),
    /// `driverName` and `driverInfo`.
    pub driver: String,
    /// The default subgroup size (`subgroupSize`).
    pub subgroup_size: u32,
    /// `minSubgroupSize`.
    pub min_subgroup_size: u32,
    /// `maxSubgroupSize`.
    pub max_subgroup_size: u32,
    /// A compute pipeline may require a subgroup size
    /// (`subgroupSizeControl` and compute in `requiredSubgroupSizeStages`).
    pub subgroup_size_control: bool,
    /// `computeFullSubgroups`.
    pub compute_full_subgroups: bool,
    /// `maxComputeSharedMemorySize` in bytes.
    pub max_compute_shared_memory_size: u32,
    /// `maxPushConstantsSize` in bytes.
    pub max_push_constants_size: u32,
    /// `maxComputeWorkGroupCount`.
    pub max_compute_work_group_count: [u32; 3],
    /// `maxComputeWorkGroupSize`.
    pub max_compute_work_group_size: [u32; 3],
    /// `maxComputeWorkGroupInvocations`.
    pub max_compute_work_group_invocations: u32,
    /// `minStorageBufferOffsetAlignment` in bytes.
    pub min_storage_buffer_offset_alignment: u64,
    /// `maxStorageBufferRange` in bytes.
    pub max_storage_buffer_range: u32,
    /// `maxMemoryAllocationSize` in bytes.
    pub max_memory_allocation_size: u64,
    /// `maxBufferSize` in bytes.
    pub max_buffer_size: u64,
    /// `shaderInt64` (variant A).
    pub shader_int64: bool,
    /// `shaderInt16` (16-bit integer types in shaders).
    pub shader_int16: bool,
    /// `bufferDeviceAddress` (required).
    pub buffer_device_address: bool,
    /// `timestampValidBits` of the compute queue (0: no timestamps).
    pub timestamp_valid_bits: u32,
    /// `timestampPeriod`: nanoseconds per timestamp tick.
    pub timestamp_period: f32,
    /// `VK_EXT_memory_budget` is enabled.
    pub memory_budget_ext: bool,
    /// The size of the largest device-local heap in bytes.
    pub device_local_heap: u64,
    /// Device memory is mapped for the host: an integrated GPU or a CPU
    /// implementation with host-visible, coherent device memory, unless
    /// [`Options::discrete`].
    pub unified_memory: bool,
}

impl VulkanInfo {
    /// Whether a compute pipeline can run in subgroups of exactly `size`
    /// invocations with full subgroups: requested where the device lets
    /// compute pipelines choose, else the device's fixed size.
    pub fn takes_subgroup(&self, size: u32) -> bool {
        let fixed = self.subgroup_size == size && self.max_subgroup_size == size;
        let chosen = self.subgroup_size_control
            && (self.min_subgroup_size..=self.max_subgroup_size).contains(&size);
        self.compute_full_subgroups && (fixed || chosen)
    }

    /// The largest buffer the context makes, in bytes.
    pub fn max_buffer_len(&self) -> u64 {
        self.max_memory_allocation_size.min(self.max_buffer_size)
    }
}

/// How [`Context::open`] picks and uses the device.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Options {
    /// Also accept a CPU implementation (lavapipe, SwiftShader), ranked after
    /// every GPU. Test only: CI runners have no GPU.
    pub allow_cpu: bool,
    /// Treat the device as discrete: device buffers are never mapped for the
    /// host, which reaches them through staging copies. Test only: runs the
    /// path of a discrete GPU on a unified device.
    pub discrete: bool,
    /// Only the device of this index in the enumeration order (decimal), or
    /// those whose name contains this text (ASCII case ignored). `None`: any
    /// usable device.
    pub select: Option<String>,
}

/// Whether `select` names the device at `index` called `name`.
fn selects(select: &str, index: usize, name: &str) -> bool {
    match select.trim().parse::<usize>() {
        Ok(i) => i == index,
        Err(_) => name.to_ascii_lowercase().contains(&select.trim().to_ascii_lowercase()),
    }
}

/// The device, shared by the context, its buffers and its pipelines.
struct Shared {
    _entry: ash::Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    failed: AtomicBool,
    /// Bytes of device memory allocated by this context.
    allocated: AtomicU64,
    /// The queue lock: held from a submission to its fence. A buffer is also
    /// freed only outside that window: MoltenVK makes every buffer with a
    /// device address resident for each submission, so freeing one
    /// meanwhile, even one the submission never reads, faults the GPU
    /// (`VK_ERROR_DEVICE_LOST`).
    in_flight: Mutex<()>,
}

impl Shared {
    fn alive(&self) -> Result<()> {
        if self.failed.load(Ordering::Acquire) {
            Err(Error::Failed)
        } else {
            Ok(())
        }
    }

    /// Maps a Vulkan result; a lost device latches the context as failed.
    fn check<T>(&self, call: &'static str, r: std::result::Result<T, vk::Result>) -> Result<T> {
        r.map_err(|result| match result {
            vk::Result::ERROR_DEVICE_LOST => {
                self.failed.store(true, Ordering::Release);
                Error::DeviceLost
            }
            vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
                Error::OutOfMemory
            }
            result => Error::Vulkan { call, result },
        })
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        // SAFETY: the last owner; every buffer, pipeline and command object
        // of the device was destroyed before.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// The objects of one command buffer at a time.
struct Submit {
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// Two timestamps around the commands; null without timestamps.
    query: vk::QueryPool,
}

impl Submit {
    fn null(queue: vk::Queue) -> Submit {
        Submit {
            queue,
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            query: vk::QueryPool::null(),
        }
    }

    /// Creates the pool, command buffer, fence and, with `timestamps`, the
    /// query pool. On failure the objects made so far stay in `self` for
    /// [`Submit::destroy`].
    fn create(&mut self, s: &Shared, family: u32, timestamps: bool) -> Result<()> {
        let d = &s.device;
        let pool = vk::CommandPoolCreateInfo::default().queue_family_index(family);
        // SAFETY: valid device and create infos for every call below.
        self.pool = s.check("vkCreateCommandPool", unsafe {
            d.create_command_pool(&pool, None)
        })?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        self.cmd = s.check("vkAllocateCommandBuffers", unsafe {
            d.allocate_command_buffers(&alloc)
        })?[0];
        self.fence = s.check("vkCreateFence", unsafe {
            d.create_fence(&vk::FenceCreateInfo::default(), None)
        })?;
        if timestamps {
            let q = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(2);
            self.query = s.check("vkCreateQueryPool", unsafe {
                d.create_query_pool(&q, None)
            })?;
        }
        Ok(())
    }

    /// Destroys the objects (null handles are skipped by Vulkan).
    fn destroy(&self, d: &ash::Device) {
        // SAFETY: the device is idle; every handle is null or owned here.
        unsafe {
            d.destroy_query_pool(self.query, None);
            d.destroy_fence(self.fence, None);
            d.destroy_command_pool(self.pool, None);
        }
    }
}

/// Environment variable naming an optional file to persist the driver's
/// `VkPipelineCache` across process restarts (eliminates repeated cold shader
/// compilation on drivers such as Qualcomm Adreno).
pub const PIPELINE_CACHE_ENV: &str = "MIDNIGHT_VULKAN_PIPELINE_CACHE";

/// The pipeline cache key.
type Key = (&'static str, Variant, PipelineOptions);

/// A Vulkan device with a compute queue.
pub struct Context {
    shared: Arc<Shared>,
    info: VulkanInfo,
    pipeline_layout: vk::PipelineLayout,
    pipeline_cache: vk::PipelineCache,
    pipeline_cache_path: Option<std::path::PathBuf>,
    /// The main command buffer ([`Context::commands`]).
    submit: Mutex<Submit>,
    /// The command buffer of fills and read-backs.
    aux: Mutex<Submit>,
    pipelines: Mutex<HashMap<Key, Arc<Pipeline>>>,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context").field("info", &self.info).finish_non_exhaustive()
    }
}

fn c_string(s: std::result::Result<&CStr, std::ffi::FromBytesUntilNulError>) -> String {
    s.map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Maps a Vulkan result of instance or device creation.
fn init_error<T>(call: &'static str, r: std::result::Result<T, vk::Result>) -> Result<T> {
    r.map_err(|result| match result {
        vk::Result::ERROR_INCOMPATIBLE_DRIVER | vk::Result::ERROR_INITIALIZATION_FAILED => {
            Error::NoDevice
        }
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            Error::OutOfMemory
        }
        result => Error::Vulkan { call, result },
    })
}

/// The rank of a device type; lower is preferred, `None` is never used. A
/// CPU implementation is taken only with [`Options::allow_cpu`].
fn type_rank(t: vk::PhysicalDeviceType, allow_cpu: bool) -> Option<u32> {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => Some(0),
        vk::PhysicalDeviceType::INTEGRATED_GPU => Some(1),
        vk::PhysicalDeviceType::VIRTUAL_GPU => Some(2),
        vk::PhysicalDeviceType::CPU => allow_cpu.then_some(4),
        _ => Some(3),
    }
}

/// A candidate device: its handle, compute queue family and properties.
struct Candidate {
    physical: vk::PhysicalDevice,
    family: u32,
    info: VulkanInfo,
    portability_subset: bool,
}

type Mem = vk::MemoryPropertyFlags;

/// `(wanted, avoided)` memory flags, in order of preference.
type Preferences = &'static [(Mem, Mem)];

/// Device buffers of a unified device: mapped, preferring host-cached coherent
/// memory so CPU readbacks, copies, and SRS/key verification run at cached RAM
/// speed without GPU penalty.
const UNIFIED: Preferences = &[
    (
        Mem::from_raw(
            Mem::DEVICE_LOCAL.as_raw()
                | Mem::HOST_VISIBLE.as_raw()
                | Mem::HOST_COHERENT.as_raw()
                | Mem::HOST_CACHED.as_raw(),
        ),
        Mem::LAZILY_ALLOCATED,
    ),
    (
        Mem::from_raw(
            Mem::DEVICE_LOCAL.as_raw() | Mem::HOST_VISIBLE.as_raw() | Mem::HOST_COHERENT.as_raw(),
        ),
        Mem::LAZILY_ALLOCATED,
    ),
];
/// Device buffers of a discrete device: device memory the host does not see
/// first.
const DEVICE_ONLY: Preferences = &[
    (
        Mem::DEVICE_LOCAL,
        Mem::from_raw(Mem::HOST_VISIBLE.as_raw() | Mem::LAZILY_ALLOCATED.as_raw()),
    ),
    (Mem::DEVICE_LOCAL, Mem::LAZILY_ALLOCATED),
];
/// Staging buffers: host-visible and coherent, cached where available.
const STAGING: Preferences = &[
    (
        Mem::from_raw(
            Mem::HOST_VISIBLE.as_raw() | Mem::HOST_COHERENT.as_raw() | Mem::HOST_CACHED.as_raw(),
        ),
        Mem::LAZILY_ALLOCATED,
    ),
    (
        Mem::from_raw(Mem::HOST_VISIBLE.as_raw() | Mem::HOST_COHERENT.as_raw()),
        Mem::LAZILY_ALLOCATED,
    ),
];

impl Context {
    /// Loads the Vulkan loader ([`LOADER_ENV`]) and opens the preferred
    /// device: a discrete GPU, else an integrated one; CPU implementations
    /// only with [`Options::allow_cpu`].
    pub fn open(options: Options) -> Result<Context> {
        if !cfg!(midnight_vulkan_spirv) {
            return Err(Error::NoSpirv);
        }
        // SAFETY: loading the Vulkan loader runs its initialisers, which is
        // the documented use of the library.
        let entry = match std::env::var_os(LOADER_ENV) {
            Some(path) => unsafe { ash::Entry::load_from(&path) },
            None => unsafe { ash::Entry::load() },
        }
        .map_err(|e| Error::Loader(e.to_string()))?;
        // SAFETY: valid entry; no layer name.
        let exts = init_error("vkEnumerateInstanceExtensionProperties", unsafe {
            entry.enumerate_instance_extension_properties(None)
        })?;
        let has = |name: &CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
        let mut names: Vec<*const c_char> = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if has(vk::KHR_PORTABILITY_ENUMERATION_NAME) {
            names.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let app = vk::ApplicationInfo::default()
            .application_name(c"midnight-vulkan")
            .engine_name(c"midnight-vulkan")
            .api_version(vk::API_VERSION_1_3);
        let create = vk::InstanceCreateInfo::default()
            .application_info(&app)
            .enabled_extension_names(&names)
            .flags(flags);
        // SAFETY: the create info and the strings it points to outlive the
        // call.
        let instance = init_error("vkCreateInstance", unsafe {
            entry.create_instance(&create, None)
        })?;
        match Self::create_device(&instance, options) {
            Ok((device, memory, candidate)) => {
                Self::finish(entry, instance, device, memory, candidate)
            }
            Err(e) => {
                // SAFETY: on failure the instance has no child object.
                unsafe { instance.destroy_instance(None) };
                Err(e)
            }
        }
    }

    fn create_device(
        instance: &ash::Instance,
        options: Options,
    ) -> Result<(ash::Device, vk::PhysicalDeviceMemoryProperties, Candidate)> {
        // SAFETY: valid instance.
        let physicals = init_error("vkEnumeratePhysicalDevices", unsafe {
            instance.enumerate_physical_devices()
        })?;
        let mut best: Option<(u32, Candidate)> = None;
        for (i, p) in physicals.into_iter().enumerate() {
            let Some(c) = Self::candidate(instance, p, &options)? else {
                continue;
            };
            if options.select.as_deref().is_some_and(|s| !selects(s, i, &c.info.name)) {
                continue;
            }
            let rank = type_rank(c.info.device_type, options.allow_cpu).unwrap_or(u32::MAX);
            if best.as_ref().is_none_or(|(r, _)| rank < *r) {
                best = Some((rank, c));
            }
        }
        let (_, c) = best.ok_or_else(|| match &options.select {
            Some(s) => Error::Unsupported(format!("no usable device matches {s:?}")),
            None => Error::NoDevice,
        })?;

        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(c.info.buffer_device_address);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(c.info.subgroup_size_control)
            .compute_full_subgroups(c.info.compute_full_subgroups);
        let features = vk::PhysicalDeviceFeatures::default()
            .shader_int64(c.info.shader_int64)
            .shader_int16(c.info.shader_int16);
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(c.family)
            .queue_priorities(&priorities)];
        let mut exts: Vec<*const c_char> = Vec::new();
        if c.portability_subset {
            exts.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }
        if c.info.memory_budget_ext {
            exts.push(vk::EXT_MEMORY_BUDGET_NAME.as_ptr());
        }
        let create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&exts)
            .enabled_features(&features)
            .push_next(&mut v12)
            .push_next(&mut v13);
        // SAFETY: every enabled feature and extension is supported (queried
        // in `candidate`); the create info outlives the call.
        let device = init_error("vkCreateDevice", unsafe {
            instance.create_device(c.physical, &create, None)
        })?;
        // SAFETY: valid physical device.
        let memory = unsafe { instance.get_physical_device_memory_properties(c.physical) };
        Ok((device, memory, c))
    }

    /// The device `p` if it is usable: Vulkan 1.3, a GPU (or a CPU
    /// implementation when allowed), with a compute queue and
    /// `bufferDeviceAddress`.
    fn candidate(
        instance: &ash::Instance,
        p: vk::PhysicalDevice,
        options: &Options,
    ) -> Result<Option<Candidate>> {
        let mut v11 = vk::PhysicalDeviceVulkan11Properties::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Properties::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Properties::default();
        let mut props = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut v11)
            .push_next(&mut v12)
            .push_next(&mut v13);
        // SAFETY: valid physical device; the chain outlives the call.
        unsafe { instance.get_physical_device_properties2(p, &mut props) };
        let base = props.properties;
        if vk::api_version_major(base.api_version) < 1
            || (vk::api_version_major(base.api_version) == 1
                && vk::api_version_minor(base.api_version) < 3)
            || type_rank(base.device_type, options.allow_cpu).is_none()
        {
            return Ok(None);
        }
        // SAFETY: valid physical device.
        let families = unsafe { instance.get_physical_device_queue_family_properties(p) };
        let compute: Vec<(usize, &vk::QueueFamilyProperties)> = families
            .iter()
            .enumerate()
            .filter(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE) && f.queue_count > 0)
            .collect();
        // a universal queue first: the one every driver schedules best
        let Some(&(family, family_props)) = compute
            .iter()
            .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .or(compute.first())
        else {
            return Ok(None);
        };

        let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut f13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut features =
            vk::PhysicalDeviceFeatures2::default().push_next(&mut f12).push_next(&mut f13);
        // SAFETY: as above.
        unsafe { instance.get_physical_device_features2(p, &mut features) };
        let shader_int64 = features.features.shader_int64 == vk::TRUE;
        let shader_int16 = features.features.shader_int16 == vk::TRUE;
        // SAFETY: valid physical device.
        let exts = unsafe { instance.enumerate_device_extension_properties(p) };
        let exts = init_error("vkEnumerateDeviceExtensionProperties", exts)?;
        let has = |name: &CStr| exts.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
        let portability_subset = has(vk::KHR_PORTABILITY_SUBSET_NAME);
        // SAFETY: valid physical device.
        let memory = unsafe { instance.get_physical_device_memory_properties(p) };
        let heaps = &memory.memory_heaps[..memory.memory_heap_count as usize];
        let device_local_heap = heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .max()
            .unwrap_or(0);
        let host_mapped_device = memory.memory_types[..memory.memory_type_count as usize]
            .iter()
            .any(|t| UNIFIED.iter().any(|(want, _)| t.property_flags.contains(*want)));
        let shares_memory = matches!(
            base.device_type,
            vk::PhysicalDeviceType::INTEGRATED_GPU | vk::PhysicalDeviceType::CPU
        );

        let limits = base.limits;
        let v = base.api_version;
        let info = VulkanInfo {
            name: c_string(base.device_name_as_c_str()),
            vendor_id: base.vendor_id,
            device_id: base.device_id,
            device_type: base.device_type,
            api_version: (
                vk::api_version_major(v),
                vk::api_version_minor(v),
                vk::api_version_patch(v),
            ),
            driver: format!(
                "{} {}",
                c_string(v12.driver_name_as_c_str()),
                c_string(v12.driver_info_as_c_str())
            ),
            subgroup_size: v11.subgroup_size,
            min_subgroup_size: v13.min_subgroup_size,
            max_subgroup_size: v13.max_subgroup_size,
            subgroup_size_control: f13.subgroup_size_control == vk::TRUE
                && v13.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE),
            compute_full_subgroups: f13.compute_full_subgroups == vk::TRUE,
            max_compute_shared_memory_size: limits.max_compute_shared_memory_size,
            max_push_constants_size: limits.max_push_constants_size,
            max_compute_work_group_count: limits.max_compute_work_group_count,
            max_compute_work_group_size: limits.max_compute_work_group_size,
            max_compute_work_group_invocations: limits.max_compute_work_group_invocations,
            min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
            max_storage_buffer_range: limits.max_storage_buffer_range,
            max_memory_allocation_size: v11.max_memory_allocation_size,
            max_buffer_size: v13.max_buffer_size,
            shader_int64,
            shader_int16,
            buffer_device_address: f12.buffer_device_address == vk::TRUE,
            timestamp_valid_bits: family_props.timestamp_valid_bits,
            timestamp_period: limits.timestamp_period,
            memory_budget_ext: has(vk::EXT_MEMORY_BUDGET_NAME),
            device_local_heap,
            unified_memory: shares_memory && host_mapped_device && !options.discrete,
        };
        if !info.buffer_device_address {
            return Ok(None);
        }
        Ok(Some(Candidate {
            physical: p,
            family: family as u32,
            info,
            portability_subset,
        }))
    }

    /// Creates the command, query and layout objects.
    fn finish(
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        memory: vk::PhysicalDeviceMemoryProperties,
        c: Candidate,
    ) -> Result<Context> {
        let shared = Arc::new(Shared {
            _entry: entry,
            instance,
            physical: c.physical,
            device,
            memory,
            failed: AtomicBool::new(false),
            allocated: AtomicU64::new(0),
            in_flight: Mutex::new(()),
        });
        // From here `Shared` owns the device; objects created below are
        // destroyed by `Context::drop` or, on failure, by `Partial::drop`.
        // SAFETY: the family and index exist (one queue created).
        let queue = unsafe { shared.device.get_device_queue(c.family, 0) };
        let mut p = Partial {
            shared: &shared,
            submit: Submit::null(queue),
            aux: Submit::null(queue),
            pipeline_layout: vk::PipelineLayout::null(),
        };
        let s = &shared;
        let timestamps = c.info.timestamp_valid_bits > 0 && c.info.timestamp_period > 0.0;
        p.submit.create(s, c.family, timestamps)?;
        p.aux.create(s, c.family, false)?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(MAX_PARAMS_BYTES.min(c.info.max_push_constants_size))];
        let pl = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push);
        // SAFETY: valid device and create info.
        p.pipeline_layout = s.check("vkCreatePipelineLayout", unsafe {
            s.device.create_pipeline_layout(&pl, None)
        })?;

        let (submit, aux, pipeline_layout) = p.take();
        let pipeline_cache_path = std::env::var_os(PIPELINE_CACHE_ENV)
            .filter(|p| !p.is_empty())
            .map(std::path::PathBuf::from);
        let pipeline_cache = if let Some(ref path) = pipeline_cache_path {
            let initial = std::fs::read(path).unwrap_or_default();
            let ci = vk::PipelineCacheCreateInfo::default().initial_data(&initial);
            // SAFETY: valid device and create info; drivers validate and ignore
            // incompatible cache blobs per the Vulkan specification.
            unsafe { shared.device.create_pipeline_cache(&ci, None) }
                .or_else(|_| {
                    let empty = vk::PipelineCacheCreateInfo::default();
                    unsafe { shared.device.create_pipeline_cache(&empty, None) }
                })
                .unwrap_or(vk::PipelineCache::null())
        } else {
            vk::PipelineCache::null()
        };
        Ok(Context {
            shared: shared.clone(),
            info: c.info,
            pipeline_layout,
            pipeline_cache,
            pipeline_cache_path,
            submit: Mutex::new(submit),
            aux: Mutex::new(aux),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// The device's properties.
    pub fn info(&self) -> &VulkanInfo {
        &self.info
    }

    /// Whether a device loss latched the context as failed.
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Acquire)
    }

    /// `(used, total)` bytes of the largest device-local heap: the heap's
    /// usage and budget from `VK_EXT_memory_budget`, else the bytes this
    /// context allocated and three quarters of the heap.
    pub fn budget(&self) -> (u64, u64) {
        let s = &self.shared;
        let own = s.allocated.load(Ordering::Relaxed);
        let fallback = (own, self.info.device_local_heap / 4 * 3);
        if !self.info.memory_budget_ext {
            return fallback;
        }
        let mut b = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut b);
        // SAFETY: valid physical device; the extension is enabled; the chain
        // outlives the call.
        unsafe { s.instance.get_physical_device_memory_properties2(s.physical, &mut props) };
        let m = props.memory_properties;
        let heap = (0..m.memory_heap_count as usize)
            .filter(|h| m.memory_heaps[*h].flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .max_by_key(|h| m.memory_heaps[*h].size);
        match heap {
            Some(h) if b.heap_budget[h] > 0 => (b.heap_usage[h], b.heap_budget[h]),
            _ => fallback,
        }
    }

    fn lock(m: &Mutex<Submit>) -> MutexGuard<'_, Submit> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The pipelines created so far, in no particular order.
    pub fn pipelines(&self) -> Vec<Arc<Pipeline>> {
        let cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        cache.values().cloned().collect()
    }

    /// The compute pipeline of `kernel` in `variant`, created at first use.
    pub fn pipeline(
        &self,
        kernel: &str,
        variant: Variant,
        options: PipelineOptions,
    ) -> Result<Arc<Pipeline>> {
        self.shared.alive()?;
        let shader = SHADERS
            .iter()
            .find(|s| s.name == kernel)
            .ok_or_else(|| Error::Invalid(format!("no kernel {kernel}")))?;
        let key = (shader.name, variant, options);
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let p = Arc::new(self.create_pipeline(shader, variant, options)?);
        cache.insert(key, p.clone());
        self.save_pipeline_cache();
        Ok(p)
    }

    fn save_pipeline_cache(&self) {
        let Some(ref path) = self.pipeline_cache_path else {
            return;
        };
        if self.pipeline_cache == vk::PipelineCache::null() {
            return;
        }
        // SAFETY: valid device and pipeline cache handle; serialized by `self.pipelines`.
        let Ok(bytes) = (unsafe { self.shared.device.get_pipeline_cache_data(self.pipeline_cache) })
        else {
            return;
        };
        if bytes.is_empty() {
            return;
        }
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &bytes).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }

    fn create_pipeline(
        &self,
        shader: &Shader,
        variant: Variant,
        options: PipelineOptions,
    ) -> Result<Pipeline> {
        let info = &self.info;
        if variant == Variant::Int64 && !info.shader_int64 {
            return Err(Error::Unsupported("shaderInt64 (variant A)".into()));
        }
        let wg = options.workgroup;
        if wg == 0
            || wg > info.max_compute_work_group_size[0]
            || wg > info.max_compute_work_group_invocations
        {
            return Err(Error::Invalid(format!("workgroup width {wg}")));
        }
        // A device whose subgroup size cannot be chosen but is fixed at the
        // requested one (Apple GPUs: 32) needs no size request.
        let mut request_size = false;
        if let Some(s) = options.subgroup {
            if !s.is_power_of_two()
                || s < info.min_subgroup_size
                || s > info.max_subgroup_size
                || !wg.is_multiple_of(s)
            {
                return Err(Error::Invalid(format!(
                    "subgroup size {s} (device range {}..={}, workgroup {wg})",
                    info.min_subgroup_size, info.max_subgroup_size
                )));
            }
            if !info.takes_subgroup(s) {
                return Err(Error::Unsupported(format!(
                    "a compute subgroup size of {s} (subgroupSizeControl, computeFullSubgroups)"
                )));
            }
            request_size = info.subgroup_size_control;
        }
        let bytes = match variant {
            Variant::Int64 => shader.int64,
            Variant::Int32 => shader.int32,
        };
        let code = ash::util::read_spv(&mut Cursor::new(bytes))
            .map_err(|e| Error::Invalid(format!("SPIR-V of {}: {e}", shader.name)))?;
        let s = &self.shared;
        let d = &s.device;
        let module_info = vk::ShaderModuleCreateInfo::default().code(&code);
        // SAFETY: valid device; the SPIR-V was validated by spirv-opt.
        let module = s.check("vkCreateShaderModule", unsafe {
            d.create_shader_module(&module_info, None)
        })?;
        let mut entries =
            vec![vk::SpecializationMapEntry::default().constant_id(0).offset(0).size(4)];
        let mut data = wg.to_ne_bytes().to_vec();
        for (i, v) in options.spec.iter().enumerate() {
            if *v != 0 {
                entries.push(
                    vk::SpecializationMapEntry::default()
                        .constant_id(i as u32 + 1)
                        .offset(data.len() as u32)
                        .size(4),
                );
                data.extend_from_slice(&v.to_ne_bytes());
            }
        }
        let spec = vk::SpecializationInfo::default().map_entries(&entries).data(&data);
        let mut required = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(options.subgroup.unwrap_or(0));
        let mut stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main")
            .specialization_info(&spec);
        if options.subgroup.is_some() {
            stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS);
        }
        if request_size {
            stage = stage.push_next(&mut required);
        }
        let create = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.pipeline_layout)];
        let t0 = Instant::now();
        // SAFETY: valid device, module and layout; the create info outlives
        // the call.
        let r = unsafe { d.create_compute_pipelines(self.pipeline_cache, &create, None) };
        let creation = t0.elapsed();
        // SAFETY: the module is no longer needed once the pipeline exists.
        unsafe { d.destroy_shader_module(module, None) };
        let pipeline = s.check("vkCreateComputePipelines", r.map_err(|(_, e)| e))?[0];
        Ok(Pipeline {
            shared: s.clone(),
            pipeline,
            name: shader.name,
            variant,
            options,
            creation,
        })
    }

    /// The memory types of `bits` matching each preference in turn.
    fn memory_types(&self, bits: u32, preferences: Preferences) -> Vec<u32> {
        let m = &self.shared.memory;
        let mut out = Vec::new();
        for (want, avoid) in preferences {
            for i in 0..m.memory_type_count {
                let flags = m.memory_types[i as usize].property_flags;
                if bits & (1 << i) != 0
                    && flags.contains(*want)
                    && !flags.intersects(*avoid)
                    && !out.contains(&i)
                {
                    out.push(i);
                }
            }
        }
        out
    }

    /// A buffer of `len` bytes in the first memory type of `preferences`
    /// that has room, mapped for the host when `map` and the type is
    /// host-visible and coherent.
    fn create_buffer(&self, len: usize, preferences: Preferences, map: bool) -> Result<Buffer> {
        self.shared.alive()?;
        if len as u64 > self.info.max_buffer_len() {
            return Err(Error::OutOfMemory);
        }
        let s = &self.shared;
        let d = &s.device;
        let size = (len.max(4) as u64).next_multiple_of(4);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        let create = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        // SAFETY: valid device and create info.
        let buffer = s.check("vkCreateBuffer", unsafe { d.create_buffer(&create, None) })?;
        // SAFETY: valid buffer.
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let mut memory = Err(Error::OutOfMemory);
        let mut host = false;
        let mut cached = false;
        for t in self.memory_types(req.memory_type_bits, preferences) {
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(t)
                .push_next(&mut flags);
            // SAFETY: valid device and allocate info.
            memory = s.check("vkAllocateMemory", unsafe {
                d.allocate_memory(&alloc, None)
            });
            if !matches!(memory, Err(Error::OutOfMemory)) {
                let props = s.memory.memory_types[t as usize].property_flags;
                host = props.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                cached = props.contains(vk::MemoryPropertyFlags::HOST_CACHED);
                break;
            }
        }
        let memory = match memory {
            Ok(m) => m,
            Err(e) => {
                // SAFETY: the buffer has no memory and is unused.
                unsafe { d.destroy_buffer(buffer, None) };
                return Err(e);
            }
        };
        s.allocated.fetch_add(req.size, Ordering::Relaxed);
        let mut b = Buffer {
            shared: s.clone(),
            buffer,
            memory,
            len,
            size: req.size,
            mapped: None,
            host_cached: false,
            address: 0,
        };
        // SAFETY: the memory was allocated for this buffer's requirements.
        s.check("vkBindBufferMemory", unsafe {
            d.bind_buffer_memory(buffer, memory, 0)
        })?;
        if host && map {
            // SAFETY: host-visible memory, not mapped yet.
            let p = s.check("vkMapMemory", unsafe {
                d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            })?;
            b.mapped = NonNull::new(p.cast());
            b.host_cached = cached;
        }
        let i = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        // SAFETY: a buffer with SHADER_DEVICE_ADDRESS usage bound to memory
        // allocated with DEVICE_ADDRESS.
        b.address = unsafe { d.get_buffer_device_address(&i) };
        Ok(b)
    }

    /// A device buffer of `len` bytes, unspecified contents: mapped for the
    /// host on a unified device, device-local and unmapped otherwise.
    pub fn alloc(&self, len: usize) -> Result<Buffer> {
        if self.info.unified_memory {
            self.create_buffer(len, UNIFIED, true)
        } else {
            self.create_buffer(len, DEVICE_ONLY, false)
        }
    }

    /// A host-visible, mapped staging buffer of `len` bytes, unspecified
    /// contents. Kernels may read it through its device address.
    pub fn staging(&self, len: usize) -> Result<Buffer> {
        let b = self.create_buffer(len, STAGING, true)?;
        if b.mapped.is_none() {
            return Err(Error::Unsupported("host-visible coherent memory".into()));
        }
        Ok(b)
    }

    /// Fills `size` bytes of `buffer` from byte `offset` (both multiples of
    /// 4; `None`: to the end) with `word`, on the GPU, and waits. The caller
    /// has exclusive use of the range. Callable from any thread, also while
    /// a [`Commands`] is being recorded.
    pub fn fill(
        &self,
        buffer: &Buffer,
        offset: usize,
        size: Option<usize>,
        word: u32,
    ) -> Result<()> {
        let mut cmds = self.begin(&self.aux)?;
        cmds.fill(buffer, offset, size, word)?;
        cmds.submit().map(|_| ())
    }

    /// Reads `buffer` back in parts through a staging buffer, calling `f` on
    /// each part in order until it returns `false`; returns whether every
    /// call returned `true`. The staging buffer is zeroed before it is
    /// freed. The caller has exclusive use of the buffer. Callable from any
    /// thread, as [`Context::fill`].
    pub fn read_chunks(&self, buffer: &Buffer, mut f: impl FnMut(&[u8]) -> bool) -> Result<bool> {
        let len = buffer.len;
        let staging = self.staging(len.clamp(4, READ_CHUNK))?;
        let Some(p) = staging.mapped else {
            return Err(Error::Unsupported("host-visible coherent memory".into()));
        };
        let mut all = true;
        let mut at = 0;
        let r = loop {
            if at >= len || !all {
                break Ok(all);
            }
            let n = READ_CHUNK.min(len - at);
            let copied = self.begin(&self.aux).and_then(|mut cmds| {
                cmds.copy_region(buffer, at, &staging, 0, n)?;
                cmds.submit()
            });
            if let Err(e) = copied {
                break Err(e);
            }
            // SAFETY: the mapping holds at least `n` bytes; the copy into it
            // completed (fence waited, host-visible coherent memory) and no
            // device work uses it now.
            all &= f(unsafe { std::slice::from_raw_parts(p.as_ptr(), n) });
            at += n;
        };
        // SAFETY: as above, the whole mapping; nothing else uses it.
        unsafe { std::ptr::write_bytes(p.as_ptr(), 0, staging.len) };
        std::sync::atomic::compiler_fence(Ordering::SeqCst);
        r
    }

    /// Starts recording the main command buffer. The context runs one at a
    /// time: this waits for the [`Commands`] of another thread to be
    /// submitted or dropped.
    pub fn commands(&self) -> Result<Commands<'_>> {
        self.begin(&self.submit)
    }

    /// Resets the command buffer of `slot` (locked) and starts recording it.
    fn begin<'a>(&'a self, slot: &'a Mutex<Submit>) -> Result<Commands<'a>> {
        self.shared.alive()?;
        let submit = Self::lock(slot);
        let s = &self.shared;
        let d = &s.device;
        // SAFETY: the pool's command buffer is not in use: the last
        // submission was waited for (or never submitted).
        unsafe {
            s.check(
                "vkResetCommandPool",
                d.reset_command_pool(submit.pool, vk::CommandPoolResetFlags::empty()),
            )?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            s.check(
                "vkBeginCommandBuffer",
                d.begin_command_buffer(submit.cmd, &begin),
            )?;
            if submit.query != vk::QueryPool::null() {
                d.cmd_reset_query_pool(submit.cmd, submit.query, 0, 2);
                d.cmd_write_timestamp(
                    submit.cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    submit.query,
                    0,
                );
            }
            // Make host writes to mapped unified (especially HOST_CACHED)
            // buffers visible to device reads before the first command runs.
            if self.info.unified_memory {
                let from_host = [vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(
                        vk::AccessFlags::SHADER_READ
                            | vk::AccessFlags::SHADER_WRITE
                            | vk::AccessFlags::TRANSFER_READ
                            | vk::AccessFlags::TRANSFER_WRITE,
                    )];
                d.cmd_pipeline_barrier(
                    submit.cmd,
                    vk::PipelineStageFlags::HOST,
                    vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &from_host,
                    &[],
                    &[],
                );
            }
        }
        Ok(Commands {
            ctx: self,
            submit,
            recorded: false,
            keep: Vec::new(),
        })
    }
}

/// Holds the objects created by [`Context::finish`] until it succeeds.
struct Partial<'a> {
    shared: &'a Arc<Shared>,
    submit: Submit,
    aux: Submit,
    pipeline_layout: vk::PipelineLayout,
}

impl Partial<'_> {
    fn take(&mut self) -> (Submit, Submit, vk::PipelineLayout) {
        let q = self.submit.queue;
        (
            std::mem::replace(&mut self.submit, Submit::null(q)),
            std::mem::replace(&mut self.aux, Submit::null(q)),
            std::mem::take(&mut self.pipeline_layout),
        )
    }
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        let d = &self.shared.device;
        self.submit.destroy(d);
        self.aux.destroy(d);
        // SAFETY: the device is idle; the layout is null or owned here.
        unsafe { d.destroy_pipeline_layout(self.pipeline_layout, None) };
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let d = &self.shared.device;
        // SAFETY: valid device.
        let _ = unsafe { d.device_wait_idle() };
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.submit.get_mut().unwrap_or_else(|e| e.into_inner()).destroy(d);
        self.aux.get_mut().unwrap_or_else(|e| e.into_inner()).destroy(d);
        // SAFETY: no pipeline uses the layout or cache any more.
        unsafe {
            d.destroy_pipeline_cache(self.pipeline_cache, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

/// The options of a pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineOptions {
    /// The workgroup width (specialization constant 0).
    pub workgroup: u32,
    /// A required subgroup size, with full subgroups (the workgroup width
    /// must be a multiple of it): requested where the device lets compute
    /// pipelines choose it, else accepted when the device's size is fixed
    /// at it. `None`: the driver's choice.
    pub subgroup: Option<u32>,
    /// Specialization constants `1..=3`; 0 keeps the shader's default.
    pub spec: [u32; SPEC_CONSTANTS],
}

impl Default for PipelineOptions {
    fn default() -> Self {
        PipelineOptions {
            workgroup: DEFAULT_WORKGROUP,
            subgroup: None,
            spec: [0; SPEC_CONSTANTS],
        }
    }
}

/// A compute pipeline of the context.
pub struct Pipeline {
    shared: Arc<Shared>,
    pipeline: vk::Pipeline,
    name: &'static str,
    variant: Variant,
    options: PipelineOptions,
    /// The wall time of `vkCreateComputePipelines`.
    creation: Duration,
}

impl Pipeline {
    /// The wall time the driver took to create the pipeline (its shader
    /// compilation).
    pub fn creation_time(&self) -> Duration {
        self.creation
    }

    /// The kernel name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The workgroup width.
    pub fn workgroup(&self) -> u32 {
        self.options.workgroup
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("variant", &self.variant)
            .field("options", &self.options)
            .finish()
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: no command buffer using the pipeline is pending: commands
        // keep it alive until their submission was waited for.
        unsafe { self.shared.device.destroy_pipeline(self.pipeline, None) };
    }
}

/// A buffer with its own memory and a device address.
pub struct Buffer {
    shared: Arc<Shared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: usize,
    /// The bytes of the memory allocation.
    size: u64,
    mapped: Option<NonNull<u8>>,
    host_cached: bool,
    address: u64,
}

// SAFETY: the handles may be used from any thread; the mapping is only
// written through `&self` by the context's copy functions, which the caller
// orders with the device work.
unsafe impl Send for Buffer {}
// SAFETY: as above.
unsafe impl Sync for Buffer {}

impl Buffer {
    /// The length in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no byte.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The device address of the first byte.
    pub fn device_address(&self) -> u64 {
        self.address
    }

    /// The host address of the mapping, at least [`Buffer::len`] bytes.
    pub fn mapped_ptr(&self) -> Option<NonNull<u8>> {
        self.mapped
    }

    /// Whether the host mapping is CPU-cached (`HOST_CACHED`).
    pub fn host_cached(&self) -> bool {
        self.host_cached
    }
}

impl fmt::Debug for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffer")
            .field("len", &self.len)
            .field("mapped", &self.mapped.is_some())
            .finish()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _gate = self.shared.in_flight.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: no pending command buffer uses the buffer (commands borrow
        // it until their submission was waited for), and none is in flight;
        // freeing mapped memory unmaps it.
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
        self.shared.allocated.fetch_sub(self.size, Ordering::Relaxed);
    }
}

/// The push-constant block of one dispatch, at most [`MAX_PARAMS_BYTES`]
/// bytes: buffer addresses first, then scalar words. A larger parameter
/// struct is passed in a buffer whose address is in the block. It borrows
/// every buffer whose address it holds, so the buffers outlive the
/// [`Commands`].
#[derive(Clone, Debug, Default)]
pub struct Params<'a> {
    words: Vec<u32>,
    buffers: PhantomData<&'a Buffer>,
}

impl<'a> Params<'a> {
    /// An empty block.
    pub fn new() -> Params<'a> {
        Params::default()
    }

    /// Appends the address of byte `offset` of `buffer`; an offset past the
    /// end is recorded as the end (the kernel reads nothing there).
    pub fn addr_at(mut self, buffer: &'a Buffer, offset: usize) -> Params<'a> {
        debug_assert!(
            self.words.len().is_multiple_of(2),
            "an address after a scalar"
        );
        let a = buffer.address + offset.min(buffer.len) as u64;
        self.words.extend([a as u32, (a >> 32) as u32]);
        self
    }

    /// Appends one word.
    pub fn u32(mut self, v: u32) -> Params<'a> {
        self.words.push(v);
        self
    }

    /// Appends words (a scalar is its eight little-endian 32-bit limbs).
    pub fn words(mut self, v: &[u32]) -> Params<'a> {
        self.words.extend_from_slice(v);
        self
    }

    /// Appends 64-bit values as pairs of words, low word first (a scalar's
    /// `[u64; 4]` limbs).
    pub fn u64s(mut self, v: &[u64]) -> Params<'a> {
        self.words.extend(v.iter().flat_map(|w| [*w as u32, (*w >> 32) as u32]));
        self
    }

    /// The block as words.
    pub fn as_words(&self) -> &[u32] {
        &self.words
    }
}

/// The grid of a dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grid {
    /// At least this many invocations: at least `ceil(n / workgroup)`
    /// workgroups; the kernel guards its own element count.
    Threads(u64),
    /// Exactly this many workgroups.
    Groups(u64),
}

/// A command buffer being recorded. It holds its submission lock and
/// borrows every buffer it records until [`submit`] returns; dropped without
/// a submission, nothing runs.
///
/// [`submit`]: Commands::submit
pub struct Commands<'a> {
    ctx: &'a Context,
    submit: MutexGuard<'a, Submit>,
    recorded: bool,
    /// The pipelines recorded through [`Commands::dispatch_shared`].
    keep: Vec<Arc<Pipeline>>,
}

impl fmt::Debug for Commands<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Commands").field("recorded", &self.recorded).finish()
    }
}

/// `(x, y)` with `x * y == groups`, `x <= max_x`, `y <= max_y`: the 2D grid
/// of exactly `groups` workgroups (`None` if there is none).
fn fold(groups: u64, max_x: u32, max_y: u32) -> Option<(u32, u32)> {
    if groups <= max_x as u64 {
        return Some((groups as u32, 1));
    }
    let min_x = groups.div_ceil(max_y as u64);
    (min_x..=max_x as u64)
        .rev()
        .find(|x| groups.is_multiple_of(*x))
        .map(|x| (x as u32, (groups / x) as u32))
}

/// `(x, y)` with `x * y >= groups`, `x <= max_x`, `y <= max_y`, and fewer
/// than `x` surplus workgroups: a 2D grid of at least `groups` workgroups.
fn fold_at_least(groups: u64, max_x: u32, max_y: u32) -> Option<(u32, u32)> {
    let y = groups.div_ceil(max_x as u64);
    if y > max_y as u64 {
        return None;
    }
    let y = y.max(1);
    Some((groups.div_ceil(y) as u32, y as u32))
}

impl<'a> Commands<'a> {
    /// Records `pipeline` over `grid`, with `params` as its push constants,
    /// keeping `pipeline` alive until the commands are submitted or dropped.
    /// A grid beyond `maxComputeWorkGroupCount[0]` workgroups is folded into
    /// a second dimension: of exactly as many workgroups for
    /// [`Grid::Groups`], of at least as many for [`Grid::Threads`] (kernels
    /// index by `global_index()` / `group_index()`). The dispatch sees the
    /// writes of everything recorded before it.
    pub fn dispatch_shared(
        &mut self,
        pipeline: &Arc<Pipeline>,
        params: &Params<'a>,
        grid: Grid,
    ) -> Result<()> {
        self.record(pipeline, params, grid)?;
        if !self.keep.iter().any(|p| Arc::ptr_eq(p, pipeline)) {
            self.keep.push(pipeline.clone());
        }
        Ok(())
    }

    fn record(&mut self, pipeline: &Pipeline, params: &Params<'a>, grid: Grid) -> Result<()> {
        let info = &self.ctx.info;
        let push = params.as_words();
        let push_bytes = size_of_val(push) as u32;
        if push_bytes > MAX_PARAMS_BYTES.min(info.max_push_constants_size) {
            return Err(Error::Invalid(format!(
                "{push_bytes} bytes of params for {}",
                pipeline.name
            )));
        }
        let [max_x, max_y, _] = info.max_compute_work_group_count;
        let (groups, folded) = match grid {
            Grid::Threads(n) => {
                let g = n.div_ceil(pipeline.workgroup() as u64);
                (g, fold_at_least(g, max_x, max_y))
            }
            Grid::Groups(g) => (g, fold(g, max_x, max_y)),
        };
        if groups == 0 {
            return Ok(());
        }
        let (gx, gy) = folded.ok_or_else(|| {
            Error::Invalid(format!(
                "{groups} workgroups of {} fit no dispatch grid",
                pipeline.name
            ))
        })?;
        self.barrier();
        let d = &self.ctx.shared.device;
        let bytes: Vec<u8> = push.iter().flat_map(|w| w.to_ne_bytes()).collect();
        let cmd = self.submit.cmd;
        // SAFETY: the command buffer is recording; the pipeline and every
        // buffer whose address is pushed outlive the submission (borrowed for
        // 'a, or kept).
        unsafe {
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            if !bytes.is_empty() {
                d.cmd_push_constants(
                    cmd,
                    self.ctx.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &bytes,
                );
            }
            d.cmd_dispatch(cmd, gx, gy, 1);
        }
        self.recorded = true;
        Ok(())
    }

    /// Makes the writes of every command recorded so far visible to the
    /// commands recorded after it (a no-op before the first command).
    fn barrier(&mut self) {
        if !self.recorded {
            return;
        }
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_READ
                    | vk::AccessFlags::TRANSFER_WRITE,
            )];
        // SAFETY: the command buffer is recording.
        unsafe {
            self.ctx.shared.device.cmd_pipeline_barrier(
                self.submit.cmd,
                stages,
                stages,
                vk::DependencyFlags::empty(),
                &b,
                &[],
                &[],
            )
        };
    }

    /// Records a copy of the first `bytes` bytes of `src` to `dst`.
    pub fn copy(&mut self, src: &'a Buffer, dst: &'a Buffer, bytes: usize) -> Result<()> {
        self.copy_region(src, 0, dst, 0, bytes)
    }

    /// Records a copy of `bytes` bytes from byte `src_at` of `src` to byte
    /// `dst_at` of `dst`.
    pub fn copy_region(
        &mut self,
        src: &'a Buffer,
        src_at: usize,
        dst: &'a Buffer,
        dst_at: usize,
        bytes: usize,
    ) -> Result<()> {
        let fits = |at: usize, b: &Buffer| at.checked_add(bytes).is_some_and(|e| e <= b.len);
        if !fits(src_at, src) || !fits(dst_at, dst) {
            return Err(Error::Invalid(format!("a copy of {bytes} bytes")));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.barrier();
        let region = [vk::BufferCopy::default()
            .src_offset(src_at as u64)
            .dst_offset(dst_at as u64)
            .size(bytes as u64)];
        // SAFETY: the command buffer is recording; both ranges are inside
        // their buffers.
        unsafe {
            self.ctx
                .shared
                .device
                .cmd_copy_buffer(self.submit.cmd, src.buffer, dst.buffer, &region)
        };
        self.recorded = true;
        Ok(())
    }

    /// Records a fill of `size` bytes of `dst` (`None`: to the end) from byte
    /// `offset` with `word`; `offset` and `size` are multiples of 4.
    pub fn fill(
        &mut self,
        dst: &'a Buffer,
        offset: usize,
        size: Option<usize>,
        word: u32,
    ) -> Result<()> {
        let whole = dst.len.max(4).next_multiple_of(4);
        let end = size.map_or(Some(whole), |s| offset.checked_add(s));
        if !offset.is_multiple_of(4)
            || offset > whole
            || size.is_some_and(|s| !s.is_multiple_of(4))
            || end.is_none_or(|e| e > whole)
        {
            return Err(Error::Invalid(format!(
                "a fill at {offset} of {size:?} bytes"
            )));
        }
        self.barrier();
        let size = size.map_or(vk::WHOLE_SIZE, |s| s as u64);
        // SAFETY: the command buffer is recording; the range is inside the
        // buffer.
        unsafe {
            self.ctx.shared.device.cmd_fill_buffer(
                self.submit.cmd,
                dst.buffer,
                offset as u64,
                size,
                word,
            )
        };
        self.recorded = true;
        Ok(())
    }

    /// Submits the commands and waits for them. Returns the GPU time between
    /// timestamps around the commands where the queue has timestamps, else
    /// the wall time from the submission to the signalled fence.
    pub fn submit(self) -> Result<Duration> {
        let s = &self.ctx.shared;
        let d = &s.device;
        let cmd = self.submit.cmd;
        let query = self.submit.query;
        let to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: the command buffer is recording; queue, fence and query
        // pool belong to the context and are used under the submission lock
        // held by `self`, the queue under the queue lock.
        unsafe {
            d.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &to_host,
                &[],
                &[],
            );
            if query != vk::QueryPool::null() {
                d.cmd_write_timestamp(cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, query, 1);
            }
            s.check("vkEndCommandBuffer", d.end_command_buffer(cmd))?;
            s.check("vkResetFences", d.reset_fences(&[self.submit.fence]))?;
            let cmds = [cmd];
            let info = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let gate = s.in_flight.lock().unwrap_or_else(|e| e.into_inner());
            let t0 = Instant::now();
            s.check(
                "vkQueueSubmit",
                d.queue_submit(self.submit.queue, &info, self.submit.fence),
            )?;
            s.check(
                "vkWaitForFences",
                d.wait_for_fences(&[self.submit.fence], true, u64::MAX),
            )?;
            let wall = t0.elapsed();
            drop(gate);
            if query == vk::QueryPool::null() {
                return Ok(wall);
            }
            let mut ticks = [0u64; 2];
            s.check(
                "vkGetQueryPoolResults",
                d.get_query_pool_results(
                    query,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                ),
            )?;
            let bits = self.ctx.info.timestamp_valid_bits;
            let mask = if bits >= 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let delta = ticks[1].wrapping_sub(ticks[0]) & mask;
            let ns = delta as f64 * self.ctx.info.timestamp_period as f64;
            Ok(Duration::from_nanos(ns as u64))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Error, fold, fold_at_least};

    #[test]
    fn grids_are_exact() {
        assert_eq!(fold(5, 65535, 65535), Some((5, 1)));
        assert_eq!(fold(1 << 18, 65535, 65535), Some((32768, 8)));
        assert_eq!(fold(100_000, 65535, 65535), Some((50_000, 2)));
        // 65537 is prime: no exact grid
        assert_eq!(fold(3 * 65537, 65535, 65535), None);
    }

    #[test]
    fn thread_grids_cover_every_group() {
        for g in [1u64, 5, 65535, 65536, 3 * 65537, (1 << 22) + 1] {
            let (x, y) = fold_at_least(g, 65535, 65535).unwrap();
            let total = x as u64 * y as u64;
            assert!(total >= g && total - g < x as u64, "{g}: {x} x {y}");
        }
        assert_eq!(fold_at_least(1 << 40, 65535, 65535), None);
    }

    #[test]
    fn buffers_follow_the_memory_mode() {
        let c = crate::ctx::context().expect("usable Vulkan device").device();
        let info = c.info();
        if std::env::var(crate::ctx::DISCRETE_ENV).as_deref() == Ok("1") {
            assert!(!info.unified_memory);
        }
        let b = c.alloc(4096 + 4).unwrap();
        assert_eq!(b.mapped_ptr().is_some(), info.unified_memory);
        assert!(c.staging(4096).unwrap().mapped_ptr().is_some());
        c.fill(&b, 0, None, 0x0102_0304).unwrap();
        c.fill(&b, 8, Some(4), 0).unwrap();
        let mut bytes = Vec::new();
        assert!(c.read_chunks(&b, |p| {
            bytes.extend_from_slice(p);
            true
        }).unwrap());
        let words: Vec<u32> =
            bytes.chunks(4).map(|w| u32::from_ne_bytes(w.try_into().unwrap())).collect();
        assert_eq!(words.len(), 1025);
        assert!(
            words
                .iter()
                .enumerate()
                .all(|(i, w)| *w == if i == 2 { 0 } else { 0x0102_0304 })
        );
        assert!(matches!(c.fill(&b, 2, Some(4), 0), Err(Error::Invalid(_))));
        assert!(matches!(
            c.fill(&b, 4096, Some(8), 0),
            Err(Error::Invalid(_))
        ));
        let (used, total) = c.budget();
        assert!(used > 0 && total > 0, "{used} of {total}");
    }
}
