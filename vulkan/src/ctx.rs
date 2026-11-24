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

//! The Vulkan device context: one device, one compute queue, the embedded
//! compute pipelines, storage buffers, and recorded command buffers that are
//! submitted and waited for.
//!
//! Every kernel binds storage buffers at set 0, bindings `0..MAX_BUFFERS`,
//! and reads one params struct from push constants; every pipeline shares
//! one pipeline layout. The workgroup width is specialization constant 0.
//!
//! Every Vulkan failure is a returned [`Error`]. `VK_ERROR_DEVICE_LOST`
//! latches the context as failed: every later call returns
//! [`Error::Failed`].

use std::{
    collections::HashMap,
    ffi::{CStr, c_char},
    fmt,
    io::Cursor,
    ptr::NonNull,
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use ash::vk;

/// The environment variable naming the Vulkan loader library to load
/// instead of the system's (`libvulkan.so.1`, `libvulkan.dylib`,
/// `vulkan-1.dll`).
pub const LOADER_ENV: &str = "MIDNIGHT_VULKAN_LOADER";
/// The storage buffer bindings of the shared descriptor set layout.
pub const MAX_BUFFERS: u32 = 8;
/// The bytes of push constants a kernel's params struct may take: the
/// minimum `maxPushConstantsSize` Vulkan guarantees.
pub const MAX_PARAMS_BYTES: u32 = 128;
/// The descriptor sets one command buffer may use (one per dispatch).
const MAX_DISPATCHES: u32 = 1024;
/// The workgroup width of [`PipelineOptions::default`].
pub const DEFAULT_WORKGROUP: u32 = 256;

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

impl Variant {
    /// Both variants.
    pub const ALL: [Variant; 2] = [Variant::Int64, Variant::Int32];
}

/// A failure of the device or of a call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// The Vulkan loader library could not be loaded.
    Loader(String),
    /// No Vulkan 1.3 GPU with a compute queue.
    NoDevice,
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
            Error::NoDevice => write!(f, "no Vulkan 1.3 GPU with a compute queue"),
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

/// Plain data that may be copied to and from device buffers byte for byte.
///
/// # Safety
///
/// Every bit pattern of the type's size is a valid value, and it has no
/// padding.
pub unsafe trait Pod: Copy + Send + Sync + 'static {}

// SAFETY: integers have no padding and no invalid bit patterns.
unsafe impl Pod for u8 {}
// SAFETY: as above.
unsafe impl Pod for u16 {}
// SAFETY: as above.
unsafe impl Pod for u32 {}
// SAFETY: as above.
unsafe impl Pod for u64 {}
// SAFETY: as above.
unsafe impl Pod for i32 {}
// SAFETY: as above.
unsafe impl Pod for i64 {}
// SAFETY: an array of `Pod` has no padding between its elements.
unsafe impl<T: Pod, const N: usize> Pod for [T; N] {}

/// The device as its properties describe it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
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
    /// `shaderInt64` (variant A).
    pub shader_int64: bool,
    /// `bufferDeviceAddress`.
    pub buffer_device_address: bool,
    /// An integrated GPU: device memory is mapped for the host.
    pub unified_memory: bool,
}

/// The device, shared by the context, its buffers and its pipelines.
struct Shared {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    failed: AtomicBool,
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

/// The objects of one submission at a time.
struct Submit {
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorPool,
}

/// The pipeline cache key: kernel, variant, workgroup width, subgroup size.
type Key = (&'static str, Variant, u32, Option<u32>);

/// A Vulkan device with a compute queue.
pub struct Context {
    shared: Arc<Shared>,
    info: DeviceInfo,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    submit: Mutex<Submit>,
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

/// The rank of a device type; lower is preferred, `None` is never used.
fn type_rank(t: vk::PhysicalDeviceType) -> Option<u32> {
    match t {
        vk::PhysicalDeviceType::DISCRETE_GPU => Some(0),
        vk::PhysicalDeviceType::INTEGRATED_GPU => Some(1),
        vk::PhysicalDeviceType::VIRTUAL_GPU => Some(2),
        vk::PhysicalDeviceType::CPU => None,
        _ => Some(3),
    }
}

/// A candidate device: its handle, compute queue family and properties.
struct Candidate {
    physical: vk::PhysicalDevice,
    family: u32,
    info: DeviceInfo,
    portability_subset: bool,
}

impl Context {
    /// Loads the Vulkan loader ([`LOADER_ENV`]) and opens the preferred
    /// device: a discrete GPU, else an integrated one; CPU implementations
    /// are skipped.
    pub fn new() -> Result<Context> {
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
        match Self::open(&instance) {
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

    /// Picks the device and creates it.
    fn open(
        instance: &ash::Instance,
    ) -> Result<(ash::Device, vk::PhysicalDeviceMemoryProperties, Candidate)> {
        // SAFETY: valid instance.
        let physicals = init_error("vkEnumeratePhysicalDevices", unsafe {
            instance.enumerate_physical_devices()
        })?;
        let mut best: Option<(u32, Candidate)> = None;
        for p in physicals {
            let Some(c) = Self::candidate(instance, p)? else {
                continue;
            };
            let rank = type_rank(c.info.device_type).unwrap_or(u32::MAX);
            if best.as_ref().is_none_or(|(r, _)| rank < *r) {
                best = Some((rank, c));
            }
        }
        let (_, c) = best.ok_or(Error::NoDevice)?;

        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(c.info.buffer_device_address);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(c.info.subgroup_size_control)
            .compute_full_subgroups(c.info.compute_full_subgroups);
        let features = vk::PhysicalDeviceFeatures::default().shader_int64(c.info.shader_int64);
        let priorities = [1.0f32];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(c.family)
            .queue_priorities(&priorities)];
        let mut exts: Vec<*const c_char> = Vec::new();
        if c.portability_subset {
            exts.push(vk::KHR_PORTABILITY_SUBSET_NAME.as_ptr());
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

    /// The device `p` if it is a usable GPU: Vulkan 1.3, not a CPU
    /// implementation, with a compute queue.
    fn candidate(instance: &ash::Instance, p: vk::PhysicalDevice) -> Result<Option<Candidate>> {
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
            || type_rank(base.device_type).is_none()
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
        let Some(&(family, _)) = compute
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
        // SAFETY: valid physical device.
        let exts = unsafe { instance.enumerate_device_extension_properties(p) };
        let exts = init_error("vkEnumerateDeviceExtensionProperties", exts)?;
        let portability_subset = exts
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(vk::KHR_PORTABILITY_SUBSET_NAME));

        let limits = base.limits;
        let v = base.api_version;
        let info = DeviceInfo {
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
            shader_int64,
            buffer_device_address: f12.buffer_device_address == vk::TRUE,
            unified_memory: base.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU,
        };
        Ok(Some(Candidate {
            physical: p,
            family: family as u32,
            info,
            portability_subset,
        }))
    }

    /// Creates the queue, command, descriptor and layout objects.
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
            device,
            memory,
            failed: AtomicBool::new(false),
        });
        // From here `Shared` owns the device; objects created below are
        // destroyed by `Context::drop` or, on failure, by `Partial::drop`.
        let mut p = Partial {
            shared: &shared,
            submit: Submit {
                // SAFETY: the family and index exist (one queue created).
                queue: unsafe { shared.device.get_device_queue(c.family, 0) },
                pool: vk::CommandPool::null(),
                cmd: vk::CommandBuffer::null(),
                fence: vk::Fence::null(),
                descriptors: vk::DescriptorPool::null(),
            },
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
        };
        let d = &shared.device;
        let s = &shared;
        let pool = vk::CommandPoolCreateInfo::default().queue_family_index(c.family);
        // SAFETY: valid device and create infos for every call below.
        p.submit.pool = s.check("vkCreateCommandPool", unsafe {
            d.create_command_pool(&pool, None)
        })?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(p.submit.pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        p.submit.cmd = s.check("vkAllocateCommandBuffers", unsafe {
            d.allocate_command_buffers(&alloc)
        })?[0];
        p.submit.fence = s.check("vkCreateFence", unsafe {
            d.create_fence(&vk::FenceCreateInfo::default(), None)
        })?;
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(MAX_DISPATCHES * MAX_BUFFERS)];
        let dpool = vk::DescriptorPoolCreateInfo::default()
            .max_sets(MAX_DISPATCHES)
            .pool_sizes(&sizes);
        p.submit.descriptors = s.check("vkCreateDescriptorPool", unsafe {
            d.create_descriptor_pool(&dpool, None)
        })?;
        let bindings: Vec<_> = (0..MAX_BUFFERS)
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let layout = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        p.set_layout = s.check("vkCreateDescriptorSetLayout", unsafe {
            d.create_descriptor_set_layout(&layout, None)
        })?;
        let push = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(MAX_PARAMS_BYTES.min(c.info.max_push_constants_size))];
        let sets = [p.set_layout];
        let pl = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&sets)
            .push_constant_ranges(&push);
        p.pipeline_layout = s.check("vkCreatePipelineLayout", unsafe {
            d.create_pipeline_layout(&pl, None)
        })?;

        let (submit, set_layout, pipeline_layout) = p.take();
        Ok(Context {
            shared: shared.clone(),
            info: c.info,
            set_layout,
            pipeline_layout,
            submit: Mutex::new(submit),
            pipelines: Mutex::new(HashMap::new()),
        })
    }

    /// The device's properties.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Whether a device loss latched the context as failed.
    pub fn failed(&self) -> bool {
        self.shared.failed.load(Ordering::Acquire)
    }

    fn lock_submit(&self) -> MutexGuard<'_, Submit> {
        self.submit.lock().unwrap_or_else(|e| e.into_inner())
    }

    // -----------------------------------------------------------------------
    // Pipelines

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
        let key = (shader.name, variant, options.workgroup, options.subgroup);
        let mut cache = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let p = Arc::new(self.create_pipeline(shader, variant, options)?);
        cache.insert(key, p.clone());
        Ok(p)
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
            let fixed = info.subgroup_size == s && info.max_subgroup_size == s;
            if !info.compute_full_subgroups || !(info.subgroup_size_control || fixed) {
                return Err(Error::Unsupported(format!(
                    "a compute subgroup size of {s} (subgroupSizeControl, computeFullSubgroups)"
                )));
            }
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
        let entries = [vk::SpecializationMapEntry::default().constant_id(0).offset(0).size(4)];
        let data = wg.to_ne_bytes();
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
        // SAFETY: valid device, module and layout; the create info outlives
        // the call.
        let r = unsafe { d.create_compute_pipelines(vk::PipelineCache::null(), &create, None) };
        // SAFETY: the module is no longer needed once the pipeline exists.
        unsafe { d.destroy_shader_module(module, None) };
        let pipeline = s.check("vkCreateComputePipelines", r.map_err(|(_, e)| e))?[0];
        Ok(Pipeline {
            shared: s.clone(),
            pipeline,
            name: shader.name,
            variant,
            workgroup: wg,
        })
    }

    // -----------------------------------------------------------------------
    // Buffers

    /// The index of the first memory type of `bits` with every flag of each
    /// preference in turn.
    fn memory_types(&self, bits: u32, preferences: &[vk::MemoryPropertyFlags]) -> Vec<u32> {
        let m = &self.shared.memory;
        let mut out = Vec::new();
        for want in preferences {
            for i in 0..m.memory_type_count {
                let flags = m.memory_types[i as usize].property_flags;
                if bits & (1 << i) != 0 && flags.contains(*want) && !out.contains(&i) {
                    out.push(i);
                }
            }
        }
        out
    }

    fn create_buffer(&self, len: usize, preferences: &[vk::MemoryPropertyFlags]) -> Result<Buffer> {
        self.shared.alive()?;
        if len as u64 > self.info.max_memory_allocation_size {
            return Err(Error::OutOfMemory);
        }
        let s = &self.shared;
        let d = &s.device;
        let size = (len.max(4) as u64).next_multiple_of(4);
        let mut usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        if self.info.buffer_device_address {
            usage |= vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
        }
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
        for t in self.memory_types(req.memory_type_bits, preferences) {
            let mut flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mut alloc =
                vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(t);
            if self.info.buffer_device_address {
                alloc = alloc.push_next(&mut flags);
            }
            // SAFETY: valid device and allocate info.
            memory = s.check("vkAllocateMemory", unsafe {
                d.allocate_memory(&alloc, None)
            });
            if !matches!(memory, Err(Error::OutOfMemory)) {
                host = s.memory.memory_types[t as usize].property_flags.contains(
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
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
        let mut b = Buffer {
            shared: s.clone(),
            buffer,
            memory,
            len,
            mapped: None,
            address: None,
        };
        // SAFETY: the memory was allocated for this buffer's requirements.
        s.check("vkBindBufferMemory", unsafe {
            d.bind_buffer_memory(buffer, memory, 0)
        })?;
        if host {
            // SAFETY: host-visible memory, not mapped yet.
            let p = s.check("vkMapMemory", unsafe {
                d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            })?;
            b.mapped = NonNull::new(p.cast());
        }
        if self.info.buffer_device_address {
            let i = vk::BufferDeviceAddressInfo::default().buffer(buffer);
            // SAFETY: a buffer with SHADER_DEVICE_ADDRESS usage bound to
            // memory allocated with DEVICE_ADDRESS.
            b.address = Some(unsafe { d.get_buffer_device_address(&i) });
        }
        Ok(b)
    }

    /// A device buffer of `len` bytes, unspecified contents. On a unified
    /// memory device it is mapped for the host.
    pub fn alloc(&self, len: usize) -> Result<Buffer> {
        use vk::MemoryPropertyFlags as M;
        if self.info.unified_memory {
            self.create_buffer(
                len,
                &[
                    M::DEVICE_LOCAL | M::HOST_VISIBLE | M::HOST_COHERENT,
                    M::DEVICE_LOCAL,
                ],
            )
        } else {
            self.create_buffer(len, &[M::DEVICE_LOCAL])
        }
    }

    /// A host-visible staging buffer of `len` bytes.
    fn staging(&self, len: usize) -> Result<Buffer> {
        use vk::MemoryPropertyFlags as M;
        self.create_buffer(
            len,
            &[
                M::HOST_VISIBLE | M::HOST_COHERENT | M::HOST_CACHED,
                M::HOST_VISIBLE | M::HOST_COHERENT,
            ],
        )
    }

    /// A device buffer holding `data`.
    pub fn upload<T: Pod>(&self, data: &[T]) -> Result<Buffer> {
        let b = self.alloc(size_of_val(data))?;
        self.write(&b, data)?;
        Ok(b)
    }

    /// Writes `data` at the start of `buffer`.
    pub fn write<T: Pod>(&self, buffer: &Buffer, data: &[T]) -> Result<()> {
        let bytes = size_of_val(data);
        if bytes > buffer.len {
            return Err(Error::Invalid(format!(
                "{bytes} bytes into a buffer of {}",
                buffer.len
            )));
        }
        if buffer.mapped.is_some() {
            // SAFETY: `T: Pod`; the mapping holds `buffer.len` bytes.
            unsafe { buffer.host_write(data) };
            return Ok(());
        }
        let staging = self.staging(bytes)?;
        // SAFETY: as above.
        unsafe { staging.host_write(data) };
        let mut cmds = self.commands()?;
        cmds.copy(&staging, buffer, bytes)?;
        cmds.submit()?;
        Ok(())
    }

    /// The first `buffer.len() / size_of::<T>()` values of `buffer`.
    pub fn read<T: Pod>(&self, buffer: &Buffer) -> Result<Vec<T>> {
        let n = buffer.len / size_of::<T>().max(1);
        let bytes = n * size_of::<T>();
        let mut out: Vec<T> = Vec::with_capacity(n);
        let copy = |src: &Buffer, out: &mut Vec<T>| {
            if let Some(p) = src.mapped {
                // SAFETY: the mapping holds at least `bytes` bytes and `out`
                // has capacity for them; every bit pattern is a `T`.
                unsafe {
                    std::ptr::copy_nonoverlapping(p.as_ptr(), out.as_mut_ptr().cast::<u8>(), bytes);
                    out.set_len(n);
                }
            }
        };
        if buffer.mapped.is_some() {
            copy(buffer, &mut out);
            return Ok(out);
        }
        let staging = self.staging(bytes)?;
        let mut cmds = self.commands()?;
        cmds.copy(buffer, &staging, bytes)?;
        cmds.submit()?;
        copy(&staging, &mut out);
        Ok(out)
    }

    // -----------------------------------------------------------------------
    // Commands

    /// Starts recording a command buffer. The context runs one at a time:
    /// this waits for the [`Commands`] of another thread to be submitted or
    /// dropped.
    pub fn commands(&self) -> Result<Commands<'_>> {
        self.shared.alive()?;
        let submit = self.lock_submit();
        let s = &self.shared;
        let d = &s.device;
        // SAFETY: the pool's command buffer and sets are not in use: the last
        // submission was waited for (or never submitted).
        unsafe {
            s.check(
                "vkResetCommandPool",
                d.reset_command_pool(submit.pool, vk::CommandPoolResetFlags::empty()),
            )?;
            s.check(
                "vkResetDescriptorPool",
                d.reset_descriptor_pool(submit.descriptors, vk::DescriptorPoolResetFlags::empty()),
            )?;
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            s.check(
                "vkBeginCommandBuffer",
                d.begin_command_buffer(submit.cmd, &begin),
            )?;
        }
        Ok(Commands {
            ctx: self,
            submit,
            dispatches: 0,
        })
    }
}

/// Holds the objects created by [`Context::finish`] until it succeeds.
struct Partial<'a> {
    shared: &'a Arc<Shared>,
    submit: Submit,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
}

impl Partial<'_> {
    fn take(&mut self) -> (Submit, vk::DescriptorSetLayout, vk::PipelineLayout) {
        let null = Submit {
            queue: self.submit.queue,
            pool: vk::CommandPool::null(),
            cmd: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            descriptors: vk::DescriptorPool::null(),
        };
        (
            std::mem::replace(&mut self.submit, null),
            std::mem::take(&mut self.set_layout),
            std::mem::take(&mut self.pipeline_layout),
        )
    }
}

/// Destroys the objects of `submit` and the layouts (null handles are
/// skipped by Vulkan).
fn destroy(
    d: &ash::Device,
    submit: &Submit,
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
) {
    // SAFETY: the device is idle; every handle is null or owned here.
    unsafe {
        d.destroy_pipeline_layout(pipeline_layout, None);
        d.destroy_descriptor_set_layout(set_layout, None);
        d.destroy_descriptor_pool(submit.descriptors, None);
        d.destroy_fence(submit.fence, None);
        d.destroy_command_pool(submit.pool, None);
    }
}

impl Drop for Partial<'_> {
    fn drop(&mut self) {
        destroy(
            &self.shared.device,
            &self.submit,
            self.set_layout,
            self.pipeline_layout,
        );
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let d = &self.shared.device;
        // SAFETY: valid device.
        let _ = unsafe { d.device_wait_idle() };
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        let submit = self.submit.get_mut().unwrap_or_else(|e| e.into_inner());
        destroy(d, submit, self.set_layout, self.pipeline_layout);
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
}

impl Default for PipelineOptions {
    fn default() -> Self {
        PipelineOptions {
            workgroup: DEFAULT_WORKGROUP,
            subgroup: None,
        }
    }
}

/// A compute pipeline of the context.
pub struct Pipeline {
    shared: Arc<Shared>,
    pipeline: vk::Pipeline,
    name: &'static str,
    variant: Variant,
    workgroup: u32,
}

impl Pipeline {
    /// The kernel name.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// The arithmetic variant.
    pub fn variant(&self) -> Variant {
        self.variant
    }

    /// The workgroup width.
    pub fn workgroup(&self) -> u32 {
        self.workgroup
    }
}

impl fmt::Debug for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipeline")
            .field("name", &self.name)
            .field("variant", &self.variant)
            .field("workgroup", &self.workgroup)
            .finish()
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        // SAFETY: no command buffer using the pipeline is pending: commands
        // borrow it until their submission was waited for.
        unsafe { self.shared.device.destroy_pipeline(self.pipeline, None) };
    }
}

/// A storage buffer with its own memory.
pub struct Buffer {
    shared: Arc<Shared>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    len: usize,
    mapped: Option<NonNull<u8>>,
    address: Option<u64>,
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

    /// The device address (`bufferDeviceAddress`), if the device has it.
    pub fn device_address(&self) -> Option<u64> {
        self.address
    }

    /// Whether the buffer is mapped for the host.
    pub fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    /// Copies `data` into the mapping.
    ///
    /// # Safety
    ///
    /// The buffer is mapped and at least `size_of_val(data)` long, and no
    /// device work touches it meanwhile.
    unsafe fn host_write<T: Pod>(&self, data: &[T]) {
        if let Some(p) = self.mapped {
            // SAFETY: by the contract above.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().cast::<u8>(),
                    p.as_ptr(),
                    size_of_val(data),
                )
            };
        }
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
        // SAFETY: no pending command buffer uses the buffer (commands borrow
        // it until their submission was waited for); freeing mapped memory
        // unmaps it.
        unsafe {
            self.shared.device.destroy_buffer(self.buffer, None);
            self.shared.device.free_memory(self.memory, None);
        }
    }
}

/// A command buffer being recorded. It holds the context's submission lock
/// and borrows every buffer and pipeline it records until [`submit`]
/// returns; dropped without a submission, nothing runs.
///
/// [`submit`]: Commands::submit
pub struct Commands<'a> {
    ctx: &'a Context,
    submit: MutexGuard<'a, Submit>,
    dispatches: u32,
}

impl fmt::Debug for Commands<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Commands").field("dispatches", &self.dispatches).finish()
    }
}

impl<'a> Commands<'a> {
    /// Records `pipeline` over `threads` invocations, with `buffers` at
    /// bindings `0..` and `params` as push constants. The grid is
    /// `ceil(threads / workgroup)` workgroups, folded into a second
    /// dimension above `maxComputeWorkGroupCount[0]`; the kernel guards its
    /// own element count. Successive dispatches may overlap: see
    /// [`barrier`](Self::barrier).
    pub fn dispatch(
        &mut self,
        pipeline: &'a Pipeline,
        buffers: &[&'a Buffer],
        params: &[u32],
        threads: u64,
    ) -> Result<()> {
        let info = &self.ctx.info;
        if buffers.len() > MAX_BUFFERS as usize {
            return Err(Error::Invalid(format!(
                "{} buffers (at most {MAX_BUFFERS})",
                buffers.len()
            )));
        }
        let push_bytes = size_of_val(params) as u32;
        if push_bytes > MAX_PARAMS_BYTES.min(info.max_push_constants_size) {
            return Err(Error::Invalid(format!("{push_bytes} bytes of params")));
        }
        if let Some(b) =
            buffers.iter().find(|b| b.len as u64 > info.max_storage_buffer_range as u64)
        {
            return Err(Error::Unsupported(format!(
                "a storage buffer of {} bytes (maxStorageBufferRange {})",
                b.len, info.max_storage_buffer_range
            )));
        }
        if threads == 0 {
            return Ok(());
        }
        let groups = threads.div_ceil(pipeline.workgroup as u64);
        let [max_x, max_y, _] = info.max_compute_work_group_count;
        let gx = groups.min(max_x as u64);
        let gy = groups.div_ceil(gx);
        if gy > max_y as u64 {
            return Err(Error::Invalid(format!(
                "{threads} threads exceed the dispatch grid"
            )));
        }
        if self.dispatches == MAX_DISPATCHES {
            return Err(Error::Invalid(format!(
                "more than {MAX_DISPATCHES} dispatches in one submission"
            )));
        }
        let s = &self.ctx.shared;
        let d = &s.device;
        let layouts = [self.ctx.set_layout];
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.submit.descriptors)
            .set_layouts(&layouts);
        // SAFETY: valid pool and layout; the pool has room for
        // MAX_DISPATCHES sets.
        let set = s.check("vkAllocateDescriptorSets", unsafe {
            d.allocate_descriptor_sets(&alloc)
        })?[0];
        let infos: Vec<[vk::DescriptorBufferInfo; 1]> = buffers
            .iter()
            .map(|b| {
                [vk::DescriptorBufferInfo::default()
                    .buffer(b.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)]
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = infos
            .iter()
            .enumerate()
            .map(|(i, info)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(info)
            })
            .collect();
        let params_bytes: Vec<u8> = params.iter().flat_map(|w| w.to_ne_bytes()).collect();
        let cmd = self.submit.cmd;
        // SAFETY: the command buffer is recording; every handle is valid and
        // outlives the submission (borrowed for 'a).
        unsafe {
            d.update_descriptor_sets(&writes, &[]);
            d.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            d.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.ctx.pipeline_layout,
                0,
                &[set],
                &[],
            );
            if !params_bytes.is_empty() {
                d.cmd_push_constants(
                    cmd,
                    self.ctx.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &params_bytes,
                );
            }
            d.cmd_dispatch(cmd, gx as u32, gy as u32, 1);
        }
        self.dispatches += 1;
        Ok(())
    }

    /// Makes the writes of every command recorded so far visible to the
    /// commands recorded after it.
    pub fn barrier(&mut self) {
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let access = vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::TRANSFER_WRITE;
        let b = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(access)];
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

    /// Records a copy of the first `bytes` bytes of `src` to `dst`, between
    /// barriers.
    pub fn copy(&mut self, src: &'a Buffer, dst: &'a Buffer, bytes: usize) -> Result<()> {
        if bytes > src.len || bytes > dst.len {
            return Err(Error::Invalid(format!("a copy of {bytes} bytes")));
        }
        if bytes == 0 {
            return Ok(());
        }
        self.barrier();
        let region = [vk::BufferCopy::default().size(bytes as u64)];
        // SAFETY: the command buffer is recording; both buffers hold `bytes`.
        unsafe {
            self.ctx
                .shared
                .device
                .cmd_copy_buffer(self.submit.cmd, src.buffer, dst.buffer, &region)
        };
        self.barrier();
        Ok(())
    }

    /// Submits the commands and waits for them. Returns the wall time from
    /// the submission to the signalled fence.
    pub fn submit(self) -> Result<Duration> {
        let s = &self.ctx.shared;
        let d = &s.device;
        let cmd = self.submit.cmd;
        let to_host = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        // SAFETY: the command buffer is recording; queue and fence belong to
        // the context and are used under the submission lock held by `self`.
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
            s.check("vkEndCommandBuffer", d.end_command_buffer(cmd))?;
            s.check("vkResetFences", d.reset_fences(&[self.submit.fence]))?;
            let cmds = [cmd];
            let info = [vk::SubmitInfo::default().command_buffers(&cmds)];
            let t0 = Instant::now();
            s.check(
                "vkQueueSubmit",
                d.queue_submit(self.submit.queue, &info, self.submit.fence),
            )?;
            s.check(
                "vkWaitForFences",
                d.wait_for_fences(&[self.submit.fence], true, u64::MAX),
            )?;
            Ok(t0.elapsed())
        }
    }
}
