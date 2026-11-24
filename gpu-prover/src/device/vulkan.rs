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

//! The binding to `midnight-vulkan`, compiled only with the cfg
//! `midnight_vulkan_linked`: the generic binding of module `accel` on the
//! Vulkan backend. Without a Vulkan loader, a usable device or the
//! kernels' SPIR-V no device is detected.

use midnight_vulkan::{self as vulkan, Variant, Vulkan};

use super::{
    DeviceInfo,
    accel::{Accel, AccelDevice, Binding},
};

/// The Vulkan device of this process (a discrete GPU when there is one).
pub type VulkanDevice = AccelDevice<Vulkan>;

/// The binding's state for Vulkan.
static BINDING: Binding = Binding::new();

impl Accel for Vulkan {
    fn binding() -> &'static Binding {
        &BINDING
    }

    fn present() -> bool {
        vulkan::present().is_some()
    }

    fn info() -> Option<DeviceInfo> {
        let i = vulkan::present()?;
        let budget = midnight_accel::present::<Vulkan>()?.memory_budget;
        let variant = match vulkan::variant() {
            Some(Variant::Int64) => "-int64",
            Some(Variant::Int32) => "-int32",
            None => "",
        };
        let (major, minor, patch) = i.api_version;
        // `total_bytes` is the budget of the device-local heap
        Some(DeviceInfo {
            name: i.name.clone(),
            arch: format!("{:04x}:{:04x}", i.vendor_id, i.device_id),
            build_archs: format!("spirv-vulkan1.3{variant}"),
            runtime_version: format!("{major}.{minor}.{patch}"),
            driver_version: i.driver.clone(),
            total_bytes: budget,
            size_thresholds: vulkan::thresholds()
                .named()
                .into_iter()
                .map(|(c, k)| (c.to_string(), k))
                .collect(),
        })
    }
}

#[cfg(test)]
super::accel::binding_tests!(midnight_vulkan::Vulkan);
