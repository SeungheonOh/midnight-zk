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

//! The binding to `midnight-metal`, compiled only with the cfg
//! `midnight_metal_linked`: the generic binding of module `accel` on the
//! Metal backend. On a target other than macOS no device is detected.

use midnight_metal::{self as metal, Metal};

use super::{
    DeviceInfo,
    accel::{Accel, AccelDevice, Binding},
};

/// The Metal device of this process (the system default device).
pub type MetalDevice = AccelDevice<Metal>;

/// The binding's state for Metal.
static BINDING: Binding = Binding::new();

impl Accel for Metal {
    fn binding() -> &'static Binding {
        &BINDING
    }

    fn present() -> bool {
        metal::present().is_some()
    }

    fn info() -> Option<DeviceInfo> {
        let i = metal::present()?;
        // unified memory: `total_bytes` is the recommended working set
        Some(DeviceInfo {
            name: i.name.clone(),
            arch: format!("apple{}", i.apple_family),
            build_archs: format!("msl-{}", i.msl_version),
            runtime_version: String::new(),
            driver_version: String::new(),
            total_bytes: i.recommended_max_working_set,
            size_thresholds: metal::thresholds()
                .named()
                .into_iter()
                .map(|(c, k)| (c.to_string(), k))
                .collect(),
        })
    }
}

#[cfg(test)]
super::accel::binding_tests!(midnight_metal::Metal);
