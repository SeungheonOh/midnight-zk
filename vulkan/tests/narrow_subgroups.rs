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

//! The MSM with `msm_ones` reducing its runs of 32 invocations over
//! subgroup lanes narrower than 32 and through shared memory: the path of a
//! device whose subgroups are narrower than Metal's SIMD-groups (lavapipe,
//! some integrated GPUs), forced on this device; and in subgroups of 64
//! where the device can pin them (AMD's wave64). The lanes do not depend on
//! the MSM size: one size runs every input of the shared MSM test.

use midnight_vulkan::ctx::{Vulkan, set_ones_lanes, set_ones_width};

#[test]
fn msm_is_exact_with_narrow_subgroup_lanes() {
    for lanes in [16, 8, 4] {
        // a device narrower than `lanes` reduces over fewer lanes anyway
        if !set_ones_lanes(Some(lanes)) {
            eprintln!("{lanes} lanes are above the subgroup width: skipped");
            continue;
        }
        midnight_accel::testkit::msm::msm_matches_cpu_one_size::<Vulkan>();
    }
    set_ones_lanes(None);
}

#[test]
fn msm_is_exact_in_subgroups_of_64() {
    if !set_ones_width(Some(64)) {
        eprintln!("the device cannot pin subgroups of 64: skipped");
        return;
    }
    for lanes in [None, Some(16)] {
        assert!(set_ones_lanes(lanes));
        midnight_accel::testkit::msm::msm_matches_cpu_one_size::<Vulkan>();
    }
    set_ones_lanes(None);
    set_ones_width(None);
}
