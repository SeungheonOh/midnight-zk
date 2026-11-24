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

//! Vulkan compute backend for the accelerator hooks of `midnight-proofs`:
//! BLS12-381 field and G1 arithmetic in GLSL compute shaders, and the device
//! context that runs them.
//!
//! The Vulkan loader is loaded at run time: without one, [`Context::new`]
//! returns [`Error::Loader`] and the caller runs the CPU code.
//!
//! # Shaders
//!
//! `build.rs` compiles `shaders/*.comp` to SPIR-V for Vulkan 1.3 and embeds
//! it. Field elements are little-endian 32-bit limbs of the Montgomery form,
//! byte for byte the host's blst limbs. Every kernel exists in two
//! arithmetic [`Variant`]s, chosen per pipeline: 64-bit accumulators
//! (`shaderInt64`) or 32-bit words only.
//!
//! The kernels of the feature `test-support` are the test and measurement
//! kernels of this crate's tests.

#![deny(missing_docs)]
#![deny(missing_debug_implementations)]

pub mod ctx;

pub use ctx::{
    Buffer, Commands, Context, DeviceInfo, Error, Pipeline, PipelineOptions, Pod, Result, Variant,
};

/// Helpers of the crate's tests (feature `test-support`).
#[cfg(any(test, feature = "test-support"))]
#[doc(hidden)]
pub mod test_support {
    use std::{sync::OnceLock, time::Duration};

    use crate::{Buffer, Context, PipelineOptions, Result, Variant};

    /// The process-wide test context; panics when there is no device.
    pub fn context() -> &'static Context {
        static CTX: OnceLock<Result<Context>> = OnceLock::new();
        match CTX.get_or_init(Context::new) {
            Ok(c) => c,
            Err(e) => panic!("no usable Vulkan device: {e}"),
        }
    }

    /// Runs `kernel` once over `threads` invocations with the default
    /// pipeline options.
    pub fn run(
        kernel: &str,
        variant: Variant,
        buffers: &[&Buffer],
        params: &[u32],
        threads: u64,
    ) -> Result<Duration> {
        let c = context();
        let p = c.pipeline(kernel, variant, PipelineOptions::default())?;
        let mut cmds = c.commands()?;
        cmds.dispatch(&p, buffers, params, threads)?;
        cmds.submit()
    }
}

#[cfg(test)]
mod tests;
