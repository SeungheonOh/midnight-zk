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

//! The backend of a target without Metal: no device, so every entry point
//! declines `no-device`.

use std::ptr::NonNull;

use midnight_accel::{
    Backend, DeviceInfo, Encoder, HostState, RawBuffer, Tuning,
    ctx::{Arg, Error},
};

/// No Metal on this target: the type has no value.
#[derive(Debug)]
pub enum Metal {}

/// No buffer on this target: the type has no value.
#[derive(Debug)]
pub enum NoBuffer {}

impl RawBuffer for NoBuffer {
    fn len(&self) -> usize {
        match *self {}
    }

    fn id(&self) -> u64 {
        match *self {}
    }

    fn host_ptr(&self) -> Option<NonNull<u8>> {
        match *self {}
    }
}

impl<'a> Encoder<'a, Metal> for Metal {
    fn dispatch(&mut self, _: &str, _: usize, _: &[Arg<'a, Metal>], _: &[u8]) -> Result<(), Error> {
        match *self {}
    }

    fn dispatch_with(
        &mut self,
        _: &str,
        _: usize,
        _: usize,
        _: &[Arg<'a, Metal>],
        _: &[u8],
        _: &[usize],
    ) -> Result<(), Error> {
        match *self {}
    }

    fn dispatch_groups(
        &mut self,
        _: &str,
        _: usize,
        _: usize,
        _: &[Arg<'a, Metal>],
        _: &[u8],
        _: &[usize],
    ) -> Result<(), Error> {
        match *self {}
    }
}

/// The host state of the stub backend.
static HOST: HostState<Metal> = HostState::new();

impl Backend for Metal {
    const NAME: &'static str = "metal";
    const ENV_PREFIX: &'static str = "MIDNIGHT_METAL";
    type Raw = NoBuffer;
    type Cmd<'a> = Metal;
    #[inline]
    fn tuning() -> &'static Tuning {
        &crate::tuning::TUNING
    }

    fn host() -> &'static HostState<Self> {
        &HOST
    }

    fn probe() -> Option<&'static DeviceInfo> {
        None
    }

    fn context() -> Result<&'static Self, Error> {
        Err(Error::NoDevice("Metal is not built on this target".into()))
    }

    fn unified(&self) -> bool {
        match *self {}
    }

    fn budget(&self) -> (u64, u64) {
        match *self {}
    }

    fn max_buffer_len(&self) -> u64 {
        match *self {}
    }

    fn alloc_raw(&self, _: usize) -> Option<(NoBuffer, bool)> {
        match *self {}
    }

    fn run<'a, R>(
        &'a self,
        _: impl FnOnce(&mut Self::Cmd<'a>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error> {
        match *self {}
    }

    fn max_group(&self, _: &str) -> Result<usize, Error> {
        match *self {}
    }

    fn subgroup_width(&self, _: &str) -> Result<usize, Error> {
        match *self {}
    }
}
