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

//! What a GPU backend provides: the device, its buffers and the encoding of
//! dispatches. Everything else (admission, accounting, the pool, request
//! wipes, the MSM, NTT and quotient drivers) is generic over [`Backend`].
//!
//! Dispatch contract: a kernel takes its buffers at bindings `0..n` in the
//! order of the argument list, then at most one block of parameters (`n`; on
//! Metal `[[buffer(n)]]`, in GLSL a push-constant block), then the
//! shared-memory blocks in the order of their lengths.

use std::ptr::NonNull;

use crate::{
    ctx::{Arg, Error, ReadBack},
    host::HostState,
};

/// The device facts the shared code reads, without making the context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The device's name.
    pub name: String,
    /// Whether device memory is host memory (no copies between them).
    pub unified_memory: bool,
    /// The bytes the device should keep allocated at most.
    pub memory_budget: u64,
    /// The largest single buffer in bytes.
    pub max_buffer_length: u64,
    /// The most threads of one workgroup.
    pub max_threads_per_group: u64,
}

/// A GPU backend: one process-wide device context and its host state.
pub trait Backend: Sized + Send + Sync + 'static {
    /// The backend's short name: thread names and messages
    /// (`midnight-{NAME}-release`).
    const NAME: &'static str;
    /// The prefix of the backend's environment variables (`{ENV_PREFIX}=off`,
    /// `{ENV_PREFIX}_MIN_K`, ...).
    const ENV_PREFIX: &'static str;
    /// A device buffer.
    type Raw: RawBuffer;
    /// The encoder of one command buffer ([`Backend::run`]).
    type Cmd<'a>: Encoder<'a, Self>;

    /// The host state of this backend: one static per backend.
    fn host() -> &'static HostState<Self>;
    /// The device's facts, without making the context. `None`: no device.
    fn probe() -> Option<&'static DeviceInfo>;
    /// The context, made at the first call. A failure is kept: later calls
    /// return it without trying again.
    fn context() -> Result<&'static Self, Error>;
    /// Whether device memory is host memory: every buffer has a host pointer.
    fn unified(&self) -> bool;
    /// `(used, total)` device bytes: what is allocated now and what the
    /// device should keep allocated at most.
    fn budget(&self) -> (u64, u64);
    /// The largest single buffer in bytes.
    fn max_buffer_len(&self) -> u64;
    /// A new device buffer of `len` bytes (`len >= 16`), and whether it is
    /// known to be zero-filled. `None`: the allocation failed.
    fn alloc_raw(&self, len: usize) -> Option<(Self::Raw, bool)>;
    /// Encodes the dispatches of `f` into one command buffer, submits it and
    /// waits. Returns `f`'s value and the GPU time in nanoseconds. An error
    /// from `f` is returned only after the encoded work completed, so no
    /// buffer is in use when this returns; a failed command buffer is
    /// [`Error::Device`].
    fn run<'a, R>(
        &'a self,
        f: impl FnOnce(&mut Self::Cmd<'a>) -> Result<R, Error>,
    ) -> Result<(R, u64), Error>;
    /// The most threads of one workgroup of `kernel`.
    fn max_group(&self, kernel: &str) -> Result<usize, Error>;
    /// The subgroup (SIMD-group) width of `kernel`.
    fn subgroup_width(&self, kernel: &str) -> Result<usize, Error>;
}

/// A device buffer of a [`Backend`]. The shared code owns it exclusively
/// while it zeroes or reads it back.
pub trait RawBuffer: Send + Sync + 'static {
    /// The allocation's length in bytes.
    fn len(&self) -> usize;
    /// Whether the allocation is empty (never, for a buffer of the pool).
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// An identifier of the allocation, unique among live buffers.
    fn id(&self) -> u64;
    /// The host address of the contents, when the host can read and write
    /// them in place.
    fn host_ptr(&self) -> Option<NonNull<u8>>;
    /// Zeroes the whole allocation. The caller has exclusive use of the
    /// buffer (no host reference, no GPU work). Never panics.
    fn zero_all(&self) -> Result<(), Error> {
        match self.host_ptr() {
            Some(p) => {
                crate::ctx::host_zero(p, self.len());
                Ok(())
            }
            None => Err(Error::Device("no host mapping to zero".into())),
        }
    }
    /// Reads the whole allocation back: whether every byte is zero. The
    /// caller has exclusive use of the buffer. Never panics.
    fn read_back_zero(&self) -> ReadBack {
        match self.host_ptr() {
            Some(p) if crate::ctx::host_reads_zero(p, self.len()) => ReadBack::Zero,
            Some(_) => ReadBack::NonZero,
            None => ReadBack::Unreadable,
        }
    }
    /// Called once when the shared code releases the buffer, before it is
    /// dropped.
    fn on_release(&self) {}
}

/// The encoder of one command buffer. Dispatches run in order, each seeing the
/// writes of the earlier ones.
pub trait Encoder<'a, B: Backend> {
    /// `threads` threads in workgroups of the backend's choice (at most 256);
    /// the last group may be partial, so kernels check their index.
    fn dispatch(
        &mut self,
        kernel: &str,
        threads: usize,
        bufs: &[Arg<'a, B>],
        params: &[u8],
    ) -> Result<(), Error>;
    /// `threads` threads in groups of `group`, with shared memory of
    /// `shared[i]` bytes in block `i`.
    fn dispatch_with(
        &mut self,
        kernel: &str,
        threads: usize,
        group: usize,
        bufs: &[Arg<'a, B>],
        params: &[u8],
        shared: &[usize],
    ) -> Result<(), Error>;
    /// `groups` whole groups of `group` threads, with shared memory as
    /// [`Encoder::dispatch_with`].
    fn dispatch_groups(
        &mut self,
        kernel: &str,
        groups: usize,
        group: usize,
        bufs: &[Arg<'a, B>],
        params: &[u8],
        shared: &[usize],
    ) -> Result<(), Error>;
}
