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

//! The host side of the G1 MSM that needs no device: the window plan and the
//! device share, the checks on the bases, the effective size of the scalars,
//! the host collection of the partial sums, and the guards that zero values
//! derived from the scalars.

pub mod checks;
pub mod collect;
pub mod effective;
pub mod par;
pub mod plan;
pub mod secret;

/// A device XYZZ record (`g1::Xyzz`): `X, Y, ZZZ, ZZ`, six limbs each.
pub type Xyzz = [u64; 24];
