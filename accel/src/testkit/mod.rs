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

//! The tests every backend runs, generic over the backend (feature
//! `testkit`). A backend crate instantiates each group in a test target of its
//! own, so each group runs in its own process as before:
//!
//! ```ignore
//! // tests/msm.rs of a backend crate
//! midnight_accel::backend_tests!(msm, my_backend::MyBackend);
//! ```
//!
//! Groups: `host` (the device context, the pool and request wipes, the hook
//! boundary, and the field kernels), `ntt`, `msm` and `quotient`. The tests
//! call the test kernels `field_test_fr`, `field_test_fp`, `diag_iota`,
//! `msm_test_g1` and `msm_test_mask` besides the production kernels.

pub mod common;
pub mod host;
pub mod msm;
pub mod ntt;
pub mod quotient;

/// Instantiates a group of the backend tests for a backend type: `host`,
/// `ntt`, `msm` or `quotient`.
#[macro_export]
macro_rules! backend_tests {
    (host, $b:ty) => {
        $crate::__backend_tests!(host, $b;
            fr_matches_blst, fp_matches_blst, errors_are_values,
            request_buffers_and_byte_accounting, pooled_buffers_come_back_zeroed,
            per_class_thresholds, erase_report_reads_back_every_region,
            erase_report_reads_back_the_idle_pool, short_uploads_copy_exactly_their_bytes,
            settle_releases_waits_for_every_caller, a_panicking_zeroing_is_completed_and_reported,
            hooks_outside_a_request_wipe_before_they_return, a_backend_panic_is_a_counted_decline,
            ntt_pass_takes_the_planned_threadgroups;
            ignored: ; release_ignored: );
        #[test]
        fn off_switch_declines_no_device() {
            // the test's path in the binary, which reruns it in a child process
            let path = module_path!();
            let name = match path.split_once("::") {
                Some((_, inner)) => format!("{inner}::off_switch_declines_no_device"),
                None => "off_switch_declines_no_device".to_string(),
            };
            $crate::testkit::host::off_switch_declines_no_device::<$b>(&name);
        }
    };
    (ntt, $b:ty) => {
        $crate::__backend_tests!(ntt, $b;
            roots_match_curves, fft_matches_stock, building_blocks_match_stock_domain,
            building_blocks_refuse_unreduced_scalars, declines_leave_the_buffer_unchanged,
            hooks_serve_the_domain_transforms, parallel_ifft_callers_match_stock;
            ignored: large_transform_round_trip; release_ignored: );
    };
    (msm, $b:ty) => {
        $crate::__backend_tests!(msm, $b;
            g1_kernels_match_blst, mask_is_the_cuda_vector, msm_matches_cpu,
            msm_from_rayon_pool, msm_odd_lengths, abs_boundary_scalars_match_cpu,
            wide_windows_and_chunks_match_cpu, hot_bucket_is_exact_or_declined,
            sparse_msm_declines_below_min_k, unregistered_bases_decline_by_default,
            infinity_bases_and_malformed_calls, registered_table_prefixes,
            stale_table_declines_by_default_and_one_shot_when_explicit,
            bases_outside_g1_decline, hook_commitment_matches_stock;
            ignored: ; release_ignored: );
    };
    (quotient, $b:ty) => {
        $crate::__backend_tests!(quotient, $b;
            abi_layout_matches, numerator_and_quotient_match_references,
            numerator_independent_of_threads, end_to_end_small, declines_leave_out_unchanged,
            registered_key_is_resident, mixed_length_registration_serves_each_length,
            outside_a_request_every_serve_checks, inside_a_request_checks_once,
            slot_filled_with_changed_contents_is_not_present,
            need_bytes_bounds_the_measured_peak;
            ignored: ; release_ignored: end_to_end_k16);
    };
}

/// The `#[test]` functions of [`backend_tests`].
#[doc(hidden)]
#[macro_export]
macro_rules! __backend_tests {
    ($group:ident, $b:ty; $($t:ident),* $(,)?;
     ignored: $($i:ident),*; release_ignored: $($r:ident),*) => {
        $(
            #[test]
            fn $t() {
                $crate::testkit::$group::$t::<$b>();
            }
        )*
        $(
            #[test]
            #[ignore]
            fn $i() {
                $crate::testkit::$group::$i::<$b>();
            }
        )*
        $(
            #[test]
            #[cfg_attr(debug_assertions, ignore)]
            fn $r() {
                $crate::testkit::$group::$r::<$b>();
            }
        )*
    };
}
