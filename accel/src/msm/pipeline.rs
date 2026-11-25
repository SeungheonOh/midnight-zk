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

//! One device MSM pass (sppark `invoke`): each chunk's kernels in one command
//! buffer, and each chunk's host collection on its own thread while the
//! device runs the next.

use std::{mem::size_of, sync::Mutex};

use blst::blst_p1;

use super::{
    Xyzz,
    collect::{add, collect},
    par::{catch, lock},
    plan::{NBITS, Plan, lg2},
    secret::{ScrubOnExit, Secret, SecretVec},
};
use crate::{
    Backend, G1AffineLimbs, Scalar,
    ctx::{Arg, Buffer, Command, Ctx, Error},
};

/// Threads of a sort group (sppark `SORT_BLOCKDIM`).
const SORT_BLOCK: usize = 1024;
/// Threads of an `msm_ones` group (sppark `BATCH_ADD_BLOCK_SIZE`).
const ONES_BLOCK: usize = 256;
/// Threads of an `msm_accumulate` group (`ACC_GROUP` of the kernel).
const ACC_GROUP: usize = 256;
/// Threads per window of `msm_integrate` (sppark `MSM_NTHREADS`).
const MSM_NTHREADS: usize = 256;
/// The largest low-digit width of the lower sort (`MAX_LOW` of the kernel).
const MAX_LOW_BITS: u32 = 12;
/// The largest top-digit width of the upper sort (`MAX_TOP` of the kernel).
const MAX_TOP_BITS: u32 = 9;

/// The SIMD-group width the `msm_ones` reduction and the sort's prefix sums
/// assume.
const SIMD_WIDTH: usize = 32;
/// The kernels whose pipelines must have [`SIMD_WIDTH`].
const KERNELS: [&str; 9] = [
    "msm_digits",
    "msm_sort_count",
    "msm_sort_scan",
    "msm_sort_scatter",
    "msm_sort_lower",
    "msm_bucket_guard",
    "msm_ones",
    "msm_accumulate",
    "msm_integrate",
];

/// Scalar mode of `msm_digits`: the masked scalars `s + r`.
pub(super) const MODE_MASKED: u32 = 1;
/// Scalar mode of `msm_digits`: the mask `r` alone (scalars not read).
pub(super) const MODE_MASK: u32 = 2;

/// Thread counts that depend on the pipelines.
#[derive(Clone, Copy, Debug)]
pub(super) struct Geometry {
    sort_block: usize,
    ones_block: usize,
    acc_block: usize,
    /// Upper-sort groups per window ([`crate::Tuning::msm_sort_groups`]).
    sort_groups: u32,
    /// Groups of `msm_ones` ([`crate::Tuning::msm_ones_groups`]).
    ones_groups: usize,
    /// [`crate::Tuning::msm_hot_bucket_limit`].
    hot_limit: u32,
    /// `msm_integrate` threads per workgroup (a power of two).
    nthr: usize,
    /// `msm_integrate` workgroups per window (a power of two): `nthr *
    /// int_groups` runs per window.
    int_groups: usize,
}

impl Geometry {
    /// The thread counts of `pl` on this device, after checking what the
    /// kernels rely on (threadgroup sizes, sort widths, SIMD-group width).
    pub(super) fn of<B: Backend>(c: &Ctx<B>, pl: &Plan) -> Result<Geometry, Error> {
        let max = |k: &str| c.max_group(k);
        let pow2_floor = |x: usize| if x == 0 { 0 } else { 1usize << lg2(x) };
        let sort_block = [
            "msm_sort_count",
            "msm_sort_scan",
            "msm_sort_scatter",
            "msm_sort_lower",
        ]
        .iter()
        .map(|k| max(k))
        .try_fold(SORT_BLOCK, |a, m| m.map(|m| a.min(pow2_floor(m))))?;
        let ones_block = pow2_floor(max("msm_ones")?.min(ONES_BLOCK));
        let acc_block = pow2_floor(max("msm_accumulate")?.min(ACC_GROUP));
        let nthr = pow2_floor(max("msm_integrate")?.min(MSM_NTHREADS));
        // more workgroups only while each run keeps at least two buckets
        let mut int_groups = pow2_floor(B::tuning().msm_integrate_groups.max(1) as usize);
        while int_groups > 1 && lg2(nthr * int_groups) >= pl.bits {
            int_groups /= 2;
        }
        if sort_block < 32
            || ones_block < 32
            || acc_block < 32
            || nthr < 2
            || lg2(nthr) >= pl.bits
        {
            return Err(Error::Device(format!(
                "msm: threadgroup sizes {sort_block}/{ones_block}/{acc_block}/{nthr} too small"
            )));
        }
        if pl.low_bits > MAX_LOW_BITS || pl.top_bits > MAX_TOP_BITS {
            return Err(Error::Device("msm: window too wide for the sort".into()));
        }
        simd_width_ok(c)?;
        let t = B::tuning();
        Ok(Geometry {
            sort_block,
            ones_block,
            acc_block,
            sort_groups: t.msm_sort_groups,
            ones_groups: t.msm_ones_groups,
            hot_limit: t.msm_hot_bucket_limit,
            nthr,
            int_groups,
        })
    }

    /// The records `msm_ones` writes: one per 32 threads of its grid.
    pub(super) fn ones_slots(&self) -> usize {
        self.ones_groups * self.ones_block / SIMD_WIDTH
    }
}

/// Whether every MSM pipeline has SIMD-groups of [`SIMD_WIDTH`], checked once
/// per process. Another width is a `device-error` decline.
fn simd_width_ok<B: Backend>(c: &Ctx<B>) -> Result<(), Error> {
    let once = &B::host().msm.simd_ok;
    let ok = match once.get() {
        Some(&ok) => ok,
        None => {
            let mut ok = true;
            for k in KERNELS {
                ok &= c.subgroup_width(k)? == SIMD_WIDTH;
            }
            *once.get_or_init(|| ok)
        }
    };
    if ok {
        Ok(())
    } else {
        Err(Error::Device(
            "msm: a SIMD-group width other than 32".into(),
        ))
    }
}

/// The device buffers of an MSM.
pub(super) struct Bufs<'a, B: Backend> {
    /// The points: a registered table, or the caller's copied for the call.
    pub(super) points: &'a Buffer<B>,
    pub(super) scalars: &'a Buffer<B>,
    /// `nwins` rows of `stride` digits, then of sorted point indices.
    pub(super) digits: &'a Buffer<B>,
    /// The sort's (key, value) pairs of one pass.
    pub(super) temps: &'a Buffer<B>,
    /// The upper sort's per-group counters of one pass.
    pub(super) counts: &'a Buffer<B>,
    /// `nwins` rows of bucket end offsets.
    pub(super) hist: &'a Buffer<B>,
    /// `nwins` rows of bucket sums, then of `msm_integrate` partial sums.
    pub(super) buckets: &'a Buffer<B>,
    /// The partial sums of `msm_ones`.
    pub(super) ones: &'a Buffer<B>,
    /// Set by `msm_bucket_guard` when a bucket exceeds the hot-bucket limit
    /// ([`crate::Tuning::msm_hot_bucket_limit`]); reset by `msm_digits`.
    pub(super) hot: &'a Buffer<B>,
}

/// Byte sizes of the scratch buffers of `pl`, in [`Bufs`] field order from
/// `digits` to `hot`.
pub(super) fn scratch_bytes<B: Backend>(pl: &Plan, g: &Geometry) -> [usize; 7] {
    let ntop = 1usize << pl.top_bits;
    let rows = (pl.nwins as usize) << pl.bits;
    let temps_bytes = pl.pass as usize * pl.stride * 8;
    [
        pl.nwins as usize * pl.stride * 4,
        if B::MSM_LENGTH_BINS {
            temps_bytes.max(rows * 4)
        } else {
            temps_bytes
        },
        pl.pass as usize * ntop * g.sort_groups as usize * 4,
        rows * 4,
        rows * size_of::<Xyzz>(),
        g.ones_slots() * size_of::<Xyzz>(),
        if B::MSM_LENGTH_BINS { 128 * 4 } else { 4 },
    ]
}

/// The kernel parameters of one chunk (the parameter blocks of the kernels).
struct ChunkParams {
    /// `DigitsParams`.
    digits: [u32; 6],
    /// `SortParams` of each sort pass.
    sort: Vec<[u32; 9]>,
    /// `AccParams`.
    acc: [u32; 4],
    /// `IntParams`.
    int: [u32; 4],
}

fn chunk_params(pl: &Plan, g: &Geometry, mode: u32, h_off: usize, num: usize) -> ChunkParams {
    let sort = (0..pl.nwins)
        .step_by(pl.pass as usize)
        .map(|w0| {
            [
                num as u32,
                pl.stride as u32,
                w0,
                pl.nwins,
                g.sort_groups,
                pl.top_bits,
                pl.low_bits,
                pl.bits,
                pl.top - 1,
            ]
        })
        .collect();
    ChunkParams {
        digits: [
            num as u32,
            h_off as u32,
            pl.nwins,
            pl.wbits,
            pl.stride as u32,
            mode,
        ],
        sort,
        acc: [pl.nwins, pl.bits, pl.stride as u32, g.hot_limit],
        int: [pl.nwins, pl.wbits, NBITS, g.int_groups as u32],
    }
}

fn enc_digits<'a, B: Backend>(
    cmd: &mut Command<'_, 'a, B>,
    b: &Bufs<'a, B>,
    p: &'a ChunkParams,
    h_off: usize,
    num: usize,
) -> Result<(), Error> {
    // the mask mode does not read the scalars; binding them keeps the input
    // distinct from the output buffer
    let sc = if p.digits[5] == MODE_MASK {
        Arg::Buffer(b.scalars)
    } else {
        Arg::BufferAt(b.scalars, h_off * size_of::<Scalar>())
    };
    cmd.dispatch(
        "msm_digits",
        num,
        &[sc, Arg::Buffer(b.digits), Arg::Buffer(b.hot)],
        &p.digits,
    )
}

fn enc_sort<'a, B: Backend>(
    cmd: &mut Command<'_, 'a, B>,
    b: &Bufs<'a, B>,
    p: &'a ChunkParams,
    pl: &Plan,
    g: &Geometry,
) -> Result<(), Error> {
    let ntop = 1usize << pl.top_bits;
    for sp in &p.sort {
        // windows of this pass (`sp[2]`, its first window, is below `nwins`)
        let pw = (pl.nwins - sp[2]).min(pl.pass) as usize;
        let groups = pw * g.sort_groups as usize;
        cmd.dispatch_groups(
            "msm_sort_count",
            groups,
            g.sort_block,
            &[Arg::Buffer(b.digits), Arg::Buffer(b.counts)],
            sp,
            &[],
        )?;
        cmd.dispatch_groups(
            "msm_sort_scan",
            pw,
            g.sort_block,
            &[Arg::Buffer(b.counts), Arg::Buffer(b.hist)],
            sp,
            &[],
        )?;
        cmd.dispatch_groups(
            "msm_sort_scatter",
            groups,
            g.sort_block,
            &[
                Arg::Buffer(b.digits),
                Arg::Buffer(b.counts),
                Arg::Buffer(b.temps),
            ],
            sp,
            &[],
        )?;
        cmd.dispatch_groups(
            "msm_sort_lower",
            pw * ntop,
            g.sort_block,
            &[
                Arg::Buffer(b.digits),
                Arg::Buffer(b.temps),
                Arg::Buffer(b.hist),
            ],
            sp,
            &[],
        )?;
    }
    Ok(())
}

/// The hot-bucket guard, `msm_ones` and `msm_accumulate`.
fn enc_accumulate<'a, B: Backend>(
    cmd: &mut Command<'_, 'a, B>,
    b: &Bufs<'a, B>,
    p: &'a ChunkParams,
    pl: &Plan,
    g: &Geometry,
    h_off: usize,
) -> Result<(), Error> {
    let pts = Arg::BufferAt(b.points, h_off * size_of::<G1AffineLimbs>());
    let total = (pl.nwins as usize) << pl.bits;
    if B::MSM_LENGTH_BINS {
        cmd.dispatch(
            "msm_bin_count",
            total,
            &[
                Arg::Buffer(b.buckets),
                Arg::Buffer(b.hist),
                Arg::Buffer(b.hot),
            ],
            &p.acc,
        )?;
        cmd.dispatch("msm_bin_scan", 1, &[Arg::Buffer(b.hot)], &())?;
        cmd.dispatch(
            "msm_bin_scatter",
            total,
            &[
                Arg::Buffer(b.temps),
                Arg::Buffer(b.hist),
                Arg::Buffer(b.hot),
            ],
            &p.acc,
        )?;
        cmd.dispatch_groups(
            "msm_ones",
            g.ones_groups,
            g.ones_block,
            &[
                Arg::Buffer(b.ones),
                pts,
                Arg::Buffer(b.digits),
                Arg::Buffer(b.hist),
            ],
            &(),
            &[],
        )?;
        cmd.dispatch_with(
            "msm_accumulate",
            total,
            g.acc_block,
            &[
                Arg::Buffer(b.buckets),
                pts,
                Arg::Buffer(b.digits),
                Arg::Buffer(b.hist),
                Arg::Buffer(b.hot),
                Arg::Buffer(b.temps),
            ],
            &p.acc,
            &[],
        )?;
        return cmd.dispatch_with(
            "msm_accumulate_fallback",
            total,
            g.acc_block,
            &[
                Arg::Buffer(b.buckets),
                pts,
                Arg::Buffer(b.digits),
                Arg::Buffer(b.hist),
                Arg::Buffer(b.hot),
            ],
            &p.acc,
            &[],
        );
    }
    cmd.dispatch(
        "msm_bucket_guard",
        total,
        &[Arg::Buffer(b.hist), Arg::Buffer(b.hot)],
        &p.acc,
    )?;
    cmd.dispatch_groups(
        "msm_ones",
        g.ones_groups,
        g.ones_block,
        &[
            Arg::Buffer(b.ones),
            pts,
            Arg::Buffer(b.digits),
            Arg::Buffer(b.hist),
        ],
        &(),
        &[],
    )?;
    cmd.dispatch_with(
        "msm_accumulate",
        total,
        g.acc_block,
        &[
            Arg::Buffer(b.buckets),
            pts,
            Arg::Buffer(b.digits),
            Arg::Buffer(b.hist),
            Arg::Buffer(b.hot),
        ],
        &p.acc,
        &[],
    )
}

fn enc_integrate<'a, B: Backend>(
    cmd: &mut Command<'_, 'a, B>,
    b: &Bufs<'a, B>,
    p: &'a ChunkParams,
    pl: &Plan,
    g: &Geometry,
) -> Result<(), Error> {
    cmd.dispatch_groups(
        "msm_integrate",
        pl.nwins as usize * g.int_groups,
        g.nthr,
        &[Arg::Buffer(b.buckets)],
        &p.int,
        &[],
    )
}

/// A chunk's partial sums on the host: `2 * nthr` records per window from
/// `msm_integrate`, and the `msm_ones` records.
type Partials = (SecretVec<Xyzz>, SecretVec<Xyzz>);

/// One device MSM pass (sppark `invoke`) over the first `pl.n` points with the
/// scalars of `mode`. Returns the sum and the GPU time.
///
/// Each chunk's partial sums are combined on a host thread while the device
/// runs the next chunk. Any chunk or collection failure is an error, returned
/// after every started collection has been joined.
pub(super) fn run<B: Backend>(
    c: &Ctx<B>,
    b: &Bufs<'_, B>,
    pl: &Plan,
    g: &Geometry,
    mode: u32,
) -> Result<(Secret<blst_p1>, u64), Error> {
    // each chunk's collection inputs, taken by its thread (or by this thread
    // when that thread could not be started)
    type Inputs = Mutex<Option<Partials>>;
    let inputs: Vec<Inputs> = (0..pl.batch).map(|_| Mutex::new(None)).collect();
    let nthr = g.nthr * g.int_groups;
    if pl.batch == 1 {
        let (ns, (rows, ones)) = run_chunk(c, b, pl, g, mode, 0, pl.n)?;
        let _scrub = ScrubOnExit;
        let out = collect(rows, ones, pl, nthr)
            .ok_or_else(|| Error::Device("msm: host collect failed".into()))?;
        return Ok((out, ns));
    }
    let collect_inputs = |i: &Inputs| {
        let _scrub = ScrubOnExit;
        lock(i).take().and_then(|(rows, ones)| collect(rows, ones, pl, nthr))
    };
    std::thread::scope(|sc| {
        enum Pending<'s> {
            Thread(std::thread::ScopedJoinHandle<'s, Option<Secret<blst_p1>>>),
            Done(Option<Secret<blst_p1>>),
        }
        let mut pending = Vec::with_capacity(pl.batch);
        let mut gpu = 0u64;
        let mut h_off = 0usize;
        let mut err = None;
        for slot in &inputs {
            let num = pl.stride.min(pl.n - h_off);
            match run_chunk(c, b, pl, g, mode, h_off, num) {
                Ok((ns, partials)) => {
                    gpu += ns;
                    *lock(slot) = Some(partials);
                }
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
            let job = move || collect_inputs(slot);
            match std::thread::Builder::new().spawn_scoped(sc, job) {
                Ok(h) => pending.push(Pending::Thread(h)),
                Err(_) => pending.push(Pending::Done(catch(job).flatten())),
            }
            h_off += num;
        }
        let mut out = Secret(blst_p1::default());
        for h in pending {
            let r = match h {
                Pending::Thread(h) => h.join().ok().flatten(),
                Pending::Done(r) => r,
            };
            match r {
                Some(p) => add(&mut out, &p),
                None => err = err.or(Some(Error::Device("msm: host collect failed".into()))),
            }
        }
        match err {
            Some(e) => Err(e),
            None => Ok((out, gpu)),
        }
    })
}

/// Runs the chunk of `num` points from `h_off` in one command buffer and
/// copies out its partial sums before the next chunk reuses the buffers.
fn run_chunk<B: Backend>(
    c: &Ctx<B>,
    b: &Bufs<'_, B>,
    pl: &Plan,
    g: &Geometry,
    mode: u32,
    h_off: usize,
    num: usize,
) -> Result<(u64, Partials), Error> {
    let p = chunk_params(pl, g, mode, h_off, num);
    let ((), ns) = c.run(|cmd| {
        enc_digits(cmd, b, &p, h_off, num)?;
        enc_sort(cmd, b, &p, pl, g)?;
        enc_accumulate(cmd, b, &p, pl, g, h_off)?;
        enc_integrate(cmd, b, &p, pl, g)?;
        cmd.download(b.hot)?;
        cmd.download(b.buckets)?;
        cmd.download(b.ones)
    })?;
    // a bucket above the bound skipped accumulation: decline with the generic
    // message of any failed command buffer
    if b.hot.host_read(|h: &[u32]| h.first().is_none_or(|&h| h != 0))? {
        return Err(Error::Device("msm: a command buffer failed".into()));
    }
    let partials = copy_partials(b, pl, g).ok_or_else(|| Error::Memory("msm scratch".into()))?;
    Ok((ns, partials))
}

/// The partial sums of the chunk just run, copied into guards; `None` when a
/// device buffer is shorter than the plan needs.
fn copy_partials<B: Backend>(b: &Bufs<'_, B>, pl: &Plan, g: &Geometry) -> Option<Partials> {
    // each workgroup writes its `2 * nthr` records at the start of its part
    // of the row, in run order
    let per_group = 2 * g.nthr;
    let row_len = 1usize << pl.bits;
    let part = row_len / g.int_groups;
    let mut rows = SecretVec::with_capacity(pl.nwins as usize * per_group * g.int_groups);
    let mut ok = b
        .buckets
        .host_read(|bk: &[Xyzz]| {
            (0..pl.nwins as usize).fold(true, |ok, y| {
                (0..g.int_groups).fold(ok, |ok, ig| {
                    let at = y * row_len + ig * part;
                    ok & bk.get(at..at + per_group).is_some_and(|r| rows.extend_within(r))
                })
            })
        })
        .ok()?;
    let mut ones = SecretVec::with_capacity(g.ones_slots());
    ok &= b
        .ones
        .host_read(|o: &[Xyzz]| o.get(..g.ones_slots()).is_some_and(|o| ones.extend_within(o)))
        .ok()?;
    ok.then_some((rows, ones))
}
