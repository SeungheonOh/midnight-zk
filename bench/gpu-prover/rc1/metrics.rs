#[derive(Default)]
pub struct Metrics {
    pub msm: u64,
    pub ntt: u64,
    pub quotient: u64,
    pub device_errors: u64,
    pub fallbacks: u64,
    pub declines: Vec<(&'static str, &'static str, u64)>,
    pub hook_declines: Vec<(&'static str, &'static str, u64)>,
    pub h2d_copies: u64,
    pub h2d_bytes: u64,
    pub d2h_copies: u64,
    pub d2h_bytes: u64,
    pub msm_d2h_copies: u64,
    pub msm_d2h_bytes: u64,
    pub stage: [u64; 8],
    pub peak_bytes: u64,
}

/// One stderr line of where a proof's device time went: per hook class
/// (MSM, FFT, inverse FFT, quotient) the wall time inside the hook and the
/// GPU time of the calls served, then the host's copies, zeroing and command
/// buffers (`[copy, zero, zero_bg, runs, run wall, run gpu]`). Nanoseconds in,
/// milliseconds out.
#[cfg(any(feature = "rc1-bench-metal", feature = "rc1-bench-vulkan"))]
pub fn report_phases(wall: [u64; 4], gpu: [u64; 4], host: [u64; 6]) {
    let ms = |ns: u64| ns as f64 * 1e-6;
    eprintln!(
        "phases_ms msm {:.1}/{:.1} fft {:.1}/{:.1} ifft {:.1}/{:.1} quotient {:.1}/{:.1} copy {:.1} \
         zero {:.1} zero_bg {:.1} runs {} run {:.1}/{:.1} (wall/gpu)",
        ms(wall[0]),
        ms(gpu[0]),
        ms(wall[1]),
        ms(gpu[1]),
        ms(wall[2]),
        ms(gpu[2]),
        ms(wall[3]),
        ms(gpu[3]),
        ms(host[0]),
        ms(host[1]),
        ms(host[2]),
        host[3],
        ms(host[4]),
        ms(host[5]),
    );
}

/// Wall time inside each hook (MSM, FFT, inverse FFT, quotient), served or
/// declined, in nanoseconds since the last [`reset_wall`].
#[cfg(any(feature = "rc1-bench-metal", feature = "rc1-bench-vulkan"))]
static WALL: [std::sync::atomic::AtomicU64; 4] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 4];

#[cfg(any(feature = "rc1-bench-metal", feature = "rc1-bench-vulkan"))]
fn timed<R>(class: usize, f: impl FnOnce() -> R) -> R {
    let t = std::time::Instant::now();
    let r = f();
    WALL[class].fetch_add(t.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    r
}

#[cfg(any(feature = "rc1-bench-metal", feature = "rc1-bench-vulkan"))]
fn reset_wall() -> [u64; 4] {
    WALL.each_ref().map(|w| w.swap(0, std::sync::atomic::Ordering::Relaxed))
}

#[cfg(feature = "rc1-bench-cuda")]
pub mod device {
    use midnight_curves::Bls12;
    use midnight_gpu as gpu;
    use midnight_proofs::{
        gpu::{self as hooks, adapter},
        poly::{Coeff, LagrangeCoeff, kzg::params::ParamsKZG},
    };

    use super::Metrics;
    use crate::{Key, Result};

    /// The device model and its total memory in bytes.
    pub fn info() -> Result<(String, u64)> {
        let info = gpu::present().ok_or("CUDA device unavailable")?;
        Ok((info.name.clone(), info.total_bytes))
    }

    pub fn install(
        params: &ParamsKZG<Bls12>,
        pk: &Key,
    ) -> Result<([gpu::SrsHandle; 2], gpu::KeyHandle)> {
        if gpu::present().is_none() {
            return Err("CUDA device unavailable".into());
        }
        let srs = [params.bases::<Coeff>(), params.bases::<LagrangeCoeff>()]
            .map(|p| gpu::register_srs(adapter::affine(p)));
        let polys = pk.gpu_resident_polys().ok_or("unsupported quotient shape")?;
        let key = gpu::register_key(&polys.iter().map(|p| adapter::scalars(p)).collect::<Vec<_>>());
        if !hooks::install(hooks::GpuHooks {
            msm_g1: Some(gpu::msm_g1),
            fft: Some(gpu::fft),
            ifft: Some(gpu::ifft),
            quotient: Some(gpu::quotient),
        }) {
            return Err("hooks already installed".into());
        }
        Ok((srs, key))
    }

    pub struct Baseline {
        quotient: u64,
        stage: [u64; 8],
    }

    pub fn begin() -> Baseline {
        gpu::begin_request();
        gpu::reset_stats();
        hooks::reset_hook_declines();
        Baseline {
            quotient: gpu::stats().quotient,
            stage: gpu::xfer::stats().stage,
        }
    }

    pub fn finish(before: Baseline) -> Result<Metrics> {
        let stats = gpu::stats();
        let xfer = gpu::xfer::stats();
        let hook_declines = hooks::hook_declines();
        let peak_bytes = gpu::device_bytes().1;
        if !gpu::end_request() {
            return Err("device cleanup failed".into());
        }
        Ok(Metrics {
            msm: stats.msm,
            ntt: stats.ntt,
            quotient: stats.quotient - before.quotient,
            device_errors: stats.device_errors,
            fallbacks: hook_declines.iter().map(|d| d.2).sum(),
            declines: gpu::declines(),
            hook_declines,
            h2d_copies: xfer.h2d_copies,
            h2d_bytes: xfer.h2d_bytes,
            d2h_copies: xfer.d2h_copies,
            d2h_bytes: xfer.d2h_bytes,
            msm_d2h_copies: xfer.msm_d2h_copies,
            msm_d2h_bytes: xfer.msm_d2h_bytes,
            stage: std::array::from_fn(|i| {
                if i == 7 {
                    xfer.stage[i]
                } else {
                    xfer.stage[i] - before.stage[i]
                }
            }),
            peak_bytes,
        })
    }
}

/// The Metal counterpart of the CUDA `device` module above.
#[cfg(all(feature = "rc1-bench-metal", not(feature = "rc1-bench-cuda")))]
pub mod device {
    use midnight_curves::Bls12;
    use std::ffi::c_void;

    use midnight_metal::{self as metal, G1AffineLimbs, G1JacobianLimbs, Scalar};
    use midnight_proofs::{
        gpu::{self as hooks, adapter},
        poly::{
            Coeff, LagrangeCoeff, LagrangeDeltaCoeff, LagrangeDoubleDeltaCoeff,
            kzg::params::ParamsKZG,
        },
    };

    use super::{Metrics, timed};
    use crate::{Key, Result};

    fn msm_g1(s: &[Scalar], b: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
        timed(0, || metal::msm_g1(s, b))
    }

    fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
        timed(1, || metal::fft(a, omega, log_n))
    }

    fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
        timed(2, || metal::ifft(a, omega_inv, log_n, divisor))
    }

    /// # Safety
    /// As [`metal::quotient`].
    unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
        // SAFETY: the prover's job, passed through unchanged.
        timed(3, || unsafe { metal::quotient(job, out) })
    }

    /// The device model and its recommended working set in bytes.
    pub fn info() -> Result<(String, u64)> {
        let info = metal::present().ok_or("Metal device unavailable")?;
        Ok((info.name.clone(), info.recommended_max_working_set))
    }

    /// Registers the SRS vectors and the key's resident polynomials, and
    /// installs the hooks.
    pub fn install(
        params: &ParamsKZG<Bls12>,
        pk: &Key,
    ) -> Result<([metal::SrsHandle; 4], metal::KeyHandle)> {
        if !metal::device() {
            return Err("Metal device unavailable".into());
        }
        let srs = [
            params.bases::<Coeff>(),
            params.bases::<LagrangeCoeff>(),
            params.bases::<LagrangeDeltaCoeff>(),
            params.bases::<LagrangeDoubleDeltaCoeff>(),
        ]
        .map(|p| metal::register_srs(adapter::affine(p)));
        let polys = pk.gpu_resident_polys().ok_or("unsupported quotient shape")?;
        let key =
            metal::register_key(&polys.iter().map(|p| adapter::scalars(p)).collect::<Vec<_>>());
        if !hooks::install(hooks::GpuHooks {
            msm_g1: Some(msm_g1),
            fft: Some(fft),
            ifft: Some(ifft),
            quotient: Some(quotient),
        }) {
            return Err("hooks already installed".into());
        }
        Ok((srs, key))
    }

    /// The crate's counters are reset at `begin`.
    pub struct Baseline;

    pub fn begin() -> Baseline {
        metal::begin_request();
        metal::reset_stats();
        hooks::reset_hook_declines();
        super::reset_wall();
        Baseline
    }

    /// Unified memory: the transfer and staging counters are 0.
    pub fn finish(_: Baseline) -> Result<Metrics> {
        let stats = metal::stats();
        let h = metal::host_times();
        super::report_phases(
            super::reset_wall(),
            [stats.msm, stats.fft, stats.ifft, stats.quotient].map(|c| c.device_nanos),
            [
                h.copy_nanos,
                h.zero_nanos,
                h.zero_background_nanos,
                h.runs,
                h.run_nanos,
                h.run_gpu_nanos,
            ],
        );
        let hook_declines = hooks::hook_declines();
        let declines = metal::declines();
        let peak_bytes = metal::device_bytes().1;
        if !metal::end_request() {
            return Err("device cleanup failed".into());
        }
        Ok(Metrics {
            msm: stats.msm.served,
            ntt: stats.fft.served + stats.ifft.served,
            quotient: stats.quotient.served,
            device_errors: declines.iter().filter(|d| d.1 == "device-error").map(|d| d.2).sum(),
            fallbacks: hook_declines.iter().map(|d| d.2).sum(),
            declines,
            hook_declines,
            peak_bytes,
            ..Metrics::default()
        })
    }
}

/// The Vulkan counterpart of the CUDA `device` module above.
#[cfg(all(
    feature = "rc1-bench-vulkan",
    not(feature = "rc1-bench-cuda"),
    not(feature = "rc1-bench-metal")
))]
pub mod device {
    use midnight_curves::Bls12;
    use midnight_proofs::{
        gpu::{self as hooks, adapter},
        poly::{
            Coeff, LagrangeCoeff, LagrangeDeltaCoeff, LagrangeDoubleDeltaCoeff,
            kzg::params::ParamsKZG,
        },
    };
    use std::{ffi::c_void, sync::atomic::Ordering::Relaxed};

    use midnight_vulkan::{self as vulkan, G1AffineLimbs, G1JacobianLimbs, Scalar};

    use super::{Metrics, timed};
    use crate::{Key, Result};

    fn msm_g1(s: &[Scalar], b: &[G1AffineLimbs]) -> Option<G1JacobianLimbs> {
        timed(0, || vulkan::msm_g1(s, b))
    }

    fn fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
        timed(1, || vulkan::fft(a, omega, log_n))
    }

    fn ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
        timed(2, || vulkan::ifft(a, omega_inv, log_n, divisor))
    }

    /// # Safety
    /// As [`vulkan::quotient`].
    unsafe fn quotient(job: *const c_void, out: &mut [Scalar]) -> bool {
        // SAFETY: the prover's job, passed through unchanged.
        timed(3, || unsafe { vulkan::quotient(job, out) })
    }

    /// The phase line ([`super::report_phases`]), the per-kernel GPU times
    /// with `MIDNIGHT_VULKAN_PROFILE=1`, and after the first proof the
    /// pipelines' creation times.
    fn report_phases() {
        let ms = |ns: u64| ns as f64 * 1e-6;
        let st = vulkan::stats();
        let h = vulkan::host_times();
        super::report_phases(
            super::reset_wall(),
            [st.msm, st.fft, st.ifft, st.quotient].map(|c| c.device_nanos),
            [
                h.copy_nanos,
                h.zero_nanos,
                h.zero_background_nanos,
                h.runs,
                h.run_nanos,
                h.run_gpu_nanos,
            ],
        );
        let k = vulkan::device::kernel_profile();
        if !k.is_empty() {
            let list: Vec<String> =
                k.iter().map(|(n, c, t)| format!("{n} {c}x {:.1}", ms(*t))).collect();
            eprintln!("kernels_ms {}", list.join(", "));
        }
        // the pipelines are made in the first (cold) proof
        static LISTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut p = vulkan::pipeline_times();
        if !p.is_empty() && !LISTED.swap(true, Relaxed) {
            p.sort_by_key(|x| std::cmp::Reverse(x.2));
            let total: f64 = p.iter().map(|x| x.2.as_secs_f64() * 1e3).sum();
            let list: Vec<String> = p
                .iter()
                .map(|(k, wg, t)| format!("{k}/{wg} {:.1}", t.as_secs_f64() * 1e3))
                .collect();
            eprintln!("pipelines_ms total {total:.1}: {}", list.join(", "));
        }
    }

    /// The device model and the budget of its device-local memory in bytes.
    pub fn info() -> Result<(String, u64)> {
        let info = vulkan::present().ok_or("Vulkan device unavailable")?;
        let (_, total) = vulkan::mem_info().ok_or("Vulkan device unavailable")?;
        Ok((info.name.clone(), total as u64))
    }

    /// Registers the SRS vectors and the key's resident polynomials, and
    /// installs the hooks.
    pub fn install(
        params: &ParamsKZG<Bls12>,
        pk: &Key,
    ) -> Result<([vulkan::SrsHandle; 4], vulkan::KeyHandle)> {
        if !vulkan::device() {
            return Err("Vulkan device unavailable".into());
        }
        let srs = [
            params.bases::<Coeff>(),
            params.bases::<LagrangeCoeff>(),
            params.bases::<LagrangeDeltaCoeff>(),
            params.bases::<LagrangeDoubleDeltaCoeff>(),
        ]
        .map(|p| vulkan::register_srs(adapter::affine(p)));
        let polys = pk.gpu_resident_polys().ok_or("unsupported quotient shape")?;
        let key =
            vulkan::register_key(&polys.iter().map(|p| adapter::scalars(p)).collect::<Vec<_>>());
        if !hooks::install(hooks::GpuHooks {
            msm_g1: Some(msm_g1),
            fft: Some(fft),
            ifft: Some(ifft),
            quotient: Some(quotient),
        }) {
            return Err("hooks already installed".into());
        }
        Ok((srs, key))
    }

    /// The crate's counters are reset at `begin`.
    pub struct Baseline;

    pub fn begin() -> Baseline {
        vulkan::begin_request();
        vulkan::reset_stats();
        hooks::reset_hook_declines();
        super::reset_wall();
        Baseline
    }

    /// The crate counts no transfers: the transfer and staging counters are
    /// 0, also on a discrete GPU.
    pub fn finish(_: Baseline) -> Result<Metrics> {
        report_phases();
        let stats = vulkan::stats();
        let hook_declines = hooks::hook_declines();
        let declines = vulkan::declines();
        let peak_bytes = vulkan::device_bytes().1;
        if !vulkan::end_request() {
            return Err("device cleanup failed".into());
        }
        Ok(Metrics {
            msm: stats.msm.served,
            ntt: stats.fft.served + stats.ifft.served,
            quotient: stats.quotient.served,
            device_errors: declines.iter().filter(|d| d.1 == "device-error").map(|d| d.2).sum(),
            fallbacks: hook_declines.iter().map(|d| d.2).sum(),
            declines,
            hook_declines,
            peak_bytes,
            ..Metrics::default()
        })
    }
}
