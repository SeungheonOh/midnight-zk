//! CPU "echo" hooks that compute the stock results by other CPU code: the
//! proof must be byte-identical to the stock proof, and the hooks must have
//! served MSMs, FFTs and inverse FFTs. A last pass corrupts the MSM result to
//! show that the prover uses what the hook returns.
#![cfg(feature = "gpu")]

mod gpu_hooks_common;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use gpu_hooks_common::{
    assert_same_proof, digest, prove, setup, verifies, EXPECTED_PROOF_DIGEST, K,
};
use group::{prime::PrimeCurveAffine, Curve, Group};
use midnight_curves::{fft::best_fft, msm::msm_best, G1Affine, G1Projective};
use midnight_proofs::gpu::{self, G1JacobianLimbs, GpuHooks, Scalar};

static CORRUPT_MSM: AtomicBool = AtomicBool::new(false);
static MSM_BASES: Mutex<Vec<(usize, usize)>> = Mutex::new(Vec::new());

fn echo_msm(scalars: &[Scalar], bases: &[G1JacobianLimbs]) -> Option<G1JacobianLimbs> {
    MSM_BASES.lock().unwrap().push((bases.as_ptr() as usize, bases.len()));
    if CORRUPT_MSM.load(Ordering::Relaxed) {
        return Some(gpu::point_limbs(&[G1Projective::generator()])[0]);
    }
    let scalars = gpu::scalars_of_limbs(scalars);
    let bases = gpu::points_of_limbs(bases);
    // Affine Pippenger instead of the stock projective `blst` MSM: the same
    // point, usually in other Jacobian coordinates.
    let mut affine = vec![G1Affine::identity(); bases.len()];
    G1Projective::batch_normalize(bases, &mut affine);
    let r = msm_best(scalars, &affine);
    Some(gpu::point_limbs(&[r])[0])
}

fn echo_fft(a: &mut [Scalar], omega: Scalar, log_n: u32) -> bool {
    let omega = gpu::scalars_of_limbs(&[omega])[0];
    best_fft(gpu::scalars_of_limbs_mut(a), omega, log_n);
    true
}

fn echo_ifft(a: &mut [Scalar], omega_inv: Scalar, log_n: u32, divisor: Scalar) -> bool {
    let c = [omega_inv, divisor];
    let c = gpu::scalars_of_limbs(&c);
    let a = gpu::scalars_of_limbs_mut(a);
    best_fft(a, c[0], log_n);
    for x in a.iter_mut() {
        *x *= c[1];
    }
    true
}

#[test]
fn gpu_hooks_echo() {
    let (params, pk) = setup();
    let stock = prove(&params, &pk);
    assert_eq!(digest(&stock), EXPECTED_PROOF_DIGEST);
    assert_eq!(gpu::stats(), gpu::HookStats::default());

    assert!(gpu::install(GpuHooks {
        msm: Some(echo_msm),
        fft: Some(echo_fft),
        ifft: Some(echo_ifft),
    }));
    assert!(
        !gpu::install(GpuHooks::default()),
        "install must succeed once only"
    );

    gpu::reset_stats();
    MSM_BASES.lock().unwrap().clear();
    let hooked = prove(&params, &pk);
    let stats = gpu::stats();
    let mut sizes = gpu::served_sizes();
    sizes.sort();
    println!("K = {K}, prove stats: {stats:?}");
    println!("served (class, length, count): {sizes:?}");
    println!("declines: {:?}", gpu::hook_declines());

    let g = (params.g().as_ptr() as usize, params.g().len());
    let gl = (
        params.g_lagrange().as_ptr() as usize,
        params.g_lagrange().len(),
    );
    let calls = MSM_BASES.lock().unwrap().clone();
    let on_g = calls.iter().filter(|c| c.0 == g.0).count();
    let on_gl = calls.iter().filter(|c| c.0 == gl.0).count();
    println!(
        "prover MSMs: {} total, {} on params.g (len {}), {} on params.g_lagrange (len {}), lengths {:?}",
        calls.len(),
        on_g,
        g.1,
        on_gl,
        gl.1,
        calls.iter().map(|c| c.1).collect::<Vec<_>>()
    );

    assert_same_proof(&hooked, &stock, "echo hooks vs stock");
    assert!(stats.msm_served > 0 && stats.fft_served > 0 && stats.ifft_served > 0);
    assert_eq!(
        stats.msm_declined + stats.fft_declined + stats.ifft_declined,
        0
    );
    assert_eq!(
        on_g + on_gl,
        calls.len(),
        "a prover MSM did not use the params' bases"
    );
    assert!(on_g > 0 && on_gl > 0);
    assert!(calls.iter().all(|c| c.1 <= g.1));

    // The verifier's MSMs reach the hook with temporary vectors.
    gpu::reset_stats();
    MSM_BASES.lock().unwrap().clear();
    assert!(verifies(&params, &pk, &hooked));
    let calls = MSM_BASES.lock().unwrap().clone();
    println!(
        "verifier MSMs: {} (lengths {:?}), on params slices: {}",
        calls.len(),
        calls.iter().map(|c| c.1).collect::<Vec<_>>(),
        calls.iter().filter(|c| c.0 == g.0 || c.0 == gl.0).count()
    );

    // The prover uses what the hook returns.
    CORRUPT_MSM.store(true, Ordering::Relaxed);
    let corrupt = prove(&params, &pk);
    CORRUPT_MSM.store(false, Ordering::Relaxed);
    assert_ne!(corrupt, stock);
    assert!(!verifies(&params, &pk, &corrupt));
}
