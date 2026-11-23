//! Hooks that always decline: the stock code runs, the proof is the stock
//! proof, and every call is counted as declined by the hook.
#![cfg(feature = "gpu")]

mod gpu_hooks_common;

use gpu_hooks_common::{assert_same_proof, digest, prove, setup, verifies, EXPECTED_PROOF_DIGEST};
use midnight_proofs::gpu::{self, G1JacobianLimbs, GpuHooks, Scalar};

fn no_msm(_: &[Scalar], _: &[G1JacobianLimbs]) -> Option<G1JacobianLimbs> {
    None
}

fn no_fft(_: &mut [Scalar], _: Scalar, _: u32) -> bool {
    false
}

fn no_ifft(_: &mut [Scalar], _: Scalar, _: u32, _: Scalar) -> bool {
    false
}

#[test]
fn gpu_hooks_decline() {
    let (params, pk) = setup();
    let stock = prove(&params, &pk);
    assert_eq!(digest(&stock), EXPECTED_PROOF_DIGEST);

    assert!(gpu::install(GpuHooks {
        msm: Some(no_msm),
        fft: Some(no_fft),
        ifft: Some(no_ifft),
    }));
    gpu::reset_stats();
    let declined = prove(&params, &pk);
    let stats = gpu::stats();
    println!("prove stats: {stats:?}");
    println!("declines: {:?}", gpu::hook_declines());

    assert_same_proof(&declined, &stock, "declining hooks vs stock");
    assert!(verifies(&params, &pk, &declined));
    assert_eq!(stats.msm_served + stats.fft_served + stats.ifft_served, 0);
    assert!(stats.msm_declined > 0 && stats.fft_declined > 0 && stats.ifft_declined > 0);
    assert!(gpu::hook_declines().iter().all(|d| d.1 == "hook"));
}
