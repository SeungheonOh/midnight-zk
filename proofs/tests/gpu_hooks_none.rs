//! Stock proof, no hooks installed. Runs with and without feature `gpu`: the
//! proof must be the recorded stock proof in both builds.

mod gpu_hooks_common;

use gpu_hooks_common::{assert_same_proof, digest, prove, setup, verifies, EXPECTED_PROOF_DIGEST};

#[test]
fn gpu_hooks_none() {
    let (params, pk) = setup();
    let proof = prove(&params, &pk);
    println!(
        "proof: {} bytes, blake2b-256 {}",
        proof.len(),
        digest(&proof)
    );
    assert!(verifies(&params, &pk, &proof));
    assert_same_proof(&proof, &prove(&params, &pk), "two stock runs");
    assert_eq!(digest(&proof), EXPECTED_PROOF_DIGEST);

    #[cfg(feature = "gpu")]
    {
        use midnight_proofs::gpu;
        assert!(gpu::hooks().is_none());
        assert_eq!(gpu::stats(), gpu::HookStats::default());
        assert!(gpu::hook_declines().is_empty());
        assert!(gpu::served_sizes().is_empty());
    }
}
