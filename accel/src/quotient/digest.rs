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

//! The content digest of proving-key polynomials, as the CUDA crate's
//! `ident::digest`.

use std::sync::atomic::Ordering::Relaxed;

use crate::Scalar;

/// Bytes per hashed chunk of the digest.
const CHUNK: usize = 16 << 20;

fn sha256(b: &[u8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    // SAFETY: valid input and output buffers.
    unsafe { blst::blst_sha256(out.as_mut_ptr(), b.as_ptr(), b.len()) };
    out
}

/// The content digest of each of `vs`: SHA-256 of the length and of each
/// 16 MiB chunk's SHA-256, as the CUDA crate's `ident::digest`. All chunks are
/// hashed together on all cores.
pub fn digest_many(vs: &[&[Scalar]]) -> Vec<[u8; 32]> {
    let bytes: Vec<&[u8]> = vs
        .iter()
        .map(|v| {
            // SAFETY: plain limbs, no padding.
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, size_of_val(*v)) }
        })
        .collect();
    let chunks: Vec<&[u8]> = bytes.iter().flat_map(|b| b.chunks(CHUNK)).collect();
    let mut ds = vec![[0u8; 32]; chunks.len()];
    let threads = std::thread::available_parallelism()
        .map_or(4, |t| t.get())
        .clamp(1, 16)
        .min(chunks.len());
    if threads <= 1 {
        for (d, c) in ds.iter_mut().zip(&chunks) {
            *d = sha256(c);
        }
    } else {
        // each thread takes the next chunk
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| {
                        let mut done = Vec::new();
                        loop {
                            let i = next.fetch_add(1, Relaxed);
                            let Some(c) = chunks.get(i) else { break };
                            done.push((i, sha256(c)));
                        }
                        done
                    })
                })
                .collect();
            for h in hs {
                // `sha256` does not panic; only a failed allocation can, and the
                // hook turns that into a decline
                for (i, d) in h.join().expect("a digest thread") {
                    ds[i] = d;
                }
            }
        });
    }
    // each polynomial's chunk digests are the next `n` of `ds`
    let mut at = 0;
    bytes
        .iter()
        .map(|b| {
            let n = b.len().div_ceil(CHUNK);
            let mut top = Vec::with_capacity(8 + 32 * n);
            top.extend_from_slice(&(b.len() as u64).to_le_bytes());
            for d in &ds[at..at + n] {
                top.extend_from_slice(d);
            }
            at += n;
            sha256(&top)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The digest of each polynomial, one at a time, as the definition
    /// reads: SHA-256 of the length and of each chunk's SHA-256.
    fn reference(v: &[Scalar]) -> [u8; 32] {
        // SAFETY: plain limbs, no padding.
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, size_of_val(v)) };
        let mut top = (b.len() as u64).to_le_bytes().to_vec();
        for c in b.chunks(CHUNK) {
            top.extend_from_slice(&sha256(c));
        }
        sha256(&top)
    }

    /// Hashing several polynomials together gives each one's own digest
    /// (empty, shorter than a chunk, several chunks with a partial last one).
    #[test]
    fn digests_of_several_polynomials_are_their_own() {
        let words = |n: usize, seed: u64| -> Vec<Scalar> {
            (0..n as u64)
                .map(|i| [i ^ seed, i.wrapping_mul(0x9e37_79b9_7f4a_7c15), seed, !i])
                .collect()
        };
        let per_chunk = CHUNK / size_of::<Scalar>();
        let polys = [
            words(0, 1),
            words(3, 2),
            words(per_chunk, 3),
            words(2 * per_chunk + 5, 4),
            words(per_chunk / 2, 5),
        ];
        let refs: Vec<&[Scalar]> = polys.iter().map(|p| p.as_slice()).collect();
        let want: Vec<[u8; 32]> = refs.iter().map(|p| reference(p)).collect();
        assert_eq!(digest_many(&refs), want);
        assert_eq!(digest_many(&refs[3..4]), want[3..4]);
        assert!(digest_many(&[]).is_empty());
    }
}
