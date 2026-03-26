//! Candidate compression for synapse candidates (Issues #921, #922).
//!
//! When multiple independently-discovered `CandidateSynapseJson` candidates target
//! the same output neuron, this module compresses them into a single
//! `CoordinatedStructuralCandidateJson`:
//!
//! ```text
//! Before (N separate candidates):
//!   input-A → output    (gain = 0.01)
//!   input-B → output    (gain = 0.02)
//!
//! After (1 compressed candidate):
//!   input-A --+
//!             +-→ [hidden neuron, bias=0] → output   (combined gain)
//!   input-B --+
//! ```
//!
//! ## Module structure
//!
//! - `grouping` — Candidate grouping and deduplication by target neuron.
//! - `identity` — IDENTITY neuron compression (Issue #921): uses a hidden
//!   IDENTITY neuron that sums all inputs — mathematically equivalent to
//!   applying the individual synapses separately.
//! - `nonlinear` — Non-linear compression (Issue #922): uses TANH or GELU
//!   hidden neurons to capture interaction effects between inputs.
//! - `gain_estimation` — Saturation-aware gain estimation for non-linear
//!   squash functions.

mod gain_estimation;
mod grouping;
mod identity;
mod nonlinear;

pub use grouping::detect_compressible_groups;
pub use identity::compress_identity_candidates;
pub use nonlinear::compress_nonlinear_candidates;

use crate::CandidateSynapseJson;

/// A group of synapse candidates that share the same target neuron and can be
/// compressed into a single coordinated structural candidate.
#[derive(Debug)]
pub struct CompressibleGroup {
    pub to_neuron_uuid: String,
    pub candidates: Vec<CandidateSynapseJson>,
}

/// Generate a deterministic UUID for a compressed hidden neuron using FNV-1a hash.
///
/// The UUID is derived from the sorted input UUIDs and target UUID, following
/// the same pattern as `fan_in.rs`.
pub fn generate_compression_uuid(input_uuids: &[String], target_uuid: &str) -> String {
    let mut sorted_inputs: Vec<&str> = input_uuids.iter().map(String::as_str).collect();
    sorted_inputs.sort();

    // FNV-1a 64-bit hash.
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let fnv_prime: u64 = 0x0100_0000_01b3;

    for byte in b"compress:" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }
    for input in &sorted_inputs {
        for byte in input.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(fnv_prime);
        }
        // Separator between inputs.
        hash ^= u64::from(b'+');
        hash = hash.wrapping_mul(fnv_prime);
    }
    for byte in b"->" {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }
    for byte in target_uuid.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(fnv_prime);
    }

    format!("compress-{hash:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_uuid() {
        let inputs = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid1 = generate_compression_uuid(&inputs, "output-1");
        let uuid2 = generate_compression_uuid(&inputs, "output-1");
        assert_eq!(uuid1, uuid2, "UUIDs should be deterministic");
    }

    #[test]
    fn test_uuid_order_independent() {
        let inputs1 = vec!["input-b".to_string(), "input-a".to_string()];
        let inputs2 = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid1 = generate_compression_uuid(&inputs1, "output-1");
        let uuid2 = generate_compression_uuid(&inputs2, "output-1");
        assert_eq!(
            uuid1, uuid2,
            "UUIDs should be the same regardless of input order"
        );
    }

    #[test]
    fn test_uuid_differs_for_different_targets() {
        let inputs = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid1 = generate_compression_uuid(&inputs, "output-1");
        let uuid2 = generate_compression_uuid(&inputs, "output-2");
        assert_ne!(
            uuid1, uuid2,
            "Different targets should produce different UUIDs"
        );
    }

    #[test]
    fn test_uuid_starts_with_compress_prefix() {
        let inputs = vec!["input-a".to_string(), "input-b".to_string()];
        let uuid = generate_compression_uuid(&inputs, "output-1");
        assert!(
            uuid.starts_with("compress-"),
            "UUID should start with 'compress-' prefix"
        );
    }
}
