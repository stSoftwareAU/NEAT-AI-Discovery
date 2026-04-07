//! Adaptive Gaussian proposal distribution for weight search (Issue #1019).
//!
//! Replaces the fixed 9-variant weight grid with an adaptive proposal
//! distribution centred on the computed optimal weight. The distribution's
//! standard deviation (sigma) adapts based on historical acceptance rates
//! per target neuron type.
//!
//! ## Design
//!
//! - Proposals are drawn from a deterministic Gaussian-like distribution
//!   using hash-based pseudo-random numbers (no external RNG dependency).
//! - sigma adapts toward a target acceptance rate: too many accepts -> decrease sigma
//!   (focus); too few -> increase sigma (explore).
//! - Falls back to the fixed 9-variant grid when insufficient historical data
//!   is available (< `ADAPTIVE_PROPOSAL_MIN_HISTORY` samples).
//! - A configurable fraction of proposals are sign-flipped to maintain
//!   exploration of negative weight regions.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation

use crate::analysis::constants::{
    ADAPTIVE_PROPOSAL_ADAPTATION_RATE, ADAPTIVE_PROPOSAL_CANDIDATE_COUNT,
    ADAPTIVE_PROPOSAL_INITIAL_SIGMA, ADAPTIVE_PROPOSAL_MAX_SIGMA, ADAPTIVE_PROPOSAL_MIN_HISTORY,
    ADAPTIVE_PROPOSAL_MIN_SIGMA, ADAPTIVE_PROPOSAL_SIGN_FLIP_PROBABILITY,
    ADAPTIVE_PROPOSAL_TARGET_ACCEPTANCE,
};
use std::collections::HashMap;

// =============================================================================
// Acceptance Rate Tracker
// =============================================================================

/// Per-target-type acceptance rate tracker for sigma adaptation.
///
/// Tracks how many proposed weights were accepted (produced positive
/// improvement) versus total proposals, keyed by a target type label
/// (e.g., "output", "hidden", "discovery-hidden").
#[derive(Debug, Clone)]
pub struct AcceptanceTracker {
    /// Key: target type label. Value: accepted count, total count, and current sigma.
    entries: HashMap<String, AcceptanceEntry>,
}

/// Per-target-type acceptance statistics and adapted sigma.
#[derive(Debug, Clone)]
struct AcceptanceEntry {
    accepted: u32,
    total: u32,
    sigma: f32,
}

impl Default for AcceptanceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AcceptanceTracker {
    /// Creates a new empty tracker.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Returns the current sigma for the given target type.
    ///
    /// Returns the initial sigma if the target type has insufficient history.
    pub fn sigma_for(&self, target_type: &str) -> f32 {
        match self.entries.get(target_type) {
            Some(entry) if entry.total as usize >= ADAPTIVE_PROPOSAL_MIN_HISTORY => entry.sigma,
            _ => ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
        }
    }

    /// Returns true if the given target type has enough history for adaptive proposals.
    pub fn has_sufficient_history(&self, target_type: &str) -> bool {
        self.entries
            .get(target_type)
            .is_some_and(|e| (e.total as usize) >= ADAPTIVE_PROPOSAL_MIN_HISTORY)
    }

    /// Records the outcome of a batch of weight proposals.
    ///
    /// `accepted_count` is the number of proposals that produced positive improvement.
    /// `total_count` is the total number of proposals evaluated.
    pub fn record_batch(&mut self, target_type: &str, accepted_count: u32, total_count: u32) {
        let entry = self
            .entries
            .entry(target_type.to_string())
            .or_insert_with(|| AcceptanceEntry {
                accepted: 0,
                total: 0,
                sigma: ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
            });

        entry.accepted += accepted_count;
        entry.total += total_count;

        // Adapt sigma once we have enough history
        if entry.total as usize >= ADAPTIVE_PROPOSAL_MIN_HISTORY {
            let acceptance_rate = entry.accepted as f32 / entry.total as f32;
            if acceptance_rate > ADAPTIVE_PROPOSAL_TARGET_ACCEPTANCE {
                // Too many acceptances -> decrease sigma to focus
                entry.sigma /= ADAPTIVE_PROPOSAL_ADAPTATION_RATE;
            } else {
                // Too few acceptances -> increase sigma to explore
                entry.sigma *= ADAPTIVE_PROPOSAL_ADAPTATION_RATE;
            }
            entry.sigma = entry
                .sigma
                .clamp(ADAPTIVE_PROPOSAL_MIN_SIGMA, ADAPTIVE_PROPOSAL_MAX_SIGMA);
        }
    }

    /// Returns the acceptance rate for the given target type.
    ///
    /// Returns `None` if no data is available.
    pub fn acceptance_rate(&self, target_type: &str) -> Option<f32> {
        self.entries.get(target_type).and_then(|e| {
            if e.total > 0 {
                Some(e.accepted as f32 / e.total as f32)
            } else {
                None
            }
        })
    }
}

// =============================================================================
// Proposal Generation
// =============================================================================

/// The fixed 9-variant grid used as fallback when insufficient history is available.
const FIXED_GRID_MULTIPLIERS: [f32; 9] = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, -0.5, -1.0];

/// Generates weight candidates using the adaptive proposal distribution.
///
/// When sufficient historical data is available (`tracker` has >=
/// `ADAPTIVE_PROPOSAL_MIN_HISTORY` for the target type), samples from a
/// deterministic Gaussian-like distribution centred on `weight_optimal`
/// with sigma adapted from the acceptance rate tracker.
///
/// When insufficient data is available, falls back to the fixed 9-variant
/// grid for compatibility.
///
/// # Arguments
///
/// * `weight_optimal` - The computed optimal weight to centre proposals around
/// * `source_uuid` - Source neuron UUID (used for deterministic hashing)
/// * `target_uuid` - Target neuron UUID (used for deterministic hashing)
/// * `tracker` - Acceptance rate tracker for sigma adaptation
/// * `target_type` - Target neuron type label (e.g., "output", "hidden")
///
/// # Returns
///
/// A vector of proposed weight candidates.
pub fn generate_weight_candidates(
    weight_optimal: f32,
    source_uuid: &str,
    target_uuid: &str,
    tracker: &AcceptanceTracker,
    target_type: &str,
) -> Vec<f32> {
    if !tracker.has_sufficient_history(target_type) {
        return generate_fixed_grid(weight_optimal);
    }

    let sigma = tracker.sigma_for(target_type);
    generate_gaussian_candidates(weight_optimal, sigma, source_uuid, target_uuid)
}

/// Generates weight candidates from the fixed 9-variant grid (fallback).
pub fn generate_fixed_grid(weight_optimal: f32) -> Vec<f32> {
    FIXED_GRID_MULTIPLIERS
        .iter()
        .map(|&m| weight_optimal * m)
        .collect()
}

/// Generates weight candidates from a deterministic Gaussian-like distribution.
///
/// Uses hash-based pseudo-random numbers derived from source and target UUIDs
/// to produce deterministic but varied proposals. The Box-Muller transform
/// converts uniform pseudo-random pairs into Gaussian samples.
fn generate_gaussian_candidates(
    weight_optimal: f32,
    sigma: f32,
    source_uuid: &str,
    target_uuid: &str,
) -> Vec<f32> {
    let count = ADAPTIVE_PROPOSAL_CANDIDATE_COUNT;
    let mut candidates = Vec::with_capacity(count);

    // Always include the optimal weight itself
    candidates.push(weight_optimal);

    let base_hash = deterministic_hash(source_uuid, target_uuid);

    for i in 1..count {
        let hash_i = mix_hash(base_hash, i as u64);

        // Determine if this candidate should be sign-flipped
        let flip_hash = mix_hash(hash_i, 0xDEAD_BEEF);
        let flip_uniform = (flip_hash as f32) / (u64::MAX as f32);
        let sign_flip = flip_uniform < ADAPTIVE_PROPOSAL_SIGN_FLIP_PROBABILITY;

        // Generate Gaussian sample using Box-Muller transform
        // Use two independent hash values as uniform inputs
        let u1_hash = mix_hash(hash_i, 0x1234_5678);
        let u2_hash = mix_hash(hash_i, 0x9ABC_DEF0);

        // Map to (0, 1) range, avoiding exact 0 for log
        let u1 = ((u1_hash as f64) / (u64::MAX as f64)).max(1e-15);
        let u2 = (u2_hash as f64) / (u64::MAX as f64);

        // Box-Muller: z = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
        let z = ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32;

        let proposed = if sign_flip {
            -weight_optimal.abs() + z * sigma * weight_optimal.abs().max(0.1)
        } else {
            weight_optimal + z * sigma * weight_optimal.abs().max(0.1)
        };

        candidates.push(proposed);
    }

    candidates
}

/// FNV-1a hash of source and target UUIDs.
fn deterministic_hash(source_uuid: &str, target_uuid: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in source_uuid.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x0100_0000_01b3);
    for byte in target_uuid.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

/// Mix a hash value with an additional u64 to produce varied outputs.
fn mix_hash(hash: u64, extra: u64) -> u64 {
    let mut h = hash;
    h ^= extra;
    h = h.wrapping_mul(0x0100_0000_01b3);
    h ^= h >> 32;
    h = h.wrapping_mul(0x0100_0000_01b3);
    h ^= h >> 16;
    h
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_tracker_initial_sigma() {
        let tracker = AcceptanceTracker::new();
        assert!(
            (tracker.sigma_for("output") - ADAPTIVE_PROPOSAL_INITIAL_SIGMA).abs() < f32::EPSILON,
            "Initial sigma should be the default"
        );
    }

    #[test]
    fn acceptance_tracker_insufficient_history_uses_default() {
        let mut tracker = AcceptanceTracker::new();
        // Record fewer than MIN_HISTORY outcomes
        tracker.record_batch("output", 3, 5);
        assert!(
            !tracker.has_sufficient_history("output"),
            "Should not have sufficient history with only 5 samples"
        );
        assert!(
            (tracker.sigma_for("output") - ADAPTIVE_PROPOSAL_INITIAL_SIGMA).abs() < f32::EPSILON,
            "Should return default sigma with insufficient history"
        );
    }

    #[test]
    fn acceptance_tracker_adapts_sigma_down_on_high_acceptance() {
        let mut tracker = AcceptanceTracker::new();
        // Record enough outcomes with high acceptance rate
        tracker.record_batch(
            "hidden",
            ADAPTIVE_PROPOSAL_MIN_HISTORY as u32,
            ADAPTIVE_PROPOSAL_MIN_HISTORY as u32,
        );
        let sigma = tracker.sigma_for("hidden");
        assert!(
            sigma < ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
            "Sigma should decrease when acceptance rate is 100% (above target): got {sigma}"
        );
    }

    #[test]
    fn acceptance_tracker_adapts_sigma_up_on_low_acceptance() {
        let mut tracker = AcceptanceTracker::new();
        // Record enough outcomes with zero acceptance
        tracker.record_batch("hidden", 0, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);
        let sigma = tracker.sigma_for("hidden");
        assert!(
            sigma > ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
            "Sigma should increase when acceptance rate is 0% (below target): got {sigma}"
        );
    }

    #[test]
    fn acceptance_tracker_sigma_clamped_to_bounds() {
        let mut tracker = AcceptanceTracker::new();
        // Drive sigma very low with many high-acceptance batches
        for _ in 0..100 {
            tracker.record_batch("output", 100, 100);
        }
        let sigma = tracker.sigma_for("output");
        assert!(
            sigma >= ADAPTIVE_PROPOSAL_MIN_SIGMA,
            "Sigma must not go below minimum: got {sigma}"
        );

        // Drive sigma very high with many zero-acceptance batches
        let mut tracker2 = AcceptanceTracker::new();
        for _ in 0..100 {
            tracker2.record_batch("output", 0, 100);
        }
        let sigma2 = tracker2.sigma_for("output");
        assert!(
            sigma2 <= ADAPTIVE_PROPOSAL_MAX_SIGMA,
            "Sigma must not exceed maximum: got {sigma2}"
        );
    }

    #[test]
    fn acceptance_tracker_acceptance_rate() {
        let mut tracker = AcceptanceTracker::new();
        assert!(
            tracker.acceptance_rate("output").is_none(),
            "Should return None with no data"
        );
        tracker.record_batch("output", 3, 10);
        let rate = tracker.acceptance_rate("output").unwrap();
        assert!(
            (rate - 0.3).abs() < f32::EPSILON,
            "Acceptance rate should be 0.3, got {rate}"
        );
    }

    #[test]
    fn fixed_grid_produces_expected_candidates() {
        let weight = 2.0;
        let candidates = generate_fixed_grid(weight);
        assert_eq!(
            candidates.len(),
            9,
            "Fixed grid should produce 9 candidates"
        );
        // Verify key points
        assert!((candidates[0] - 0.2).abs() < f32::EPSILON, "0.1 * 2.0");
        assert!((candidates[4] - 2.0).abs() < f32::EPSILON, "1.0 * 2.0");
        assert!((candidates[7] - (-1.0)).abs() < f32::EPSILON, "-0.5 * 2.0");
        assert!((candidates[8] - (-2.0)).abs() < f32::EPSILON, "-1.0 * 2.0");
    }

    #[test]
    fn gaussian_candidates_include_optimal_weight() {
        let weight = 1.5;
        let candidates = generate_gaussian_candidates(weight, 0.5, "source-1", "target-1");
        assert_eq!(
            candidates[0], weight,
            "First candidate should be the optimal weight"
        );
    }

    #[test]
    fn gaussian_candidates_correct_count() {
        let candidates = generate_gaussian_candidates(1.0, 0.5, "source-1", "target-1");
        assert_eq!(
            candidates.len(),
            ADAPTIVE_PROPOSAL_CANDIDATE_COUNT,
            "Should generate the configured number of candidates"
        );
    }

    #[test]
    fn gaussian_candidates_are_deterministic() {
        let c1 = generate_gaussian_candidates(1.0, 0.5, "source-a", "target-b");
        let c2 = generate_gaussian_candidates(1.0, 0.5, "source-a", "target-b");
        assert_eq!(c1, c2, "Same inputs should produce identical candidates");
    }

    #[test]
    fn gaussian_candidates_differ_for_different_inputs() {
        let c1 = generate_gaussian_candidates(1.0, 0.5, "source-a", "target-b");
        let c2 = generate_gaussian_candidates(1.0, 0.5, "source-c", "target-d");
        assert_ne!(
            c1, c2,
            "Different inputs should produce different candidates"
        );
    }

    #[test]
    fn gaussian_candidates_vary_around_optimal() {
        let weight = 2.0;
        let candidates = generate_gaussian_candidates(weight, 0.5, "source-1", "target-1");
        // Check that candidates are spread out (not all identical)
        let min = candidates.iter().copied().fold(f32::INFINITY, f32::min);
        let max = candidates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 0.1,
            "Candidates should be spread out: range [{min}, {max}]"
        );
    }

    #[test]
    fn gaussian_candidates_include_some_negative() {
        // With SIGN_FLIP_PROBABILITY = 0.15, in 100 different candidate sets
        // we should see at least some negative candidates
        let mut found_negative = false;
        for i in 0..100 {
            let source = format!("source-{i}");
            let target = format!("target-{i}");
            let candidates = generate_gaussian_candidates(2.0, 0.5, &source, &target);
            if candidates.iter().any(|&c| c < 0.0) {
                found_negative = true;
                break;
            }
        }
        assert!(
            found_negative,
            "Should find at least some negative candidates across 100 trials"
        );
    }

    #[test]
    fn generate_weight_candidates_uses_fixed_grid_without_history() {
        let tracker = AcceptanceTracker::new();
        let candidates = generate_weight_candidates(1.0, "src", "tgt", &tracker, "output");
        assert_eq!(
            candidates.len(),
            9,
            "Should use fixed 9-variant grid without history"
        );
    }

    #[test]
    fn generate_weight_candidates_uses_adaptive_with_history() {
        let mut tracker = AcceptanceTracker::new();
        tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);
        let candidates = generate_weight_candidates(1.0, "src", "tgt", &tracker, "output");
        assert_eq!(
            candidates.len(),
            ADAPTIVE_PROPOSAL_CANDIDATE_COUNT,
            "Should use adaptive proposal with sufficient history"
        );
    }

    #[test]
    fn all_gaussian_candidates_are_finite() {
        for i in 0..50 {
            let source = format!("s-{i}");
            let target = format!("t-{i}");
            let candidates = generate_gaussian_candidates(1.0, 0.5, &source, &target);
            for (j, &c) in candidates.iter().enumerate() {
                assert!(
                    c.is_finite(),
                    "Candidate {j} is not finite: {c} (source={source}, target={target})"
                );
            }
        }
    }

    #[test]
    fn gaussian_candidates_with_zero_weight() {
        // Edge case: weight_optimal = 0.0. sigma scaling uses max(|w|, 0.1)
        let candidates = generate_gaussian_candidates(0.0, 0.5, "src", "tgt");
        assert_eq!(candidates[0], 0.0, "First candidate should be 0.0");
        // Other candidates should be varied around 0.0
        assert!(
            candidates.iter().any(|&c| c != 0.0),
            "Non-first candidates should differ from 0.0"
        );
    }

    #[test]
    fn gaussian_candidates_with_negative_weight() {
        let weight = -1.5;
        let candidates = generate_gaussian_candidates(weight, 0.5, "src", "tgt");
        assert_eq!(
            candidates[0], weight,
            "First candidate should be the optimal weight"
        );
        // Should have varied proposals around -1.5
        let min = candidates.iter().copied().fold(f32::INFINITY, f32::min);
        let max = candidates.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max - min > 0.1,
            "Candidates should be spread around negative weight"
        );
    }
}
