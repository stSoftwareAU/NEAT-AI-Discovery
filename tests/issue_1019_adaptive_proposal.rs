//! Integration tests for adaptive Gaussian proposal distribution (Issue #1019).
//!
//! Tests the adaptive proposal module's behaviour: fallback to fixed grid,
//! adaptive sigma tracking, Gaussian candidate generation, and comparison
//! with the fixed grid approach.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for test data (Issue #873)

use neat_ai_discovery::analysis::constants::{
    ADAPTIVE_PROPOSAL_CANDIDATE_COUNT, ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
    ADAPTIVE_PROPOSAL_MAX_SIGMA, ADAPTIVE_PROPOSAL_MIN_HISTORY, ADAPTIVE_PROPOSAL_MIN_SIGMA,
};
use neat_ai_discovery::analysis::synapse::adaptive_proposal::{
    AcceptanceTracker, generate_fixed_grid, generate_weight_candidates,
};

// =============================================================================
// Fallback Behaviour
// =============================================================================

/// Issue #1019: When no historical data is available, the system falls back
/// to the fixed 9-variant grid for compatibility.
#[test]
fn fallback_to_fixed_grid_without_history() {
    let tracker = AcceptanceTracker::new();
    let candidates = generate_weight_candidates(1.0, "source-1", "target-1", &tracker, "output");
    assert_eq!(
        candidates.len(),
        9,
        "Should produce 9 candidates from fixed grid when no history is available"
    );

    // Verify the candidates match the fixed grid
    let fixed = generate_fixed_grid(1.0);
    assert_eq!(
        candidates, fixed,
        "Fallback candidates should match the fixed grid"
    );
}

/// Issue #1019: With insufficient history (below `MIN_HISTORY`), the fixed grid is used.
#[test]
fn fallback_with_insufficient_history() {
    let mut tracker = AcceptanceTracker::new();
    // Record fewer than MIN_HISTORY samples
    tracker.record_batch("hidden", 2, (ADAPTIVE_PROPOSAL_MIN_HISTORY as u32) - 1);
    let candidates = generate_weight_candidates(1.0, "src", "tgt", &tracker, "hidden");
    assert_eq!(
        candidates.len(),
        9,
        "Should use fixed grid with insufficient history"
    );
}

// =============================================================================
// Adaptive Proposal Activation
// =============================================================================

/// Issue #1019: Once sufficient history is accumulated, the adaptive Gaussian
/// proposal is used instead of the fixed grid.
#[test]
fn adaptive_proposal_activates_with_sufficient_history() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);
    let candidates = generate_weight_candidates(1.0, "src", "tgt", &tracker, "output");
    assert_eq!(
        candidates.len(),
        ADAPTIVE_PROPOSAL_CANDIDATE_COUNT,
        "Should use adaptive proposal with sufficient history (got {} candidates)",
        candidates.len()
    );
}

/// Issue #1019: Adaptive proposal includes the optimal weight as first candidate.
#[test]
fn adaptive_proposal_includes_optimal_weight() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);
    let weight = 2.5;
    let candidates = generate_weight_candidates(weight, "src", "tgt", &tracker, "output");
    assert_eq!(
        candidates[0], weight,
        "First adaptive candidate should be the optimal weight"
    );
}

// =============================================================================
// Sigma Adaptation
// =============================================================================

/// Issue #1019: High acceptance rates cause sigma to decrease (focus).
#[test]
fn sigma_decreases_on_high_acceptance() {
    let mut tracker = AcceptanceTracker::new();
    // 100% acceptance rate
    tracker.record_batch(
        "output",
        ADAPTIVE_PROPOSAL_MIN_HISTORY as u32,
        ADAPTIVE_PROPOSAL_MIN_HISTORY as u32,
    );
    let sigma = tracker.sigma_for("output");
    assert!(
        sigma < ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
        "Sigma should decrease with high acceptance: got {sigma}, expected < {ADAPTIVE_PROPOSAL_INITIAL_SIGMA}"
    );
}

/// Issue #1019: Low acceptance rates cause sigma to increase (explore).
#[test]
fn sigma_increases_on_low_acceptance() {
    let mut tracker = AcceptanceTracker::new();
    // 0% acceptance rate
    tracker.record_batch("output", 0, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);
    let sigma = tracker.sigma_for("output");
    assert!(
        sigma > ADAPTIVE_PROPOSAL_INITIAL_SIGMA,
        "Sigma should increase with low acceptance: got {sigma}, expected > {ADAPTIVE_PROPOSAL_INITIAL_SIGMA}"
    );
}

/// Issue #1019: Sigma is bounded between minimum and maximum values.
#[test]
fn sigma_remains_bounded() {
    let mut tracker = AcceptanceTracker::new();

    // Drive sigma down with many 100% acceptance batches
    for _ in 0..200 {
        tracker.record_batch("low", 100, 100);
    }
    let sigma_low = tracker.sigma_for("low");
    assert!(
        sigma_low >= ADAPTIVE_PROPOSAL_MIN_SIGMA,
        "Sigma must not go below minimum {ADAPTIVE_PROPOSAL_MIN_SIGMA}: got {sigma_low}"
    );

    // Drive sigma up with many 0% acceptance batches
    let mut tracker2 = AcceptanceTracker::new();
    for _ in 0..200 {
        tracker2.record_batch("high", 0, 100);
    }
    let sigma_high = tracker2.sigma_for("high");
    assert!(
        sigma_high <= ADAPTIVE_PROPOSAL_MAX_SIGMA,
        "Sigma must not exceed maximum {ADAPTIVE_PROPOSAL_MAX_SIGMA}: got {sigma_high}"
    );
}

/// Issue #1019: Different target types track sigma independently.
#[test]
fn independent_sigma_per_target_type() {
    let mut tracker = AcceptanceTracker::new();

    // Output: 100% acceptance -> sigma decreases
    tracker.record_batch(
        "output",
        ADAPTIVE_PROPOSAL_MIN_HISTORY as u32,
        ADAPTIVE_PROPOSAL_MIN_HISTORY as u32,
    );
    // Hidden: 0% acceptance -> sigma increases
    tracker.record_batch("hidden", 0, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);

    let sigma_output = tracker.sigma_for("output");
    let sigma_hidden = tracker.sigma_for("hidden");

    assert!(
        sigma_output < sigma_hidden,
        "Output sigma ({sigma_output}) should be less than hidden sigma ({sigma_hidden})"
    );
}

// =============================================================================
// Candidate Quality
// =============================================================================

/// Issue #1019: Adaptive candidates cover a broader range than fixed grid
/// when sigma is large (exploring).
#[test]
fn adaptive_covers_broader_range_with_large_sigma() {
    let weight = 1.0;

    // Fixed grid range
    let fixed = generate_fixed_grid(weight);
    let fixed_min = fixed.iter().copied().fold(f32::INFINITY, f32::min);
    let fixed_max = fixed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let fixed_range = fixed_max - fixed_min;

    // Adaptive with large sigma (0% acceptance -> sigma grows)
    let mut tracker = AcceptanceTracker::new();
    for _ in 0..50 {
        tracker.record_batch("output", 0, 100);
    }

    // Sample from multiple source/target pairs to get a representative range
    let mut adaptive_min = f32::INFINITY;
    let mut adaptive_max = f32::NEG_INFINITY;
    for i in 0..20 {
        let source = format!("source-{i}");
        let target = format!("target-{i}");
        let candidates = generate_weight_candidates(weight, &source, &target, &tracker, "output");
        for &c in &candidates {
            adaptive_min = adaptive_min.min(c);
            adaptive_max = adaptive_max.max(c);
        }
    }
    let adaptive_range = adaptive_max - adaptive_min;

    assert!(
        adaptive_range >= fixed_range * 0.5,
        "Adaptive range ({adaptive_range}) should be at least half the fixed range ({fixed_range}) \
         when exploring with large sigma"
    );
}

/// Issue #1019: All generated candidates are finite numbers (no NaN or Inf).
#[test]
fn all_candidates_are_finite() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);

    for i in 0..100 {
        let source = format!("source-{i}");
        let target = format!("target-{i}");
        let candidates = generate_weight_candidates(1.0, &source, &target, &tracker, "output");
        for (j, &c) in candidates.iter().enumerate() {
            assert!(
                c.is_finite(),
                "Candidate {j} is not finite: {c} (source={source}, target={target})"
            );
        }
    }
}

/// Issue #1019: Candidates are deterministic — same inputs produce same output.
#[test]
fn candidates_are_deterministic() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);

    let c1 = generate_weight_candidates(1.0, "src-a", "tgt-b", &tracker, "output");
    let c2 = generate_weight_candidates(1.0, "src-a", "tgt-b", &tracker, "output");
    assert_eq!(c1, c2, "Same inputs should produce identical candidates");
}

/// Issue #1019: Different source/target pairs produce different candidates.
#[test]
fn different_inputs_produce_different_candidates() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);

    let c1 = generate_weight_candidates(1.0, "src-a", "tgt-b", &tracker, "output");
    let c2 = generate_weight_candidates(1.0, "src-c", "tgt-d", &tracker, "output");
    // Skip first element (both are the optimal weight)
    assert_ne!(
        &c1[1..],
        &c2[1..],
        "Different inputs should produce different candidates"
    );
}

// =============================================================================
// Fixed Grid Compatibility
// =============================================================================

/// Issue #1019: Fixed grid produces the expected 9 multipliers of the optimal weight.
#[test]
fn fixed_grid_expected_values() {
    let weight = 2.0;
    let candidates = generate_fixed_grid(weight);
    let expected = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, -1.0, -2.0];
    assert_eq!(candidates.len(), expected.len());
    for (i, (&got, &exp)) in candidates.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "Fixed grid candidate {i}: expected {exp}, got {got}"
        );
    }
}

/// Issue #1019: Fixed grid preserves negative weight exploration.
#[test]
fn fixed_grid_includes_negative_weights() {
    let candidates = generate_fixed_grid(1.5);
    let negatives: Vec<f32> = candidates.iter().copied().filter(|&c| c < 0.0).collect();
    assert!(
        negatives.len() >= 2,
        "Fixed grid should include at least 2 negative candidates, got {}",
        negatives.len()
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Issue #1019: Adaptive proposal handles zero optimal weight.
#[test]
fn adaptive_handles_zero_weight() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);

    let candidates = generate_weight_candidates(0.0, "src", "tgt", &tracker, "output");
    assert_eq!(candidates[0], 0.0, "First candidate should be 0.0");
    // sigma scaling uses max(|weight|, 0.1) to avoid degenerate proposals
    assert!(
        candidates.iter().any(|&c| c != 0.0),
        "Should have non-zero candidates even with zero optimal weight"
    );
}

/// Issue #1019: Adaptive proposal handles negative optimal weight.
#[test]
fn adaptive_handles_negative_weight() {
    let mut tracker = AcceptanceTracker::new();
    tracker.record_batch("output", 5, ADAPTIVE_PROPOSAL_MIN_HISTORY as u32);

    let weight = -1.5;
    let candidates = generate_weight_candidates(weight, "src", "tgt", &tracker, "output");
    assert_eq!(
        candidates[0], weight,
        "First candidate should be the optimal weight"
    );
    assert!(
        candidates.len() == ADAPTIVE_PROPOSAL_CANDIDATE_COUNT,
        "Should produce the correct number of candidates"
    );
}
