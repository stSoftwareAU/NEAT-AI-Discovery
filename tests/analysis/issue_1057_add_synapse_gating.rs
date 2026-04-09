//! Integration tests for add-synapse candidate gating (Issue #1057).
//!
//! Tests that add-synapse candidates are skipped when:
//! 1. `ModuleOutcomeTracker` shows historical success rate below the configurable threshold
//! 2. The creature's synapse density (synapses/neurons ratio) exceeds the maximum threshold
//!
//! These gates reduce wasted compute on add-synapse candidates that have a near-zero
//! success rate (~0.1-0.3%) across most creatures.

use neat_ai_discovery::analysis::constants::{
    ADD_SYNAPSE_MAX_DENSITY_RATIO, ADD_SYNAPSE_MIN_SUCCESS_RATE, ADD_SYNAPSE_MODULE_NAME,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::synapse::add_synapse_gating::{
    compute_synapse_density_ratio, should_skip_add_synapse_candidates,
};

/// Helper to create a tracker with known module history.
fn tracker_with_history(entries: &[(&str, u32, u32)]) -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    for &(name, attempts, successes) in entries {
        for i in 0..attempts {
            tracker.record(name, i < successes);
        }
    }
    tracker
}

// =============================================================================
// Synapse density ratio computation
// =============================================================================

#[test]
fn test_density_ratio_basic() {
    // 100 synapses, 10 neurons → ratio 10.0
    let ratio = compute_synapse_density_ratio(100, 10);
    assert!((ratio - 10.0).abs() < f64::EPSILON);
}

#[test]
fn test_density_ratio_zero_neurons_returns_zero() {
    // Edge case: 0 neurons should not panic
    let ratio = compute_synapse_density_ratio(50, 0);
    assert!((ratio - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_density_ratio_no_synapses() {
    let ratio = compute_synapse_density_ratio(0, 10);
    assert!((ratio - 0.0).abs() < f64::EPSILON);
}

// =============================================================================
// ModuleOutcomeTracker gating
// =============================================================================

#[test]
fn test_skip_when_tracker_shows_low_success_rate() {
    // 100 attempts, 0 successes → success rate ≈ 0.01 (Bayesian posterior with Beta(1,1))
    // With 100 attempts and 0 successes: (0+1)/(100+2) = 0.0098 < 0.01
    let tracker = tracker_with_history(&[(ADD_SYNAPSE_MODULE_NAME, 100, 0)]);
    let result = should_skip_add_synapse_candidates(
        &Some(tracker),
        5,    // low density
        50,   // some synapses
        None, // default threshold
        None, // default density
    );
    assert!(
        result.skip,
        "Should skip add-synapse candidates when success rate < 1%"
    );
    assert!(
        result.reason.contains("success rate"),
        "Reason should mention success rate: {}",
        result.reason
    );
}

#[test]
fn test_no_skip_when_tracker_shows_acceptable_success_rate() {
    // 100 attempts, 5 successes → ~5.8% success rate > 1%
    let tracker = tracker_with_history(&[(ADD_SYNAPSE_MODULE_NAME, 100, 5)]);
    let result = should_skip_add_synapse_candidates(
        &Some(tracker),
        5, // low density
        50,
        None,
        None,
    );
    assert!(
        !result.skip,
        "Should not skip when success rate is above threshold"
    );
}

#[test]
fn test_no_skip_when_tracker_has_no_data() {
    // No data for add-synapse module → neutral prior (0.5) → do not skip
    let tracker = ModuleOutcomeTracker::new();
    let result = should_skip_add_synapse_candidates(&Some(tracker), 5, 50, None, None);
    assert!(
        !result.skip,
        "Should not skip when no historical data is available"
    );
}

#[test]
fn test_no_skip_when_tracker_has_insufficient_data() {
    // Only 5 attempts → insufficient data → do not skip
    let tracker = tracker_with_history(&[(ADD_SYNAPSE_MODULE_NAME, 5, 0)]);
    let result = should_skip_add_synapse_candidates(&Some(tracker), 5, 50, None, None);
    assert!(
        !result.skip,
        "Should not skip when insufficient data for reliable estimate"
    );
}

#[test]
fn test_no_skip_when_no_tracker() {
    let result = should_skip_add_synapse_candidates(&None, 5, 50, None, None);
    assert!(!result.skip, "Should not skip when no tracker provided");
}

// =============================================================================
// Synapse density gating
// =============================================================================

#[test]
fn test_skip_when_density_too_high() {
    // 200 synapses / 10 neurons = ratio 20 > default threshold (14)
    let result = should_skip_add_synapse_candidates(&None, 10, 200, None, None);
    assert!(
        result.skip,
        "Should skip when synapse density ratio exceeds threshold"
    );
    assert!(
        result.reason.contains("density"),
        "Reason should mention density: {}",
        result.reason
    );
}

#[test]
fn test_no_skip_when_density_below_threshold() {
    // 30 synapses / 10 neurons = ratio 3 < default threshold (14)
    let result = should_skip_add_synapse_candidates(&None, 10, 30, None, None);
    assert!(
        !result.skip,
        "Should not skip when density is below threshold"
    );
}

// =============================================================================
// Custom thresholds
// =============================================================================

#[test]
fn test_custom_success_rate_threshold() {
    // 100 attempts, 3 successes → ~3.9% success rate
    // With custom threshold of 5%, this should be skipped
    let tracker = tracker_with_history(&[(ADD_SYNAPSE_MODULE_NAME, 100, 3)]);
    let result = should_skip_add_synapse_candidates(&Some(tracker), 10, 30, Some(0.05), None);
    assert!(
        result.skip,
        "Should skip when success rate is below custom threshold of 5%"
    );
}

#[test]
fn test_custom_density_threshold() {
    // 60 synapses / 10 neurons = ratio 6
    // With custom density threshold of 5, this should be skipped
    let result = should_skip_add_synapse_candidates(&None, 10, 60, None, Some(5.0));
    assert!(
        result.skip,
        "Should skip when density exceeds custom threshold"
    );
}

// =============================================================================
// Constants validation
// =============================================================================

#[test]
fn test_default_constants_are_sensible() {
    // Verify constants are in expected ranges at runtime (not const-evaluable due to f64 ops)
    let min_rate = ADD_SYNAPSE_MIN_SUCCESS_RATE;
    let max_density = ADD_SYNAPSE_MAX_DENSITY_RATIO;
    assert!(
        min_rate > 0.0 && min_rate < 0.1,
        "Default success rate threshold should be between 0% and 10%, got {min_rate}"
    );
    assert!(
        max_density > 5.0 && max_density < 30.0,
        "Default density ratio threshold should be between 5 and 30, got {max_density}"
    );
    assert!(
        !ADD_SYNAPSE_MODULE_NAME.is_empty(),
        "Module name must not be empty"
    );
}
