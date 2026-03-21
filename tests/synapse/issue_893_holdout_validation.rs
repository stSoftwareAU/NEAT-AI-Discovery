//! Issue #893: Hold-out validation for synapse multi-weight search
//!
//! Tests for:
//! 1. Hold-out validation constants are in valid ranges
//! 2. Constants maintain correct relationships with other thresholds
//! 3. The validation fraction produces meaningful split sizes

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for test arithmetic (Issue #873)

use neat_ai_discovery::analysis::constants::{
    HOLDOUT_MIN_SAMPLE_COUNT, HOLDOUT_VALIDATION_FRACTION, MIN_DISCOVERY_SAMPLE_COUNT,
    MIN_NEURON_SAMPLE_COUNT,
};

// =============================================================================
// Hold-out constant validation (Issue #893)
// =============================================================================

/// Issue #893: Hold-out minimum sample count must be >= `MIN_DISCOVERY_SAMPLE_COUNT`.
#[test]
fn holdout_min_sample_count_at_least_min_discovery() {
    const {
        assert!(HOLDOUT_MIN_SAMPLE_COUNT >= MIN_DISCOVERY_SAMPLE_COUNT);
    }
}

/// Issue #893: Hold-out validation fraction must be in (0.1, 0.5).
#[test]
fn holdout_validation_fraction_in_valid_range() {
    const {
        assert!(HOLDOUT_VALIDATION_FRACTION > 0.1);
        assert!(HOLDOUT_VALIDATION_FRACTION < 0.5);
    }
}

/// Issue #893: Hold-out minimum sample count must be >= 20.
#[test]
fn holdout_min_sample_count_reasonable() {
    const {
        assert!(HOLDOUT_MIN_SAMPLE_COUNT >= 20);
    }
}

/// Issue #893: The training partition from the split must have at least
/// `MIN_NEURON_SAMPLE_COUNT` samples to ensure reliable weight fitting.
#[test]
fn holdout_training_partition_has_enough_samples() {
    // At the minimum sample count, the training partition should have
    // at least MIN_NEURON_SAMPLE_COUNT samples.
    let train_count = HOLDOUT_MIN_SAMPLE_COUNT as f32 * (1.0 - HOLDOUT_VALIDATION_FRACTION);
    assert!(
        train_count >= MIN_NEURON_SAMPLE_COUNT as f32,
        "Issue #893: Training partition at minimum split size ({train_count:.0}) \
         must be >= MIN_NEURON_SAMPLE_COUNT ({MIN_NEURON_SAMPLE_COUNT})"
    );
}

/// Issue #893: The validation partition from the split must have at least 3 samples
/// to produce a meaningful improvement estimate.
#[test]
fn holdout_validation_partition_has_enough_samples() {
    let validate_count =
        (HOLDOUT_MIN_SAMPLE_COUNT as f32 * HOLDOUT_VALIDATION_FRACTION).round() as usize;
    assert!(
        validate_count >= 3,
        "Issue #893: Validation partition at minimum split size ({validate_count}) \
         must be >= 3"
    );
}

/// Issue #893: 70/30 split means roughly 70% training and 30% validation.
#[test]
fn holdout_split_produces_expected_partition_sizes() {
    let total = 100;
    let validate = (total as f32 * HOLDOUT_VALIDATION_FRACTION).round() as usize;
    let train = total - validate;

    assert_eq!(
        validate, 30,
        "Issue #893: 30% of 100 should be 30 validation samples"
    );
    assert_eq!(
        train, 70,
        "Issue #893: 70% of 100 should be 70 training samples"
    );
}

/// Issue #893: For sample counts just at the threshold, both partitions are non-empty.
#[test]
fn holdout_split_at_threshold_produces_non_empty_partitions() {
    let total = HOLDOUT_MIN_SAMPLE_COUNT;
    let validate = (total as f32 * HOLDOUT_VALIDATION_FRACTION)
        .round()
        .max(1.0) as usize;
    let train = total - validate;

    assert!(
        validate >= 1,
        "Issue #893: Validation partition must be non-empty"
    );
    assert!(
        train >= 1,
        "Issue #893: Training partition must be non-empty"
    );
}
