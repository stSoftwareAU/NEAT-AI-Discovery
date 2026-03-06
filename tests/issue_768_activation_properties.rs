//! Integration tests for shared activation function classification helpers (Issue #768).
//!
//! Verifies that the consolidated helpers in `activation_properties` correctly
//! classify all known squash function names.

use neat_ai_discovery::analysis::detection::activation_properties::{
    can_have_dead_zone, is_bounded_squash, is_saturating_squash,
};

// =============================================================================
// is_bounded_squash tests
// =============================================================================

#[test]
fn bounded_squash_recognises_all_bounded_functions() {
    let bounded = [
        "TANH",
        "LOGISTIC",
        "HARD_TANH",
        "CLIPPED",
        "BIPOLAR",
        "BIPOLAR_SIGMOID",
        "STEP",
        "SOFTSIGN",
        "ISRU",
        "ARCTAN",
        "RELU6",
    ];
    for name in &bounded {
        assert!(
            is_bounded_squash(name),
            "{name} should be classified as bounded"
        );
    }
}

#[test]
fn bounded_squash_rejects_unbounded_functions() {
    let unbounded = ["RELU", "LEAKYRELU", "ELU", "SELU", "IDENTITY", "SOFTPLUS"];
    for name in &unbounded {
        assert!(
            !is_bounded_squash(name),
            "{name} should NOT be classified as bounded"
        );
    }
}

// =============================================================================
// is_saturating_squash tests
// =============================================================================

#[test]
fn saturating_squash_recognises_saturating_functions() {
    let saturating = [
        "TANH",
        "LOGISTIC",
        "SIGMOID",
        "HARD_TANH",
        "CLIPPED",
        "SOFTSIGN",
    ];
    for name in &saturating {
        assert!(
            is_saturating_squash(name),
            "{name} should be classified as saturating"
        );
    }
}

#[test]
fn saturating_squash_is_case_insensitive() {
    assert!(is_saturating_squash("tanh"));
    assert!(is_saturating_squash("Tanh"));
    assert!(is_saturating_squash("logistic"));
    assert!(is_saturating_squash("hard_tanh"));
}

#[test]
fn saturating_squash_rejects_non_saturating() {
    assert!(!is_saturating_squash("RELU"));
    assert!(!is_saturating_squash("IDENTITY"));
    assert!(!is_saturating_squash("SOFTPLUS"));
    assert!(!is_saturating_squash("ELU"));
}

// =============================================================================
// can_have_dead_zone tests
// =============================================================================

#[test]
fn dead_zone_recognises_relu_family() {
    let relu_family = ["RELU", "LEAKYRELU", "ELU", "SELU"];
    for name in &relu_family {
        assert!(
            can_have_dead_zone(name),
            "{name} should be classified as having a dead zone"
        );
    }
}

#[test]
fn dead_zone_rejects_non_relu_functions() {
    assert!(!can_have_dead_zone("TANH"));
    assert!(!can_have_dead_zone("LOGISTIC"));
    assert!(!can_have_dead_zone("IDENTITY"));
    assert!(!can_have_dead_zone("STEP"));
}

// =============================================================================
// Cross-classification tests
// =============================================================================

#[test]
fn all_saturating_squash_functions_are_also_bounded() {
    let saturating = ["TANH", "LOGISTIC", "HARD_TANH", "CLIPPED", "SOFTSIGN"];
    for name in &saturating {
        assert!(
            is_bounded_squash(name),
            "{name} is saturating but not bounded — classification mismatch"
        );
    }
}

#[test]
fn dead_zone_functions_are_not_bounded() {
    let dead_zone = ["RELU", "LEAKYRELU", "ELU", "SELU"];
    for name in &dead_zone {
        assert!(
            !is_bounded_squash(name),
            "{name} has a dead zone but should not be bounded"
        );
    }
}
