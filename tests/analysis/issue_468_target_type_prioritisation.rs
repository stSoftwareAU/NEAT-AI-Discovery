//! Tests for Issue #468: Target-type prioritisation — favour existing hidden
//! neurons as targets.
//!
//! Production discovery-cache analysis shows existing hidden neurons as targets have a 31.4%
//! success rate compared to 5.3–5.4% for output or discovery-hidden neurons.
//! These tests verify that:
//!
//! 1. The `EXISTING_HIDDEN_TARGET_BOOST` constant is within a valid range
//! 2. The `apply_target_type_boost` function boosts existing hidden targets
//! 3. Under deadline constraints, existing hidden targets are evaluated first
//! 4. Output neurons receive no target-type boost

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::constants::EXISTING_HIDDEN_TARGET_BOOST;
use neat_ai_discovery::analysis::synapse::apply_target_type_boost;
use std::collections::HashMap;

// =============================================================================
// EXISTING_HIDDEN_TARGET_BOOST constant validation
// =============================================================================

#[test]
fn existing_hidden_target_boost_value_is_reasonable() {
    // EXISTING_HIDDEN_TARGET_BOOST must be > 1.0 (boost) and <= 3.0 to avoid
    // over-biasing. 31.4% / 5.4% ≈ 5.8× raw ratio, but we use a conservative
    // multiplier to avoid runaway bias.
    let boosted = 1.0_f64 * EXISTING_HIDDEN_TARGET_BOOST;
    let unboosted = 1.0_f64;
    assert!(
        boosted > unboosted && boosted <= 3.0 * unboosted,
        "EXISTING_HIDDEN_TARGET_BOOST should provide a bounded boost: got {boosted}"
    );
}

#[test]
fn existing_hidden_target_boost_amplifies_score_gain() {
    let base_gain = 0.05_f64;
    let boosted_gain = base_gain * EXISTING_HIDDEN_TARGET_BOOST;

    assert!(
        boosted_gain > base_gain,
        "Boosted gain ({boosted_gain}) should exceed base gain ({base_gain})"
    );
    assert!(
        (boosted_gain - base_gain * EXISTING_HIDDEN_TARGET_BOOST).abs() < f64::EPSILON,
        "Boosted gain should equal base_gain × EXISTING_HIDDEN_TARGET_BOOST"
    );
}

// =============================================================================
// apply_target_type_boost function
// =============================================================================

#[test]
fn apply_target_type_boost_boosts_existing_hidden_target() {
    let gain = 0.10_f32;
    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("hidden-uuid-abc", "hidden");

    let boosted = apply_target_type_boost(gain, "hidden-uuid-abc", &neuron_type_map);

    assert!(
        boosted > gain,
        "Existing hidden target should get boosted score: {boosted} should be > {gain}"
    );
    let expected = gain * EXISTING_HIDDEN_TARGET_BOOST as f32;
    assert!(
        (boosted - expected).abs() < 1e-6,
        "Boosted gain should equal gain × EXISTING_HIDDEN_TARGET_BOOST: expected {expected}, got {boosted}"
    );
}

#[test]
fn apply_target_type_boost_neutral_for_output_target() {
    let gain = 0.10_f32;
    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("output-uuid-xyz", "output");

    let result = apply_target_type_boost(gain, "output-uuid-xyz", &neuron_type_map);

    assert!(
        (result - gain).abs() < 1e-6,
        "Output target should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn apply_target_type_boost_neutral_for_unknown_target() {
    let gain = 0.10_f32;
    let neuron_type_map: HashMap<&str, &str> = HashMap::new(); // Empty map

    let result = apply_target_type_boost(gain, "unknown-uuid", &neuron_type_map);

    assert!(
        (result - gain).abs() < 1e-6,
        "Unknown target should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn apply_target_type_boost_zero_gain_stays_zero() {
    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("hidden-uuid-abc", "hidden");

    let result = apply_target_type_boost(0.0, "hidden-uuid-abc", &neuron_type_map);

    assert!(
        result.abs() < 1e-6,
        "Zero gain should remain zero even with boost: got {result}"
    );
}

#[test]
fn apply_target_type_boost_neutral_for_input_target() {
    let gain = 0.10_f32;
    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("input-0", "input");

    let result = apply_target_type_boost(gain, "input-0", &neuron_type_map);

    assert!(
        (result - gain).abs() < 1e-6,
        "Input target should not be boosted: expected {gain}, got {result}"
    );
}

#[test]
fn apply_target_type_boost_neutral_for_constant_target() {
    let gain = 0.10_f32;
    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("const-uuid", "constant");

    let result = apply_target_type_boost(gain, "const-uuid", &neuron_type_map);

    assert!(
        (result - gain).abs() < 1e-6,
        "Constant target should not be boosted: expected {gain}, got {result}"
    );
}

// =============================================================================
// Focus order: existing hidden targets before output targets
// =============================================================================

#[test]
fn order_focus_targets_places_existing_hidden_before_output() {
    use neat_ai_discovery::analysis::utils::order_focus_targets;

    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("output-uuid-1", "output");
    neuron_type_map.insert("hidden-uuid-a", "hidden");
    neuron_type_map.insert("output-uuid-2", "output");
    neuron_type_map.insert("hidden-uuid-b", "hidden");
    neuron_type_map.insert("hidden-uuid-c", "hidden");

    let mut targets = vec![
        "output-uuid-1".to_string(),
        "hidden-uuid-a".to_string(),
        "output-uuid-2".to_string(),
        "hidden-uuid-b".to_string(),
        "hidden-uuid-c".to_string(),
    ];

    order_focus_targets(&mut targets, Some(42), &neuron_type_map);

    // All hidden neurons should appear before any output neuron
    let first_output_pos = targets
        .iter()
        .position(|uuid| {
            neuron_type_map
                .get(uuid.as_str())
                .is_some_and(|t| *t == "output")
        })
        .expect("Should have at least one output neuron");

    let last_hidden_pos = targets
        .iter()
        .rposition(|uuid| {
            neuron_type_map
                .get(uuid.as_str())
                .is_some_and(|t| *t == "hidden")
        })
        .expect("Should have at least one hidden neuron");

    assert!(
        last_hidden_pos < first_output_pos,
        "All hidden targets should come before output targets. \
        Last hidden at position {last_hidden_pos}, first output at position {first_output_pos}. \
        Order: {targets:?}"
    );
}

#[test]
fn order_focus_targets_consistent_across_seeds() {
    use neat_ai_discovery::analysis::utils::order_focus_targets;

    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("output-uuid-1", "output");
    neuron_type_map.insert("hidden-uuid-a", "hidden");
    neuron_type_map.insert("hidden-uuid-b", "hidden");

    for seed in [0u64, 1, 42, 100, 999] {
        let mut targets = vec![
            "output-uuid-1".to_string(),
            "hidden-uuid-a".to_string(),
            "hidden-uuid-b".to_string(),
        ];

        order_focus_targets(&mut targets, Some(seed), &neuron_type_map);

        // Hidden neurons should always come first
        let hidden_count = targets
            .iter()
            .filter(|uuid| {
                neuron_type_map
                    .get(uuid.as_str())
                    .is_some_and(|t| *t == "hidden")
            })
            .count();

        for (i, uuid) in targets.iter().enumerate() {
            let is_hidden = neuron_type_map
                .get(uuid.as_str())
                .is_some_and(|t| *t == "hidden");
            if i < hidden_count {
                assert!(
                    is_hidden,
                    "Seed {seed}: position {i} should be a hidden neuron, got {uuid}"
                );
            }
        }
    }
}

#[test]
fn order_focus_targets_preserves_all_targets() {
    use neat_ai_discovery::analysis::utils::order_focus_targets;

    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("output-uuid-1", "output");
    neuron_type_map.insert("hidden-uuid-a", "hidden");

    let mut targets = vec!["output-uuid-1".to_string(), "hidden-uuid-a".to_string()];

    order_focus_targets(&mut targets, Some(42), &neuron_type_map);

    assert_eq!(
        targets.len(),
        2,
        "All targets should be preserved after ordering"
    );

    let has_output = targets.iter().any(|u| u == "output-uuid-1");
    let has_hidden = targets.iter().any(|u| u == "hidden-uuid-a");
    assert!(has_output, "Output target should be preserved");
    assert!(has_hidden, "Hidden target should be preserved");
}

#[test]
fn order_focus_targets_handles_single_target() {
    use neat_ai_discovery::analysis::utils::order_focus_targets;

    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("hidden-uuid-a", "hidden");

    let mut targets = vec!["hidden-uuid-a".to_string()];
    order_focus_targets(&mut targets, Some(42), &neuron_type_map);

    assert_eq!(targets.len(), 1);
    assert_eq!(targets[0], "hidden-uuid-a");
}

#[test]
fn order_focus_targets_handles_unknown_types_as_non_hidden() {
    use neat_ai_discovery::analysis::utils::order_focus_targets;

    let mut neuron_type_map = HashMap::new();
    neuron_type_map.insert("hidden-uuid-a", "hidden");
    // "unknown-uuid" is not in the map

    let mut targets = vec!["unknown-uuid".to_string(), "hidden-uuid-a".to_string()];

    order_focus_targets(&mut targets, Some(42), &neuron_type_map);

    // Hidden should come before unknown
    assert_eq!(targets[0], "hidden-uuid-a", "Hidden should be first");
}
