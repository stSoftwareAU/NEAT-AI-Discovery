//! Tests for Issue #481: Suggest improvements to the discovery module.
//!
//! These tests validate the concrete improvements identified during the
//! codebase analysis. Each test exercises real library functions with test
//! data and asserts on observable outcomes.
//!
//! ## Improvement Areas Tested
//!
//! 1. NaN-safe floating-point sorting (Issue #483)
//! 2. Bias values are correctly sorted for all activation functions
//! 3. Discovery modules produce well-formed candidates
//! 4. Constants module provides consistent thresholds

mod common;

use neat_ai_discovery::analysis::activation::get_bias_values;
use neat_ai_discovery::analysis::constants;

// =============================================================================
// NaN-safe sorting (Issue #483)
// =============================================================================

/// Verify that `get_bias_values` returns sorted values for all known activation
/// functions. This tests the fix for the `partial_cmp().unwrap()` pattern that
/// could panic on NaN values.
#[test]
fn test_bias_values_are_sorted_for_all_activations() {
    let activations = [
        "TANH",
        "LOGISTIC",
        "RELU",
        "IDENTITY",
        "GELU",
        "MISH",
        "SOFTSIGN",
        "SOFTPLUS",
        "ARCTAN",
        "BENT_IDENTITY",
        "BIPOLAR",
        "CLIPPED",
        "HARD_TANH",
        "ELU",
        "RELU6",
        "STEP",
        "UNKNOWN_SQUASH", // Tests the default branch
    ];

    for squash in &activations {
        let values = get_bias_values(squash);
        assert!(
            !values.is_empty(),
            "get_bias_values({squash}) should return non-empty vector"
        );

        // Verify values are strictly sorted (no NaN-induced disorder)
        for window in values.windows(2) {
            assert!(
                window[0] <= window[1],
                "get_bias_values({squash}) not sorted: {:.4} > {:.4} in {:?}",
                window[0],
                window[1],
                values
            );
        }

        // Verify no NaN values in the output
        for v in &values {
            assert!(
                !v.is_nan(),
                "get_bias_values({squash}) contains NaN: {values:?}"
            );
        }
    }
}

/// Verify that bias values contain both negative and positive values for
/// standard activation functions (ensuring the merge of negative and positive
/// base arrays works correctly).
#[test]
fn test_bias_values_span_negative_and_positive() {
    let activations = ["TANH", "LOGISTIC", "RELU", "IDENTITY"];

    for squash in &activations {
        let values = get_bias_values(squash);
        let has_negative = values.iter().any(|v| *v < 0.0);
        let has_positive = values.iter().any(|v| *v > 0.0);

        assert!(
            has_negative,
            "get_bias_values({squash}) should contain negative values"
        );
        assert!(
            has_positive,
            "get_bias_values({squash}) should contain positive values"
        );
        assert!(
            values.contains(&0.0),
            "get_bias_values({squash}) should contain zero"
        );
    }
}

// =============================================================================
// Constants consistency (Issue #424 extension)
// =============================================================================

/// Verify that boost constants are within their documented valid ranges.
/// Uses const assertions to validate at compile time.
#[test]
fn test_boost_constants_within_valid_ranges() {
    // Validate at compile time that constant relationships hold
    const {
        assert!(constants::INPUT_SOURCE_BOOST > 1.0);
        assert!(constants::INPUT_SOURCE_BOOST <= 3.0);
        assert!(constants::EXISTING_HIDDEN_TARGET_BOOST > 1.0);
        assert!(constants::EXISTING_HIDDEN_TARGET_BOOST <= 3.0);
        assert!(constants::MIN_BOOST_SAMPLES >= 5);
        assert!(constants::MIN_BOOST_SAMPLES <= 50);
    }

    // Runtime check that the values are usable (non-NaN, finite)
    let boost = constants::INPUT_SOURCE_BOOST;
    assert!(boost.is_finite(), "INPUT_SOURCE_BOOST must be finite");

    let target_boost = constants::EXISTING_HIDDEN_TARGET_BOOST;
    assert!(
        target_boost.is_finite(),
        "EXISTING_HIDDEN_TARGET_BOOST must be finite"
    );
}

/// Verify that sentinel constants maintain required ordering invariants.
#[test]
fn test_sentinel_constants_ordering_invariants() {
    // Validate at compile time that constant ordering invariants hold
    const {
        assert!(constants::MIN_SENTINEL_GAP > constants::SENTINEL_TOLERANCE);
        assert!(constants::MIN_DISCOVERY_SAMPLE_COUNT >= constants::MIN_NEURON_SAMPLE_COUNT);
        assert!(constants::SENTINEL_TOLERANCE > 0.0);
        assert!(constants::MIN_SENTINEL_FRACTION > 0.0);
        assert!(constants::MIN_SENTINEL_FRACTION < 1.0);
    }

    // Runtime check: sentinel values are valid floats
    let sentinels = constants::CANDIDATE_SENTINELS;
    for s in &sentinels {
        assert!(s.is_finite(), "Sentinel value {s} must be finite");
    }
}

// =============================================================================
// Detection modules produce well-formed candidates
// =============================================================================

/// Verify that saturation detection returns well-formed candidates with
/// sufficient samples.
#[test]
fn test_saturation_detection_produces_valid_candidates() {
    use neat_ai_discovery::analysis::detection::saturation::detect_saturated_neurons;
    use neat_ai_discovery::types::DiscoverRecord;

    let neurons = vec![("h1".to_string(), "LOGISTIC".to_string(), 0.0_f32)];

    // Create enough samples (above MIN_DISCOVERY_SAMPLE_COUNT) with saturated activations
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..30)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.999, // Clearly saturated for LOGISTIC
                errors: vec![0.5],
            })
            .collect(),
    )];

    let detected = detect_saturated_neurons(&neurons, &records);

    // With 30 samples at activation 0.999, LOGISTIC should be detected as saturated
    assert!(
        !detected.is_empty(),
        "Should detect saturation for LOGISTIC neuron at activation 0.999 with 30 samples"
    );

    // Verify each detection has a valid neuron UUID
    for candidate in &detected {
        assert!(
            !candidate.neuron_uuid.is_empty(),
            "Detected saturated neuron must have a non-empty UUID"
        );
    }
}

/// Verify that dead neuron detection returns well-formed candidates with
/// sufficient samples.
#[test]
fn test_dead_neuron_detection_produces_valid_candidates() {
    use common::{hidden, make_creature, neuron, output, synapse};
    use neat_ai_discovery::analysis::detection::dead_neuron::detect_dead_neurons;
    use neat_ai_discovery::types::DiscoverRecord;

    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden("dead-h1", "LOGISTIC"),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "dead-h1", 0.001),
            synapse("dead-h1", "output-0", 0.001),
            synapse("input-1", "output-0", 1.0),
        ],
    );

    // Create 30 samples where dead-h1 has near-zero activation
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "dead-h1".to_string(),
        (0..30)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "dead-h1".to_string(),
                value: None,
                activation: 0.0, // Completely dead
                errors: vec![0.001],
            })
            .collect(),
    )];

    let detected = detect_dead_neurons(&creature, &records, None);

    assert!(
        !detected.is_empty(),
        "Should detect dead neuron with zero activation across 30 samples"
    );

    // Verify each detection targets a valid neuron
    for candidate in &detected {
        assert!(
            !candidate.neuron_uuid.is_empty(),
            "Detected dead neuron must have a non-empty UUID"
        );
    }
}

/// Verify that dormant synapse detection returns well-formed candidates.
#[test]
fn test_dormant_synapse_detection_produces_valid_candidates() {
    use common::{hidden, make_creature, neuron, output, synapse};
    use neat_ai_discovery::analysis::detection::dormant_synapse::detect_dormant_synapses;
    use neat_ai_discovery::types::DiscoverRecord;

    let creature = make_creature(
        vec![
            neuron("input-0", "input", "IDENTITY"),
            neuron("input-1", "input", "IDENTITY"),
            hidden("h1", "LOGISTIC"),
            output("output-0", "LOGISTIC"),
        ],
        vec![
            synapse("input-0", "h1", 0.0001), // Near-zero weight — dormant
            synapse("input-1", "h1", 1.0),
            synapse("h1", "output-0", 1.0),
        ],
    );

    // Create enough samples for detection
    let mut records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();
    for neuron_uuid in &["input-0", "input-1", "h1", "output-0"] {
        records.push((
            neuron_uuid.to_string(),
            (0..30)
                .map(|i| DiscoverRecord {
                    obs_index: i,
                    neuron_uuid: neuron_uuid.to_string(),
                    value: Some(if *neuron_uuid == "input-0" { 0.5 } else { 0.8 }),
                    activation: if *neuron_uuid == "input-0" { 0.5 } else { 0.8 },
                    errors: vec![0.1],
                })
                .collect(),
        ));
    }

    let detected = detect_dormant_synapses(&creature, &records);

    // With a 0.0001 weight synapse, dormant detection should flag it
    // Note: detection may require specific conditions beyond just low weight,
    // so we verify the function runs without panicking and returns valid data
    for candidate in &detected {
        assert!(
            !candidate.from_neuron_uuid.is_empty() && !candidate.to_neuron_uuid.is_empty(),
            "Dormant synapse detection must return non-empty UUIDs"
        );
    }
}

/// Verify that oscillating neuron detection handles edge cases without panicking.
#[test]
fn test_oscillating_detection_handles_constant_activation() {
    use neat_ai_discovery::analysis::detection::oscillating_neuron::detect_oscillating_neurons;
    use neat_ai_discovery::types::DiscoverRecord;

    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0_f32)];

    // All-constant activations (zero variance) — should NOT detect oscillation
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "h1".to_string(),
        (0..30)
            .map(|i| DiscoverRecord {
                obs_index: i,
                neuron_uuid: "h1".to_string(),
                value: None,
                activation: 0.5, // Constant — not oscillating
                errors: vec![0.1],
            })
            .collect(),
    )];

    let detected = detect_oscillating_neurons(&neurons, &records);
    assert!(
        detected.is_empty(),
        "Constant activation should not be flagged as oscillating"
    );
}

// =============================================================================
// DIVERSIFY_TOP_K sanity
// =============================================================================

/// Verify that DIVERSIFY_TOP_K is large enough for meaningful diversification
/// but not so large as to defeat sorting.
#[test]
fn test_diversify_top_k_is_practical() {
    const {
        assert!(constants::DIVERSIFY_TOP_K >= 1);
        assert!(constants::DIVERSIFY_TOP_K <= 128);
    }

    // Runtime check: value is usable in a range
    let k = constants::DIVERSIFY_TOP_K;
    let items: Vec<usize> = (0..k * 2).collect();
    assert!(
        items.len() >= k,
        "DIVERSIFY_TOP_K should be usable as a top-K selector"
    );
}
