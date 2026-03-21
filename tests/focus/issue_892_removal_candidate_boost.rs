//! Tests for Issue #892: Boost remove-low-impact candidate generation and priority.
//!
//! GRQ-sampler discovery cache shows `remove-low-impact` has the highest success
//! rate at 21.5% (440/2,043). These tests verify:
//! 1. Removal candidates with low mean activation and low impact are accepted
//! 2. Candidates with high mean activation are filtered out
//! 3. Candidates with high structural impact are filtered out
//! 4. Accepted candidates receive a scoring boost (1.5×)
//! 5. Low-impact neuron detection uses widened ceiling (0.04)
//! 6. Low-impact neuron detection produces boosted estimated improvement

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::constants::REMOVAL_CANDIDATE_BOOST;
use neat_ai_discovery::analysis::detection::low_impact_neuron::{
    detect_low_impact_neurons, low_impact_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::focus::{calculate_removal_savings, rank_focus_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

use crate::common::{make_creature, neuron, output, record, synapse};

/// Build a creature from (uuid, type, squash) neurons and (from, to, weight) synapses.
fn make_creature_tuples(
    neurons: Vec<(&str, &str, &str)>,
    synapses: Vec<(&str, &str, f32)>,
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, t, _)| *t != "input")
            .map(|(uuid, ntype, squash)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: ntype.to_string(),
                squash: squash.to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, w)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight: w,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Create discovery records for a neuron with the given error and activation values.
fn make_records(uuid: &str, error: f32, activation: f32, count: u32) -> Vec<DiscoverRecord> {
    (0..count)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: Some(0.5),
            activation,
            errors: vec![error],
        })
        .collect()
}

/// Write records to a temporary parquet file.
fn write_temp_parquet(records: &[DiscoverRecord]) -> NamedTempFile {
    let tmp = NamedTempFile::new().expect("create temp file");
    write_records_to_parquet(tmp.path().to_str().unwrap(), records).expect("write parquet");
    tmp
}

// =============================================================================
// Constants validation
// =============================================================================

/// Test that the removal candidate boost matches expected value.
#[test]
fn removal_boost_matches_expected_value() {
    // Boost should reflect the 21.5% success rate (roughly 2× baseline)
    let boost = REMOVAL_CANDIDATE_BOOST;
    assert!(
        (boost - 1.5).abs() < f32::EPSILON,
        "Boost should be 1.5, got {boost}"
    );
}

// =============================================================================
// Removal candidate filtering — mean activation
// =============================================================================

/// Test: Neuron with near-zero activation is accepted as removal candidate.
#[test]
fn low_activation_neuron_is_removal_candidate() {
    // h1 has near-zero activation and small weight to output → removal candidate.
    // h2 provides a separate strong path to output so h1's structural impact is low.
    let creature = make_creature_tuples(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("h2", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![
            ("in-0", "h1", 0.001),
            ("in-0", "h2", 1.0),
            ("h1", "out", 0.001), // Very weak connection
            ("h2", "out", 1.0),   // Strong connection — dominates output
        ],
    );

    let mut records = Vec::new();
    // h1: near-zero activation (0.001) — within threshold
    records.extend(make_records("h1", 0.01, 0.001, 10));
    records.extend(make_records("h2", 0.5, 5.0, 10));
    records.extend(make_records("out", 0.3, 3.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, Some(0.01))
        .expect("rank focus neurons");

    let h1_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "h1");

    assert!(
        h1_removal.is_some(),
        "Neuron with near-zero activation should be a removal candidate. \
         Removal candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );
}

/// Test: Neuron with high mean activation is filtered out (Issue #892).
#[test]
fn high_activation_neuron_filtered_from_removal() {
    // Even if savings > activation_weighted_impact, high mean_activation
    // combined with non-zero structural impact should disqualify the candidate.
    // h2 provides a strong path so h1 has low (but non-zero) structural impact.
    let creature = make_creature_tuples(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("h2", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![
            ("in-0", "h1", 0.0001),
            ("in-0", "h2", 1.0),
            ("h1", "out", 0.0001), // Tiny weight but non-zero path to output
            ("h2", "out", 1.0),    // Strong path — dominates
        ],
    );

    let mut records = Vec::new();
    // h1: high activation (5.0) — above threshold
    records.extend(make_records("h1", 0.01, 5.0, 10));
    records.extend(make_records("h2", 0.5, 5.0, 10));
    records.extend(make_records("out", 0.3, 3.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, Some(0.01))
        .expect("rank focus neurons");

    let h1_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "h1");

    assert!(
        h1_removal.is_none(),
        "Neuron with high mean activation (5.0 > 0.04) and non-zero impact should be filtered out"
    );
}

// =============================================================================
// Removal candidate boost
// =============================================================================

/// Test: Removal candidates receive a scoring boost (Issue #892).
#[test]
fn removal_candidates_receive_scoring_boost() {
    // h1 has near-zero activation and small weight to output.
    // h2 provides a strong alternative path so h1 has low impact.
    let creature = make_creature_tuples(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("h2", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![
            ("in-0", "h1", 0.001),
            ("in-0", "h2", 1.0),
            ("h1", "out", 0.001),
            ("h2", "out", 1.0),
        ],
    );

    let mut records = Vec::new();
    records.extend(make_records("h1", 0.01, 0.001, 10));
    records.extend(make_records("h2", 0.5, 5.0, 10));
    records.extend(make_records("out", 0.3, 3.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, Some(0.01))
        .expect("rank focus neurons");

    let h1 = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "h1")
        .expect("h1 should be a removal candidate");

    // The removal_savings should include the boost
    let raw_savings = calculate_removal_savings(h1.incoming_synapses, h1.outgoing_synapses, 0.01);
    let expected_boosted = raw_savings * REMOVAL_CANDIDATE_BOOST;

    assert!(
        (h1.removal_savings - expected_boosted).abs() < 1e-10,
        "Removal savings ({}) should be boosted ({} × {} = {})",
        h1.removal_savings,
        raw_savings,
        REMOVAL_CANDIDATE_BOOST,
        expected_boosted
    );
}

// =============================================================================
// Low-impact neuron detection — widened ceiling (Issue #892)
// =============================================================================

/// Test: Neuron with mean abs activation of 0.01 is detected (was outside old ceiling of 1e-3).
#[test]
fn widened_ceiling_detects_activation_at_001() {
    let creature = make_creature(
        vec![
            neuron("hidden-mid", "hidden", "RELU"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("hidden-mid", "output-1", 0.5)],
    );

    // Mean abs activation ~0.01: above old ceiling (1e-3) but below new ceiling (0.04)
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-mid", i, 0.01, Some(0.01)))
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-mid".to_string(), records)], None);

    assert_eq!(
        candidates.len(),
        1,
        "Neuron with activation 0.01 should be detected with widened ceiling"
    );
    assert_eq!(candidates[0].neuron_uuid, "hidden-mid");
}

/// Test: Neuron with mean abs activation of 0.03 is detected (near widened ceiling).
#[test]
fn widened_ceiling_detects_activation_at_003() {
    let creature = make_creature(
        vec![
            neuron("hidden-nearceiling", "hidden", "RELU"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("hidden-nearceiling", "output-1", 0.5)],
    );

    // Mean abs activation ~0.03: well below new ceiling of 0.04
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-nearceiling", i, 0.03, Some(0.01)))
        .collect();

    let candidates = detect_low_impact_neurons(
        &creature,
        &[("hidden-nearceiling".to_string(), records)],
        None,
    );

    assert_eq!(
        candidates.len(),
        1,
        "Neuron with activation 0.03 should be detected with widened ceiling"
    );
}

/// Test: Neuron with mean abs activation of 0.05 is NOT detected (above ceiling).
#[test]
fn activation_above_widened_ceiling_not_detected() {
    let creature = make_creature(
        vec![
            neuron("hidden-high", "hidden", "RELU"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("hidden-high", "output-1", 0.5)],
    );

    // Mean abs activation ~0.05: above the new ceiling of 0.04
    let records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("hidden-high", i, 0.05, Some(0.01)))
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-high".to_string(), records)], None);

    assert!(
        candidates.is_empty(),
        "Neuron with activation 0.05 (above ceiling 0.04) should not be detected"
    );
}

// =============================================================================
// Low-impact neuron detection — boosted estimated improvement
// =============================================================================

/// Test: Low-impact candidates have boosted estimated improvement (Issue #892).
#[test]
fn low_impact_candidates_have_boosted_improvement() {
    let creature = make_creature(
        vec![
            neuron("hidden-low", "hidden", "RELU"),
            output("output-1", "IDENTITY"),
        ],
        vec![synapse("hidden-low", "output-1", 0.5)],
    );

    let records: Vec<DiscoverRecord> = (0..500)
        .map(|i| record("hidden-low", i, 1e-5, Some(0.01)))
        .collect();

    let candidates =
        detect_low_impact_neurons(&creature, &[("hidden-low".to_string(), records)], None);

    assert_eq!(candidates.len(), 1);
    let c = &candidates[0];

    // Base improvement is 0.003 (boosted from 0.002 in Issue #892)
    // With high confidence (many samples, low activation), estimated_improvement
    // should be close to 0.003 × confidence
    assert!(
        c.estimated_improvement > 0.001,
        "Estimated improvement ({}) should be meaningful (> 0.001) with boosted base",
        c.estimated_improvement
    );

    // Convert to coordinated candidates and verify expected gain
    let coordinated = low_impact_neurons_to_coordinated_candidates(std::slice::from_ref(c));
    assert_eq!(coordinated.len(), 1);
    assert!(
        coordinated[0].expected_creature_score_gain > 0.001,
        "Coordinated candidate gain ({}) should reflect boosted improvement",
        coordinated[0].expected_creature_score_gain
    );
}

// =============================================================================
// Integration: more candidates generated with wider detection
// =============================================================================

/// Test: Widened detection generates more candidates from a mixed network.
#[test]
fn widened_detection_generates_more_candidates() {
    let creature = make_creature(
        vec![
            neuron("dead", "hidden", "RELU"),
            neuron("low-old", "hidden", "TANH"), // Would have been detected before (1e-4)
            neuron("low-new", "hidden", "TANH"), // Only detected with widened ceiling (0.02)
            neuron("active", "hidden", "RELU"),
            output("output-1", "IDENTITY"),
        ],
        vec![
            synapse("dead", "output-1", 0.5),
            synapse("low-old", "output-1", 0.3),
            synapse("low-new", "output-1", 0.3),
            synapse("active", "output-1", 0.8),
        ],
    );

    // Dead neuron: activation 0.0 (below 1e-6) — NOT detected by low-impact module
    let dead_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("dead", i, 0.0, Some(-3.0)))
        .collect();

    // Low-old: activation 1e-4 — detected by both old and new ceiling
    let low_old_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("low-old", i, 1e-4, Some(1e-5)))
        .collect();

    // Low-new: activation 0.02 — only detected with widened ceiling (was above old 1e-3)
    let low_new_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| record("low-new", i, 0.02, Some(1e-5)))
        .collect();

    // Active: activation ~0.5 — NOT detected
    let active_records: Vec<DiscoverRecord> = (0..100)
        .map(|i| {
            let activation = 0.5 + 0.2 * ((i as f32 * 0.1).sin());
            record("active", i, activation, Some(1.0))
        })
        .collect();

    let candidates = detect_low_impact_neurons(
        &creature,
        &[
            ("dead".to_string(), dead_records),
            ("low-old".to_string(), low_old_records),
            ("low-new".to_string(), low_new_records),
            ("active".to_string(), active_records),
        ],
        None,
    );

    assert_eq!(
        candidates.len(),
        2,
        "Should detect 2 low-impact neurons (low-old and low-new)"
    );

    let uuids: Vec<&str> = candidates.iter().map(|c| c.neuron_uuid.as_str()).collect();
    assert!(uuids.contains(&"low-old"), "Should detect low-old");
    assert!(
        uuids.contains(&"low-new"),
        "Should detect low-new (widened ceiling)"
    );
}
