//! Issue #523: Targeted tests for the focus `ranking` sub-module.
//!
//! Tests neuron ranking with known impact scores and verifies:
//! - Ranking order respects error × impact weighting
//! - Removal candidate selection based on activation-weighted impact vs savings
//! - SynapseCounts correctness for various topologies
//! - calculate_removal_savings formula
//! - Ranking with known parquet data

mod common;

use neat_ai_discovery::focus::{
    SynapseCounts, calculate_removal_savings, compute_impacts_public, rank_focus_neurons,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Build a creature from (uuid, type, squash) neurons and (from, to, weight) synapses.
fn make_creature(
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
// SynapseCounts
// =============================================================================

#[test]
fn synapse_counts_correct_for_simple_chain() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let counts = SynapseCounts::new(&creature);

    let (h1_in, h1_out) = counts.get("h1");
    assert_eq!(h1_in, 1, "h1 should have 1 incoming");
    assert_eq!(h1_out, 1, "h1 should have 1 outgoing");

    let (out_in, out_out) = counts.get("out");
    assert_eq!(out_in, 1, "out should have 1 incoming");
    assert_eq!(out_out, 0, "out should have 0 outgoing");
}

#[test]
fn synapse_counts_for_hub_neuron() {
    // h1 receives from 3 inputs and sends to 2 outputs
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("in-1", "input", "IDENTITY"),
            ("in-2", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("out-0", "output", "LOGISTIC"),
            ("out-1", "output", "LOGISTIC"),
        ],
        vec![
            ("in-0", "h1", 1.0),
            ("in-1", "h1", 1.0),
            ("in-2", "h1", 1.0),
            ("h1", "out-0", 1.0),
            ("h1", "out-1", 1.0),
        ],
    );

    let counts = SynapseCounts::new(&creature);
    let (incoming, outgoing) = counts.get("h1");

    assert_eq!(incoming, 3, "Hub should have 3 incoming");
    assert_eq!(outgoing, 2, "Hub should have 2 outgoing");
}

#[test]
fn synapse_counts_nonexistent_neuron_returns_zero() {
    let creature = make_creature(
        vec![("in-0", "input", "IDENTITY"), ("out", "output", "LOGISTIC")],
        vec![("in-0", "out", 1.0)],
    );

    let counts = SynapseCounts::new(&creature);
    let (incoming, outgoing) = counts.get("does-not-exist");

    assert_eq!(incoming, 0);
    assert_eq!(outgoing, 0);
}

// =============================================================================
// calculate_removal_savings
// =============================================================================

#[test]
fn removal_savings_matches_neat_ai_formula() {
    // Formula: growth_cost × (1 + (incoming + outgoing) / 10)
    let growth_cost = 1e-7;

    // 0 synapses: savings = 1e-7 × 1.0 = 1e-7
    let s0 = calculate_removal_savings(0, 0, growth_cost);
    assert!((s0 - 1e-7).abs() < 1e-15, "0 synapses: {s0}");

    // 5 incoming + 5 outgoing = 10: savings = 1e-7 × 2.0 = 2e-7
    let s10 = calculate_removal_savings(5, 5, growth_cost);
    assert!((s10 - 2e-7).abs() < 1e-15, "10 synapses: {s10}");

    // 1 incoming + 0 outgoing = 1: savings = 1e-7 × 1.1 = 1.1e-7
    let s1 = calculate_removal_savings(1, 0, growth_cost);
    let expected = growth_cost * 1.1;
    assert!((s1 - expected).abs() < 1e-15, "1 synapse: {s1}");
}

#[test]
fn removal_savings_scales_with_growth_cost() {
    let s_small = calculate_removal_savings(2, 3, 1e-7);
    let s_large = calculate_removal_savings(2, 3, 1e-5);

    // Same synapse counts, but 100x larger growth cost → 100x larger savings
    assert!(
        (s_large / s_small - 100.0).abs() < 0.01,
        "Savings should scale linearly with growth_cost"
    );
}

// =============================================================================
// Impact calculation
// =============================================================================

#[test]
fn output_neuron_has_impact_one() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let impacts = compute_impacts_public(&creature);

    let out_impact = impacts.get("out").copied().unwrap_or(0.0);
    assert!(
        (out_impact - 1.0).abs() < f32::EPSILON,
        "Output impact should be 1.0, got {out_impact}"
    );
}

#[test]
fn hidden_neuron_impact_bounded_zero_to_one() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("h2", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![
            ("in-0", "h1", 3.0),
            ("in-0", "h2", 2.0),
            ("h1", "out", 0.7),
            ("h2", "out", 0.3),
        ],
    );

    let impacts = compute_impacts_public(&creature);

    for uuid in ["h1", "h2"] {
        let impact = impacts.get(uuid).copied().unwrap_or(-1.0);
        assert!(
            impact > 0.0 && impact <= 1.0,
            "{uuid} impact should be in (0, 1], got {impact}"
        );
    }
}

#[test]
fn disconnected_neuron_has_zero_impact() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("orphan", "hidden", "TANH"), // No connections
            ("out", "output", "LOGISTIC"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let impacts = compute_impacts_public(&creature);

    let orphan_impact = impacts.get("orphan").copied().unwrap_or(-1.0);
    assert!(
        orphan_impact.abs() < f32::EPSILON,
        "Disconnected neuron should have zero impact, got {orphan_impact}"
    );
}

// =============================================================================
// rank_focus_neurons — ranking order
// =============================================================================

#[test]
fn high_error_high_impact_neuron_ranked_first() {
    // h1: high error, directly connected to output → high impact
    // h2: low error, directly connected to output → high impact
    // Both connect to the same output so impact is similar.
    // h1 should rank higher because of higher error.
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("h2", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![
            ("in-0", "h1", 0.5),
            ("in-0", "h2", 0.5),
            ("h1", "out", 0.5),
            ("h2", "out", 0.5),
        ],
    );

    let mut records = Vec::new();
    records.extend(make_records("h1", 0.9, 0.5, 10)); // High error
    records.extend(make_records("h2", 0.1, 0.5, 10)); // Low error
    records.extend(make_records("out", 0.5, 0.5, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    assert!(result.neurons.len() >= 2, "Should rank at least 2 neurons");

    // h1 (high error) should appear before h2 (low error)
    let h1_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "h1")
        .expect("h1 ranked");
    let h2_pos = result
        .neurons
        .iter()
        .position(|n| n.neuron_uuid == "h2")
        .expect("h2 ranked");

    assert!(
        h1_pos < h2_pos,
        "High-error h1 (pos {h1_pos}) should rank before low-error h2 (pos {h2_pos})"
    );
}

// =============================================================================
// rank_focus_neurons — removal candidates
// =============================================================================

#[test]
fn low_impact_neuron_identified_as_removal_candidate() {
    // h1 has very low weight to output → low structural impact
    // h2 has higher weight → higher structural impact
    // With low activation, h1's activation_weighted_impact should be below removal savings.
    let creature = make_creature(
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
            ("h2", "out", 1.0),
        ],
    );

    let mut records = Vec::new();
    // h1: near-zero activation → very low activation_weighted_impact
    records.extend(make_records("h1", 0.01, 0.001, 10));
    records.extend(make_records("h2", 0.5, 5.0, 10));
    records.extend(make_records("out", 0.3, 3.0, 10));

    let tmp = write_temp_parquet(&records);
    // Use a larger cost_of_growth so removal_savings exceeds h1's tiny impact
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, Some(0.01))
        .expect("rank focus neurons");

    // h1 should be a removal candidate (low impact × low activation < savings)
    let h1_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "h1");

    assert!(
        h1_removal.is_some(),
        "h1 with near-zero impact should be a removal candidate. Candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );

    let candidate = h1_removal.unwrap();
    assert!(
        candidate.removal_savings > candidate.activation_weighted_impact,
        "Removal savings ({}) should exceed impact ({})",
        candidate.removal_savings,
        candidate.activation_weighted_impact
    );
}

#[test]
fn high_impact_neuron_not_removal_candidate() {
    // h1 is the sole path from input to output with strong weight
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let mut records = Vec::new();
    records.extend(make_records("h1", 0.5, 5.0, 10)); // High activation
    records.extend(make_records("out", 0.3, 3.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    let h1_removal = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "h1");

    assert!(
        h1_removal.is_none(),
        "High-impact h1 should NOT be a removal candidate"
    );
}

// =============================================================================
// rank_focus_neurons — max_results
// =============================================================================

#[test]
fn max_results_limits_output_size() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("h2", "hidden", "IDENTITY"),
            ("h3", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![
            ("in-0", "h1", 1.0),
            ("in-0", "h2", 1.0),
            ("in-0", "h3", 1.0),
            ("h1", "out", 1.0),
            ("h2", "out", 1.0),
            ("h3", "out", 1.0),
        ],
    );

    let mut records = Vec::new();
    for uuid in ["h1", "h2", "h3", "out"] {
        records.extend(make_records(uuid, 0.5, 0.5, 5));
    }

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, Some(2), None)
        .expect("rank focus neurons");

    assert!(
        result.neurons.len() <= 2,
        "max_results=2 should limit to 2 neurons, got {}",
        result.neurons.len()
    );
}

// =============================================================================
// rank_focus_neurons — basic statistics
// =============================================================================

#[test]
fn ranked_neurons_have_correct_error_and_impact() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let mut records = Vec::new();
    // h1: error = 0.3, activation = 2.0
    records.extend(make_records("h1", 0.3, 2.0, 10));
    // Output error must be >= h1 error to avoid clamping h1's total_error
    records.extend(make_records("out", 0.5, 1.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    let h1 = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "h1")
        .expect("h1 should be ranked");

    // raw_error should be approximately 0.3 (total_error may be clamped to max_output_error)
    assert!(
        (h1.raw_error - 0.3).abs() < 0.05,
        "h1 raw_error should be ~0.3, got {}",
        h1.raw_error
    );

    // Impact should be positive (sole path to output)
    assert!(
        h1.impact > 0.0,
        "h1 impact should be positive, got {}",
        h1.impact
    );

    // Mean activation should be approximately 2.0
    assert!(
        (h1.mean_activation - 2.0).abs() < 0.1,
        "h1 mean_activation should be ~2.0, got {}",
        h1.mean_activation
    );
}
