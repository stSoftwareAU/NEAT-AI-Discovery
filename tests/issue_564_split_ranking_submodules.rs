//! Issue #564: Verify that splitting ranking.rs into sub-modules preserves all public API.
//!
//! These tests confirm that the focus::ranking module's public types, functions, and
//! constants remain accessible via `neat_ai_discovery::focus::*` after the split into
//! a `ranking/` directory with focused sub-modules.
//!
//! Each test exercises real functionality — not source code inspection.

use neat_ai_discovery::focus::{
    RankFocusStats, RankedNeuron, RemovalCandidate, SelectionStats, SynapseCounts,
    calculate_removal_savings, rank_focus_neurons, rank_focus_neurons_with_history,
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
// Public API accessibility after split
// =============================================================================

#[test]
fn issue_564_synapse_counts_accessible_after_split() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "TANH"),
            ("out", "output", "LOGISTIC"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let counts = SynapseCounts::new(&creature);
    let (incoming, outgoing) = counts.get("h1");
    assert_eq!(incoming, 1);
    assert_eq!(outgoing, 1);
}

#[test]
fn issue_564_calculate_removal_savings_accessible_after_split() {
    let savings = calculate_removal_savings(2, 3, 1e-7);
    let expected = 1e-7_f32 * 1.5;
    assert!(
        (savings - expected).abs() < 1e-15,
        "Removal savings formula incorrect: got {savings}, expected {expected}"
    );
}

#[test]
fn issue_564_rank_focus_neurons_accessible_after_split() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let mut records = Vec::new();
    records.extend(make_records("h1", 0.3, 2.0, 10));
    records.extend(make_records("out", 0.5, 1.0, 10));

    let tmp = write_temp_parquet(&records);
    let result: RankFocusStats =
        rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
            .expect("rank focus neurons");

    assert!(!result.neurons.is_empty(), "Should return ranked neurons");
    assert!(result.processed_neurons > 0, "Should process neurons");

    // Verify RankedNeuron fields are accessible
    let h1: &RankedNeuron = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "h1")
        .expect("h1 should be ranked");
    assert!(h1.total_error > 0.0);
    assert!(h1.raw_error > 0.0);
    assert!(h1.impact > 0.0);
    assert!(h1.mean_activation > 0.0);
}

#[test]
fn issue_564_rank_focus_neurons_with_history_accessible_after_split() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    let mut records = Vec::new();
    records.extend(make_records("h1", 0.3, 2.0, 10));
    records.extend(make_records("out", 0.5, 1.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons_with_history(
        tmp.path().to_str().unwrap(),
        &creature,
        None,
        None,
        None, // No history
    )
    .expect("rank focus neurons with history");

    assert!(!result.neurons.is_empty(), "Should return ranked neurons");
}

#[test]
fn issue_564_removal_candidate_fields_accessible_after_split() {
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![
            ("in-0", "h1", 0.001),
            ("h1", "out", 0.001), // Very weak connection
        ],
    );

    let mut records = Vec::new();
    records.extend(make_records("h1", 0.01, 0.001, 10));
    records.extend(make_records("out", 0.3, 3.0, 10));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, Some(0.01))
        .expect("rank focus neurons");

    // Verify RemovalCandidate fields are accessible
    for candidate in &result.removal_candidates {
        let _: &RemovalCandidate = candidate;
        let _uuid = &candidate.neuron_uuid;
        let _err = candidate.total_error;
        let _impact = candidate.impact;
        let _mean_act = candidate.mean_activation;
        let _awi = candidate.activation_weighted_impact;
        let _incoming = candidate.incoming_synapses;
        let _outgoing = candidate.outgoing_synapses;
        let _savings = candidate.removal_savings;
        let _reduction = candidate.expected_error_reduction;
        let _reason = &candidate.reason;
    }
}

#[test]
fn issue_564_selection_stats_type_accessible_after_split() {
    // SelectionStats is a HashMap type alias — verify it can be constructed
    let mut stats: SelectionStats = std::collections::HashMap::new();
    stats.insert(("from".to_string(), "to".to_string()), 0.75);
    assert_eq!(stats.len(), 1);
}

#[test]
fn issue_564_constant_neuron_removals_accessible_after_split() {
    // Creature with a constant neuron (will have near-zero variance if all activations are identical)
    let creature = make_creature(
        vec![
            ("in-0", "input", "IDENTITY"),
            ("h1", "hidden", "IDENTITY"),
            ("out", "output", "IDENTITY"),
        ],
        vec![("in-0", "h1", 1.0), ("h1", "out", 1.0)],
    );

    // All records have identical activation (constant neuron)
    let mut records = Vec::new();
    records.extend(make_records("h1", 0.01, 1.0, 20));
    records.extend(make_records("out", 0.3, 1.0, 20));

    let tmp = write_temp_parquet(&records);
    let result = rank_focus_neurons(tmp.path().to_str().unwrap(), &creature, None, None)
        .expect("rank focus neurons");

    // The constant_neuron_removals field should be accessible
    let _removals = &result.constant_neuron_removals;
}
