//! Regression test for Issue #2346 (CWE-789: unbounded `n_outputs^2` correlation
//! matrix and uncancellable pair scan in `detect_correlated_error_patterns`).
//!
//! `detect_correlated_error_patterns` builds an `O(n_outputs^2)` correlation
//! matrix and an `O(n_inputs)` predictive-input scan per group with no bound on
//! `n_outputs` or `n_inputs`, and no deadline check inside the pairwise loop.
//! A pathological creature with a very large number of eligible output
//! neurons (or input neurons) could make the scan run unbounded work with no
//! way to cancel it. This test exercises the guarded entry point,
//! `detect_correlated_error_patterns_with_deadline`, confirming that:
//!
//! 1. An eligible-output count above `MAX_CORRELATED_ERROR_OUTPUTS` short-circuits
//!    before the matrix is built, reporting `OutputCeilingExceeded`.
//! 2. A deadline that has already elapsed stops the pairwise scan early,
//!    reporting `DeadlinePassed`.
//! 3. An input neuron count above `MAX_CORRELATED_ERROR_INPUTS` skips the
//!    predictive-input search (but still allows the output groups themselves
//!    to be produced), reporting `InputCeilingExceeded`.
//! 4. The legacy wrapper `detect_correlated_error_patterns` continues to return
//!    an empty `Vec` whenever the guarded scan would be skipped.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use std::time::SystemTime;

use neat_ai_discovery::analysis::detection::correlated_error::{
    CorrelatedErrorSkip, MAX_CORRELATED_ERROR_INPUTS, MAX_CORRELATED_ERROR_OUTPUTS,
    detect_correlated_error_patterns, detect_correlated_error_patterns_with_deadline,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a `DiscoverRecord` for a neuron with given activation and errors.
fn record(neuron_uuid: &str, obs_index: u32, activation: f32, errors: Vec<f32>) -> DiscoverRecord {
    DiscoverRecord {
        obs_index,
        neuron_uuid: neuron_uuid.to_string(),
        value: Some(activation),
        activation,
        errors,
    }
}

/// Helper: build a minimal creature with neurons and synapses.
fn make_creature(neurons: Vec<NeuronJson>, synapses: Vec<SynapseJson>) -> CreatureJson {
    let input_count = neurons.iter().filter(|n| n.neuron_type == "input").count();
    let output_count = neurons.iter().filter(|n| n.neuron_type == "output").count();
    CreatureJson {
        neurons,
        synapses,
        input: input_count,
        output: output_count,
    }
}

/// Helper: build a `NeuronJson`.
fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper: build a `SynapseJson`.
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Build a small, well-formed creature + records fixture with `n_outputs`
/// output neurons whose primary error is perfectly linear in `obs_index`
/// (so every pair is strongly correlated), driven by `n_inputs` input
/// neurons. Only the first input neuron (if any) is given records whose
/// activation tracks `obs_index`, making it the single predictive input.
/// Each output gets exactly `MIN_DISCOVERY_SAMPLE_COUNT` (20) records with a
/// one-element errors vec, to keep generated record counts small and the
/// test fast even for large `n_outputs`/`n_inputs`.
fn make_fixture(
    n_outputs: usize,
    n_inputs: usize,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    const SAMPLES: u32 = 20; // == MIN_DISCOVERY_SAMPLE_COUNT

    let mut neurons = Vec::with_capacity(n_outputs + n_inputs);
    let mut synapses = Vec::new();

    let input_uuids: Vec<String> = (0..n_inputs).map(|i| format!("input-{i}")).collect();
    for uuid in &input_uuids {
        neurons.push(neuron(uuid, "input", "IDENTITY"));
    }

    let output_uuids: Vec<String> = (0..n_outputs).map(|i| format!("output-{i}")).collect();
    for uuid in &output_uuids {
        neurons.push(neuron(uuid, "output", "IDENTITY"));
        if let Some(first_input) = input_uuids.first() {
            synapses.push(synapse(first_input, uuid, 0.5));
        }
    }

    let creature = make_creature(neurons, synapses);

    let mut neuron_records: Vec<(String, Vec<DiscoverRecord>)> = Vec::new();

    for uuid in &output_uuids {
        let records: Vec<DiscoverRecord> = (0..SAMPLES)
            .map(|i| record(uuid, i, 0.5, vec![i as f32 * 0.1]))
            .collect();
        neuron_records.push((uuid.clone(), records));
    }

    // Only the first input neuron is given records; its activation tracks
    // obs_index exactly like the error, so it correlates strongly (r == 1.0)
    // with the shared group error whenever the predictive-input search runs.
    if let Some(first_input) = input_uuids.first() {
        let records: Vec<DiscoverRecord> = (0..SAMPLES)
            .map(|i| record(first_input, i, i as f32, vec![]))
            .collect();
        neuron_records.push((first_input.clone(), records));
    }

    (creature, neuron_records)
}

/// Test 1: an eligible-output count one above the ceiling is rejected before
/// the O(n^2) matrix is built, and the legacy wrapper returns an empty `Vec`.
#[test]
fn rejects_output_count_above_ceiling_without_building_the_matrix() {
    let n_outputs = MAX_CORRELATED_ERROR_OUTPUTS + 1;
    let (creature, neuron_records) = make_fixture(n_outputs, 1);

    let scan = detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &None);

    assert!(
        scan.groups.is_empty(),
        "Groups should be empty when the output ceiling is exceeded"
    );
    assert_eq!(
        scan.skip,
        Some(CorrelatedErrorSkip::OutputCeilingExceeded {
            eligible: n_outputs,
            ceiling: MAX_CORRELATED_ERROR_OUTPUTS,
        }),
        "Skip reason should report the output ceiling breach"
    );

    let legacy_groups = detect_correlated_error_patterns(&creature, &neuron_records);
    assert!(
        legacy_groups.is_empty(),
        "Legacy wrapper should also return no groups when the output ceiling is exceeded"
    );
}

/// Test 2 (control): a small correlated output set, well under the ceiling,
/// is not skipped — proving the fixture would otherwise produce groups.
#[test]
fn small_correlated_output_set_is_not_skipped() {
    let (creature, neuron_records) = make_fixture(3, 1);

    let scan = detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &None);

    assert!(
        !scan.groups.is_empty(),
        "Small correlated output set should produce at least one group"
    );
    assert_eq!(
        scan.skip, None,
        "No skip reason should be reported for a small, well-formed fixture"
    );
}

/// Test 3: a deadline that has already elapsed stops the pairwise scan early.
#[test]
fn elapsed_deadline_stops_the_pair_scan() {
    let (creature, neuron_records) = make_fixture(3, 1);

    let deadline = Some(SystemTime::UNIX_EPOCH);
    let scan =
        detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &deadline);

    assert!(
        scan.groups.is_empty(),
        "Groups should be empty once the deadline has passed"
    );
    assert_eq!(
        scan.skip,
        Some(CorrelatedErrorSkip::DeadlinePassed),
        "Skip reason should report the elapsed deadline"
    );
}

/// Test 4: an input neuron count above the ceiling skips the predictive-input
/// search, but output groups are still produced with empty predictive inputs.
#[test]
fn input_count_above_ceiling_skips_predictive_search() {
    let n_inputs = MAX_CORRELATED_ERROR_INPUTS + 1;
    let (creature, neuron_records) = make_fixture(3, n_inputs);

    let scan = detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &None);

    assert!(
        !scan.groups.is_empty(),
        "Output groups should still be produced when only the input ceiling is exceeded"
    );
    for group in &scan.groups {
        assert!(
            group.predictive_input_uuids.is_empty(),
            "Predictive inputs should be empty when the input ceiling is exceeded, got: {:?}",
            group.predictive_input_uuids
        );
    }
    assert_eq!(
        scan.skip,
        Some(CorrelatedErrorSkip::InputCeilingExceeded {
            inputs: n_inputs,
            ceiling: MAX_CORRELATED_ERROR_INPUTS,
        }),
        "Skip reason should report the input ceiling breach"
    );
}

/// Companion to the previous test: with the same small output fixture but an
/// input count well below the ceiling, the predictive input is found — this
/// proves the fixture reaches the predictive-input search at all.
#[test]
fn predictive_input_found_below_input_ceiling() {
    let (creature, neuron_records) = make_fixture(3, 1);

    let scan = detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &None);

    assert!(
        !scan.groups.is_empty(),
        "Should detect at least one correlated error group"
    );
    assert_eq!(
        scan.skip, None,
        "No skip reason should be reported below both ceilings"
    );
    let group = &scan.groups[0];
    assert!(
        group
            .predictive_input_uuids
            .contains(&"input-0".to_string()),
        "input-0 should be identified as predictive, got: {:?}",
        group.predictive_input_uuids
    );
}

/// Test 6: with no input neurons at all, `find_predictive_inputs` never checks
/// the deadline (its loop iterates zero times), so only the per-matrix-row
/// deadline check can stop the scan. First confirm the fixture produces
/// groups with no deadline, then confirm an elapsed deadline reports
/// `DeadlinePassed` with no groups.
#[test]
fn elapsed_deadline_stops_matrix_build_with_no_input_neurons() {
    let (creature, neuron_records) = make_fixture(3, 0);

    let scan = detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &None);
    assert!(
        !scan.groups.is_empty(),
        "Fixture with no input neurons should still produce at least one group"
    );
    assert_eq!(
        scan.skip, None,
        "No skip reason should be reported for a small, well-formed fixture"
    );

    let deadline = Some(SystemTime::UNIX_EPOCH);
    let scan =
        detect_correlated_error_patterns_with_deadline(&creature, &neuron_records, &deadline);

    assert!(
        scan.groups.is_empty(),
        "Groups should be empty once the deadline has passed, even with no input neurons"
    );
    assert_eq!(
        scan.skip,
        Some(CorrelatedErrorSkip::DeadlinePassed),
        "Skip reason should report the elapsed deadline via the per-matrix-row check"
    );
}
