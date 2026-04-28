//! Integration tests for Issue #1003: Parallelise identity and nonlinear
//! candidate compression via `rayon::join`.
//!
//! Both `compress_identity_candidates` and `compress_nonlinear_candidates`
//! operate on immutable shared references and must produce identical results
//! whether invoked sequentially or concurrently.

use neat_ai_discovery::analysis::candidate_compression::{
    compress_identity_candidates, compress_nonlinear_candidates,
};
use neat_ai_discovery::{CandidateSynapseJson, CreatureJson, NeuronJson, SynapseJson};

/// Helper: create a `CandidateSynapseJson`.
fn candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: from.to_string(),
        to_neuron_uuid: to.to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 80,
        total_count: 100,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
        comment: None,
        variant_key: None,
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

/// Helper: build a minimal creature.
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

/// Build a set of candidates and a creature suitable for both identity and
/// nonlinear compression tests.
fn build_test_data() -> (Vec<CandidateSynapseJson>, CreatureJson) {
    let candidates = vec![
        candidate("input-a", "output-1", 0.3, 0.05),
        candidate("input-b", "output-1", 0.5, 0.06),
        candidate("input-c", "output-1", 0.4, 0.04),
    ];
    let creature = make_creature(
        vec![
            neuron("input-a", "input", "IDENTITY"),
            neuron("input-b", "input", "IDENTITY"),
            neuron("input-c", "input", "IDENTITY"),
            neuron("output-1", "output", "TANH"),
        ],
        vec![
            synapse("input-a", "output-1", 0.3),
            synapse("input-b", "output-1", 0.5),
            synapse("input-c", "output-1", 0.4),
        ],
    );
    (candidates, creature)
}

/// Test that `rayon::join` produces the same results as sequential calls.
#[test]
fn test_parallel_compression_matches_sequential() {
    let (candidates, creature) = build_test_data();

    // Sequential execution.
    let seq_identity = compress_identity_candidates(&candidates, &creature);
    let seq_nonlinear = compress_nonlinear_candidates(&candidates, &creature);

    // Parallel execution via rayon::join.
    let (par_identity, par_nonlinear) = rayon::join(
        || compress_identity_candidates(&candidates, &creature),
        || compress_nonlinear_candidates(&candidates, &creature),
    );

    assert_eq!(
        seq_identity.len(),
        par_identity.len(),
        "Identity compressed count should match between sequential and parallel"
    );
    assert_eq!(
        seq_nonlinear.len(),
        par_nonlinear.len(),
        "Nonlinear compressed count should match between sequential and parallel"
    );

    // Verify each identity candidate matches.
    for (seq, par) in seq_identity.iter().zip(par_identity.iter()) {
        assert_eq!(
            seq.operations.len(),
            par.operations.len(),
            "Operation counts should match"
        );
        assert!(
            (seq.expected_creature_score_gain - par.expected_creature_score_gain).abs() < 1e-6,
            "Score gains should match"
        );
    }

    // Verify each nonlinear candidate matches.
    for (seq, par) in seq_nonlinear.iter().zip(par_nonlinear.iter()) {
        assert_eq!(
            seq.operations.len(),
            par.operations.len(),
            "Operation counts should match"
        );
        assert!(
            (seq.expected_creature_score_gain - par.expected_creature_score_gain).abs() < 1e-6,
            "Score gains should match"
        );
    }
}

/// Test that both compression functions can be called concurrently on the
/// same shared references without data races or panics.
#[test]
fn test_concurrent_compression_no_data_race() {
    let (candidates, creature) = build_test_data();

    // Run multiple times to increase chance of exposing any race condition.
    for _ in 0..10 {
        let (identity, nonlinear) = rayon::join(
            || compress_identity_candidates(&candidates, &creature),
            || compress_nonlinear_candidates(&candidates, &creature),
        );

        // Both should produce valid (possibly empty) results without panicking.
        assert!(
            identity.len() <= candidates.len(),
            "Should not produce more compressed than input candidates"
        );
        assert!(
            nonlinear.len() <= candidates.len(),
            "Should not produce more compressed than input candidates"
        );
    }
}

/// Test that combining the parallel results produces the same merged vector
/// as the sequential approach used in orchestration.
#[test]
fn test_parallel_combined_output_matches_sequential() {
    let (candidates, creature) = build_test_data();

    // Sequential: identity then nonlinear, then combine.
    let mut seq_combined = compress_identity_candidates(&candidates, &creature);
    seq_combined.extend(compress_nonlinear_candidates(&candidates, &creature));

    // Parallel: both at once, then combine.
    let (par_identity, par_nonlinear) = rayon::join(
        || compress_identity_candidates(&candidates, &creature),
        || compress_nonlinear_candidates(&candidates, &creature),
    );
    let mut par_combined = par_identity;
    par_combined.extend(par_nonlinear);

    assert_eq!(
        seq_combined.len(),
        par_combined.len(),
        "Combined candidate count should match"
    );
}
