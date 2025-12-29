//! Regression tests for Issue #134.
//!
//! Goal: prove (via public API) that simulating the target neuron’s non-linear squash can
//! flip the sign of the predicted improvement compared with the linear VALUE-domain model.
//!
//! We keep this under `tests/` (integration tests) to keep the implementation module readable.

mod common;

use neat_ai_discovery::analysis::analyze_synapses;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use tempfile::tempdir;

fn bent_identity(x: f32) -> f32 {
    // Matches NEAT-AI BENT_IDENTITY:
    // f(x) = (sqrt(x^2 + 1) - 1)/2 + x
    ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x
}

/// Issue #134: Construct a scenario where the linear model would accept a helpful synapse,
/// but target activation simulation (BENT_IDENTITY) rejects it as net harmful.
#[test]
fn issue_134_bent_identity_target_simulation_rejects_linear_false_positive() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Use a constant source activation that matches ArcTan(1) ≈ 0.7853982.
    // This mimics an ArcTan intermediate signal without needing to model the new neuron
    // directly in the synapse analysis API.
    let arctan_1 = std::f32::consts::FRAC_PI_4;

    // Two samples with opposing error signs and strongly asymmetric target values.
    //
    // In VALUE-domain linear approximation, the optimal weight is positive and appears helpful.
    // In ACTIVATION-domain with BENT_IDENTITY simulation, that same optimal weight is net harmful.
    let records = vec![
        // obs 0: large positive VALUE-domain error, negative target_value (low local slope)
        DiscoverRecord::new(0, "input-0".to_string(), None, arctan_1, Vec::new()),
        DiscoverRecord::new(
            0,
            "output-0".to_string(),
            Some(-10.0),
            bent_identity(-10.0),
            vec![2.0],
        ),
        // obs 1: negative error, positive target_value (high local slope)
        DiscoverRecord::new(1, "input-0".to_string(), None, arctan_1, Vec::new()),
        DiscoverRecord::new(
            1,
            "output-0".to_string(),
            Some(10.0),
            bent_identity(10.0),
            vec![-1.0],
        ),
    ];

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "BENT_IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file: parquet_file.clone(),
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
    };

    let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

    // With correct target simulation, this linear-optimal candidate is net harmful
    // and must not be returned.
    assert!(
        result.helpful_synapses.is_empty(),
        "Expected no helpful synapses when BENT_IDENTITY target simulation is enabled for this constructed false-positive scenario"
    );
    assert!(
        !result.no_candidate_reasons.is_empty(),
        "Expected diagnostics when no candidates are returned"
    );
}

/// Sanity check: the same record pattern with an IDENTITY target should produce a helpful synapse.
/// This ensures the dataset is not degenerate and that we genuinely rely on target simulation for
/// BENT_IDENTITY to avoid false positives.
#[test]
fn issue_134_identity_target_accepts_linear_candidate_sanity_check() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let arctan_1 = std::f32::consts::FRAC_PI_4;

    let records = vec![
        DiscoverRecord::new(0, "input-0".to_string(), None, arctan_1, Vec::new()),
        DiscoverRecord::new(0, "output-0".to_string(), Some(-10.0), -10.0, vec![2.0]),
        DiscoverRecord::new(1, "input-0".to_string(), None, arctan_1, Vec::new()),
        DiscoverRecord::new(1, "output-0".to_string(), Some(10.0), 10.0, vec![-1.0]),
    ];

    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    };

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
    };

    let result = analyze_synapses(&input).expect("Synapse analysis should succeed");
    assert!(
        !result.helpful_synapses.is_empty(),
        "Expected at least one helpful synapse for IDENTITY target in the sanity-check scenario"
    );
}
