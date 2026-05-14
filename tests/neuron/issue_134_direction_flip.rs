//! Regression tests for Issue #134.
//!
//! Goal: prove (via public API) that simulating the target neuron’s non-linear squash can
//! flip the sign of the predicted improvement compared with the linear VALUE-domain model.
//!
//! We keep this under `tests/` (integration tests) to keep the implementation module readable.

use crate::skip_without_gpu;
use neat_ai_discovery::analysis::analyze_synapses;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use serial_test::serial;
use tempfile::tempdir;

fn bent_identity(x: f32) -> f32 {
    // Matches NEAT-AI BENT_IDENTITY:
    // f(x) = (sqrt(x^2 + 1) - 1)/2 + x
    ((x * x + 1.0).sqrt() - 1.0) / 2.0 + x
}

/// Issue #134: Construct a scenario where the linear model would accept a helpful synapse,
/// but target activation simulation (`BENT_IDENTITY`) rejects it as net harmful.
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
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

    // With correct target simulation, this linear-optimal candidate is net harmful
    // and must not be returned.
    assert!(
        result.helpful_synapses.is_empty(),
        "Expected no helpful synapses when BENT_IDENTITY target simulation is enabled for this constructed false-positive scenario"
    );
    // Note: diagnostics may or may not be present depending on whether the
    // multi-weight search (Issue #730) creates an intermediate candidate that
    // is later filtered in post-processing. The key outcome is that no helpful
    // synapses are returned.
}

/// Sanity check: the same record pattern with an IDENTITY target should produce a helpful synapse.
/// This ensures the dataset is not degenerate and that we genuinely rely on target simulation for
/// `BENT_IDENTITY` to avoid false positives.
#[test]
#[serial]
fn issue_134_identity_target_accepts_linear_candidate_sanity_check() {
    skip_without_gpu!();

    // Issue #1191: this 5-record fixture produces synapse candidates whose
    // post-discount gains fall below the new 1e-5 production noise floor.
    // Disable the floor so the IDENTITY-target sanity check is observable
    // independently.
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN", "0");
    }

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    let arctan_1 = std::f32::consts::FRAC_PI_4;

    // Issue #789: Use 5 observations (not 2) so the improved ratio can exceed the
    // raised MIN_IMPROVED_RATIO threshold of 0.6. With only 2 samples, 1/2=0.5
    // would be filtered out.
    let records = vec![
        DiscoverRecord::new(0, "input-0".to_string(), None, arctan_1, Vec::new()),
        DiscoverRecord::new(0, "output-0".to_string(), Some(-10.0), -10.0, vec![2.0]),
        // Vary the input activation slightly so synapse analysis emits an add-synapse candidate
        // rather than folding the constant source into a `setBias` coordinated candidate (Issue #178).
        DiscoverRecord::new(1, "input-0".to_string(), None, arctan_1 * 0.9, Vec::new()),
        DiscoverRecord::new(1, "output-0".to_string(), Some(10.0), 10.0, vec![-1.0]),
        DiscoverRecord::new(2, "input-0".to_string(), None, arctan_1 * 1.1, Vec::new()),
        DiscoverRecord::new(2, "output-0".to_string(), Some(-5.0), -5.0, vec![1.5]),
        DiscoverRecord::new(3, "input-0".to_string(), None, arctan_1 * 0.8, Vec::new()),
        DiscoverRecord::new(3, "output-0".to_string(), Some(-8.0), -8.0, vec![1.8]),
        DiscoverRecord::new(4, "input-0".to_string(), None, arctan_1 * 1.05, Vec::new()),
        DiscoverRecord::new(4, "output-0".to_string(), Some(-3.0), -3.0, vec![1.2]),
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
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_synapses(&input).expect("Synapse analysis should succeed");

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_MIN_EXPECTED_GAIN");
    }

    assert!(
        !result.helpful_synapses.is_empty(),
        "Expected at least one helpful synapse for IDENTITY target in the sanity-check scenario"
    );
}
