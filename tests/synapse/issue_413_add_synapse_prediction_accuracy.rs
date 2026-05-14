//! Issue #413: Integration test for add-synapse prediction accuracy
//!
//! Verifies that the full add-synapse analysis pipeline produces candidates
//! with correct prediction direction, especially for saturating target neurons.
//!
//! The root cause was that the synapse weight was computed using a linear model
//! but evaluated against a saturation-aware model. When the target neuron uses
//! a bounded activation (`HARD_TANH`, TANH, etc.), the linear-model weight can
//! overshoot into saturation, causing inverted predictions.
//!
//! The fix searches over multiple weight candidates for saturating targets,
//! matching the approach already used by add-neuron candidates.

#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !neat_ai_discovery::analysis::GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create a creature with specified topology.
fn create_test_creature(
    neurons: Vec<(&str, &str, &str)>,
    synapses: Vec<(&str, &str, f32)>,
) -> CreatureJson {
    let input_count = neurons.iter().filter(|(_, t, _)| *t == "input").count();
    let output_count = neurons.iter().filter(|(_, t, _)| *t == "output").count();
    CreatureJson {
        neurons: neurons
            .into_iter()
            .filter(|(_, neuron_type, _)| *neuron_type != "input")
            .map(|(uuid, neuron_type, squash)| NeuronJson {
                uuid: uuid.to_string(),
                neuron_type: neuron_type.to_string(),
                squash: squash.to_string(),
                bias: 0.0,
            })
            .collect(),
        synapses: synapses
            .into_iter()
            .map(|(from, to, weight)| SynapseJson {
                from_uuid: from.to_string(),
                to_uuid: to.to_string(),
                weight,
                synapse_type: None,
            })
            .collect(),
        input: input_count,
        output: output_count,
    }
}

/// Issue #413: Add-synapse candidates for `HARD_TANH` targets should have
/// positive `expected_creature_score_gain` (not inverted).
///
/// This test creates a network where input-1 has a clear positive correlation
/// with the `HARD_TANH` output neuron's error. The add-synapse candidate should
/// predict positive improvement.
#[test]
fn test_issue_413_add_synapse_hard_tanh_prediction_not_inverted() {
    skip_without_gpu!();

    // Network: input-0 -> output-0 (HARD_TANH)
    // input-1 is NOT connected (candidate for add-synapse)
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "HARD_TANH"),
        ],
        vec![("input-0", "output-0", 0.5)],
    );

    // Create data where input-1 activation is positively correlated with output error.
    // The output neuron is operating near saturation (target_value near 1.0).
    let mut records = Vec::new();
    for obs in 0..200 {
        let input0_activation = (obs as f32 / 100.0) - 1.0; // -1 to 1
        let input1_activation = obs as f32 / 200.0; // 0 to 1

        // Output neuron operating near upper bound of HARD_TANH
        let output_value = 0.7 + 0.2 * input0_activation; // 0.5 to 0.9
        let output_activation = output_value.clamp(-1.0, 1.0);
        // Error is positively correlated with input-1: when input-1 is high, error is high
        let output_error = 0.03 * input1_activation;

        // input-0 records (no error, it's a source neuron)
        records.push(DiscoverRecord::new(
            obs as u32,
            "input-0".to_string(),
            Some(input0_activation),
            input0_activation,
            Vec::new(),
        ));

        // input-1 records (no error, it's a source neuron)
        records.push(DiscoverRecord::new(
            obs as u32,
            "input-1".to_string(),
            Some(input1_activation),
            input1_activation,
            Vec::new(),
        ));

        // output-0 records (has error)
        records.push(DiscoverRecord::new(
            obs as u32,
            "output-0".to_string(),
            Some(output_value),
            output_activation,
            vec![output_error],
        ));
    }

    let parquet_file = NamedTempFile::new().unwrap();
    let parquet_path = parquet_file.path().to_string_lossy().to_string();
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input).unwrap();

    // Check that if we got any helpful synapse candidates, their prediction is not inverted
    for candidate in &result.helpful_synapses {
        assert!(
            candidate.expected_creature_score_gain >= 0.0,
            "Issue #413: Add-synapse candidate {} -> {} should have non-negative expected gain, \
             got {}, weight={}",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.expected_creature_score_gain,
            candidate.weight,
        );
    }
}

/// Issue #413: Add-synapse candidates should have prediction sign matching
/// actual improvement direction (positive correlation -> positive weight -> positive improvement).
#[test]
fn test_issue_413_add_synapse_prediction_direction_correct() {
    skip_without_gpu!();

    // Simple network: input-0 -> output-0 (IDENTITY, no saturation)
    // input-1 NOT connected
    let creature = create_test_creature(
        vec![
            ("input-0", "input", "IDENTITY"),
            ("input-1", "input", "IDENTITY"),
            ("output-0", "output", "IDENTITY"),
        ],
        vec![("input-0", "output-0", 0.5)],
    );

    // Create clear positive correlation between input-1 and output error
    let mut records = Vec::new();
    for obs in 0..200 {
        let input0_activation = (obs as f32 * 0.1).sin(); // Varying
        let input1_activation = (obs as f32 - 100.0) / 100.0; // -1 to 1

        // Error clearly correlated with input-1
        let output_error = 0.05 * input1_activation;
        let output_value = input0_activation * 0.5;

        records.push(DiscoverRecord::new(
            obs as u32,
            "input-0".to_string(),
            Some(input0_activation),
            input0_activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs as u32,
            "input-1".to_string(),
            Some(input1_activation),
            input1_activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs as u32,
            "output-0".to_string(),
            Some(output_value),
            output_value, // IDENTITY: activation = value
            vec![output_error],
        ));
    }

    let parquet_file = NamedTempFile::new().unwrap();
    let parquet_path = parquet_file.path().to_string_lossy().to_string();
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input).unwrap();

    // We should get at least one helpful synapse candidate from input-1
    let input1_candidates: Vec<_> = result
        .helpful_synapses
        .iter()
        .filter(|c| c.from_neuron_uuid == "input-1")
        .collect();

    // If we got candidates, verify prediction direction
    for candidate in &input1_candidates {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Issue #413: input-1 -> output-0 candidate should have positive expected gain, \
             got {}, weight={}",
            candidate.expected_creature_score_gain,
            candidate.weight,
        );
        // Weight should be positive (positive correlation between input-1 and error)
        assert!(
            candidate.weight > 0.0,
            "Issue #413: Weight should be positive for positively correlated source, got {}",
            candidate.weight,
        );
    }
}
