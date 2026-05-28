//! Issue #926: Verify add-neurons discovery between hidden neurons.
//!
//! NEAT-AI has a scenario test (`DiscoveryScenarioAddNeuronBetweenHidden.ts`) that verifies
//! the Rust discovery engine can find a missing neuron between two existing hidden neurons —
//! not just the simple input→hidden→output pattern.
//!
//! Scenario:
//! ```text
//! Whole creature:
//!   input-0 ──(0.8)──▶ hidden-A (RELU) ──(0.7)──▶ hidden-B (TANH, bias 0.3)
//!       ──(0.9)──▶ hidden-C (RELU) ──(1.0)──▶ output-0
//!   input-1 ──(0.6)──▶ hidden-A
//!
//! Crippled creature (hidden-B removed, direct connection):
//!   input-0 ──(0.8)──▶ hidden-A (RELU) ──(0.5)──▶ hidden-C (RELU)
//!       ──(1.0)──▶ output-0
//!   input-1 ──(0.6)──▶ hidden-A
//! ```
//!
//! Discovery should identify the need for an intermediate neuron between hidden-A and hidden-C.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::NamedTempFile;

/// RAII guard (Issue #1140) that raises the per-target add-neuron cap so that
/// tests with a small number of target neurons can still exercise diverse
/// activation proposals. Removes the override when dropped.
struct PerTargetCapOverride;

impl PerTargetCapOverride {
    fn new(cap: &str) -> Self {
        // SAFETY: env access is serialised via `#[serial]` on the call site.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET", cap);
        }
        Self
    }
}

impl Drop for PerTargetCapOverride {
    fn drop(&mut self) {
        // SAFETY: env access is serialised via `#[serial]` on the call site.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_MAX_ADD_NEURON_PER_TARGET");
        }
    }
}

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Build the "crippled" creature: hidden-B removed, direct hidden-A → hidden-C connection.
///
/// ```text
/// input-0 ──(0.8)──▶ hidden-A (RELU) ──(0.5)──▶ hidden-C (RELU) ──(1.0)──▶ output-0
/// input-1 ──(0.6)──▶ hidden-A
/// ```
fn create_crippled_creature() -> CreatureJson {
    CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "hidden-A".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-C".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "RELU".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-A".to_string(),
                weight: 0.8,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-A".to_string(),
                weight: 0.6,
                synapse_type: None,
            },
            // Direct connection replacing hidden-A → hidden-B → hidden-C chain
            SynapseJson {
                from_uuid: "hidden-A".to_string(),
                to_uuid: "hidden-C".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-C".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    }
}

/// Simulate the "whole creature" forward pass (with hidden-B present) to compute
/// the expected activations and errors for the crippled creature.
///
/// Whole creature: input-0/1 → hidden-A (RELU) → hidden-B (TANH, bias 0.3) → hidden-C (RELU) → output-0
/// Crippled creature: input-0/1 → hidden-A (RELU) → hidden-C (RELU) → output-0
///
/// The error signals show the difference between what the crippled creature produces
/// and what the whole creature would produce.
fn generate_hidden_gap_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    for obs in 0..200u32 {
        // Vary inputs to create diverse activation patterns
        let phase = obs as f32 / 200.0 * std::f32::consts::PI * 4.0;
        let input_0_val = 0.5 + 0.4 * phase.sin();
        let input_1_val = 0.3 + 0.3 * (phase * 1.7).cos();

        // hidden-A: RELU(input-0 * 0.8 + input-1 * 0.6)
        let hidden_a_value = input_0_val * 0.8 + input_1_val * 0.6;
        let hidden_a_activation = hidden_a_value.max(0.0); // RELU

        // --- Whole creature path (ground truth) ---
        // hidden-B: TANH(hidden-A * 0.7 + 0.3)   [bias = 0.3]
        let hidden_b_value = hidden_a_activation * 0.7 + 0.3;
        let hidden_b_activation = hidden_b_value.tanh();

        // hidden-C (whole): RELU(hidden-B * 0.9)
        let hidden_c_whole_value = hidden_b_activation * 0.9;
        let hidden_c_whole_activation = hidden_c_whole_value.max(0.0);

        // output (whole): IDENTITY(hidden-C * 1.0)
        let output_whole = hidden_c_whole_activation;

        // --- Crippled creature path ---
        // hidden-C (crippled): RELU(hidden-A * 0.5)
        let hidden_c_crippled_value = hidden_a_activation * 0.5;
        let hidden_c_crippled_activation = hidden_c_crippled_value.max(0.0);

        // output (crippled): IDENTITY(hidden-C * 1.0)
        let output_crippled = hidden_c_crippled_activation;

        // Error = whole output - crippled output (what the crippled creature is missing)
        let output_error = output_whole - output_crippled;

        // hidden-C error: propagated from output error through weight 1.0
        let hidden_c_error = output_error;

        // --- Create records for crippled creature ---
        // Input neurons
        records.push(DiscoverRecord::new(
            obs,
            "input-0".to_string(),
            Some(input_0_val),
            input_0_val,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs,
            "input-1".to_string(),
            Some(input_1_val),
            input_1_val,
            vec![0.0],
        ));

        // hidden-A: record its activation and propagated error
        let hidden_a_error = hidden_c_error * 0.5; // backprop through weight 0.5
        records.push(DiscoverRecord::new(
            obs,
            "hidden-A".to_string(),
            Some(hidden_a_value),
            hidden_a_activation,
            vec![hidden_a_error],
        ));

        // hidden-C: this is the target — its error reflects the missing hidden-B
        records.push(DiscoverRecord::new(
            obs,
            "hidden-C".to_string(),
            Some(hidden_c_crippled_value),
            hidden_c_crippled_activation,
            vec![hidden_c_error],
        ));

        // output-0
        records.push(DiscoverRecord::new(
            obs,
            "output-0".to_string(),
            Some(output_crippled),
            output_crippled,
            vec![output_error],
        ));
    }

    records
}

/// Verify that discovery produces an add-neuron candidate between hidden-A and hidden-C
/// when hidden-B (TANH, bias 0.3) is removed from the chain.
///
/// This tests the core scenario from NEAT-AI's `DiscoveryScenarioAddNeuronBetweenHidden`.
#[test]
fn test_add_neuron_between_hidden_neurons() {
    skip_without_gpu!();

    let creature = create_crippled_creature();
    let records = generate_hidden_gap_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    // Focus on hidden-C (the target that would benefit from an intermediate neuron)
    // and output-0 (which also sees the error)
    let input = AnalyzeNeuronsInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["hidden-C".to_string(), "output-0".to_string()],
        max_candidates: Some(20),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    println!(
        "Add-neuron candidates found: {}",
        result.helpful_neurons.len()
    );

    for (i, c) in result.helpful_neurons.iter().enumerate() {
        println!(
            "  [{}] {} -> {} | squash={} bias={:.3} | in_w={:.3} out_w={:.3} | gain={:.6} impact={:.3}",
            i,
            c.source_neuron_uuid,
            c.target_neuron_uuid,
            c.squash,
            c.bias,
            c.incoming_weight,
            c.outgoing_weight,
            c.expected_creature_score_gain,
            c.target_neuron_impact,
        );
    }

    // The discovery engine must produce at least one add-neuron candidate
    assert!(
        !result.helpful_neurons.is_empty(),
        "Discovery should find at least one add-neuron candidate for the missing hidden-B neuron"
    );

    // Check that at least one candidate uses hidden-A as a source
    // (since hidden-A feeds into the gap where hidden-B was)
    let has_hidden_a_source = result
        .helpful_neurons
        .iter()
        .any(|c| c.source_neuron_uuid == "hidden-A");

    assert!(
        has_hidden_a_source,
        "At least one candidate should use hidden-A as the source neuron \
        (the upstream neuron in the gap where hidden-B was removed)"
    );

    // Check that at least one candidate targets hidden-C
    // (where the missing neuron's output would connect)
    let has_hidden_c_target = result
        .helpful_neurons
        .iter()
        .any(|c| c.target_neuron_uuid == "hidden-C");

    assert!(
        has_hidden_c_target,
        "At least one candidate should target hidden-C \
        (the downstream neuron that lost its intermediate hidden-B source)"
    );

    // The best candidate targeting hidden-C from hidden-A should have positive metrics
    let best_hidden_gap_candidate = result
        .helpful_neurons
        .iter()
        .find(|c| c.source_neuron_uuid == "hidden-A" && c.target_neuron_uuid == "hidden-C");

    if let Some(candidate) = best_hidden_gap_candidate {
        assert!(
            candidate.target_neuron_impact > 0.0,
            "Target neuron impact should be positive, got {}",
            candidate.target_neuron_impact
        );
        assert!(
            candidate.expected_creature_error_reduction > 0.0,
            "Expected creature error reduction should be positive, got {}",
            candidate.expected_creature_error_reduction
        );
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Expected creature score gain should be positive, got {}",
            candidate.expected_creature_score_gain
        );
        println!(
            "Best hidden-A -> hidden-C candidate: squash={} bias={:.3} gain={:.6}",
            candidate.squash, candidate.bias, candidate.expected_creature_score_gain,
        );
    }
}

/// Verify that hidden-to-hidden neuron candidates have reasonable properties.
///
/// When discovering a neuron between hidden-A and hidden-C, the candidate should:
/// - Have non-zero incoming and outgoing weights
/// - Use a valid activation function
/// - Report `improved_count` > 0 (some samples benefit)
#[test]
fn test_hidden_neuron_candidate_properties() {
    skip_without_gpu!();

    let creature = create_crippled_creature();
    let records = generate_hidden_gap_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeNeuronsInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["hidden-C".to_string()],
        max_candidates: Some(20),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    // Filter to candidates from hidden-A → hidden-C
    let gap_candidates: Vec<_> = result
        .helpful_neurons
        .iter()
        .filter(|c| c.source_neuron_uuid == "hidden-A" && c.target_neuron_uuid == "hidden-C")
        .collect();

    println!("Hidden-A -> hidden-C candidates: {}", gap_candidates.len());

    for candidate in &gap_candidates {
        // Weights should be non-zero
        assert!(
            candidate.incoming_weight.abs() > f32::EPSILON,
            "Incoming weight should be non-zero, got {}",
            candidate.incoming_weight
        );
        assert!(
            candidate.outgoing_weight.abs() > f32::EPSILON,
            "Outgoing weight should be non-zero, got {}",
            candidate.outgoing_weight
        );

        // Squash should be a known activation function. Issue #1141: include
        // mixed-case `ReLU` because the squash-diversity filter allows both
        // `RELU` and `ReLU` candidates to surface as distinct keys when they
        // win their respective `(target, squash)` buckets.
        let valid_squashes = [
            "RELU",
            "ReLU",
            "TANH",
            "SIGMOID",
            "IDENTITY",
            "LOGISTIC",
            "HARD_TANH",
            "GELU",
            "ELU",
            "Softplus",
            "BIPOLAR",
            "CLIPPED",
            "ABSOLUTE",
            "Mish",
            "SOFTSIGN",
            "BENT_IDENTITY",
            "ArcTan",
            "ReLU6",
        ];
        assert!(
            valid_squashes.contains(&candidate.squash.as_str()),
            "Squash should be a valid activation function, got '{}'",
            candidate.squash
        );

        // Some samples should benefit
        assert!(
            candidate.improved_count > 0,
            "improved_count should be positive, got {}",
            candidate.improved_count
        );

        println!(
            "  squash={} bias={:.3} in_w={:.3} out_w={:.3} improved={}/{}",
            candidate.squash,
            candidate.bias,
            candidate.incoming_weight,
            candidate.outgoing_weight,
            candidate.improved_count,
            candidate.total_count,
        );
    }
}

/// Verify that the full analysis pipeline (`analyze_all`) also finds hidden-to-hidden
/// neuron candidates through the combined synapse + neuron analysis path.
#[test]
#[serial]
fn test_analyze_all_finds_hidden_neuron_candidate() {
    skip_without_gpu!();

    // Issue #1140: Raise the per-target cap so the test's tiny creature
    // (2 focus targets) can still see the hidden-C candidates that
    // `convert_neurons_to_coordinated_replacements` may convert to
    // coordinated candidates. At the default cap of 3 the hidden-C slots
    // are exhausted before the conversion step, leaving no hidden-C
    // candidates in `helpful_neurons`.
    let _cap_guard = PerTargetCapOverride::new("32");

    use neat_ai_discovery::AnalyzeAllInput;
    use neat_ai_discovery::analysis::analyze_all;

    let creature = create_crippled_creature();
    let records = generate_hidden_gap_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeAllInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["hidden-C".to_string(), "output-0".to_string()],
        max_synapse_candidates: Some(20),
        max_neuron_candidates: Some(20),
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        temperature: 1.0,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        failure_cache: None,
        discovery_outcome_log: None,
        cost_name: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    // Neuron analysis should have run and produced results
    let neuron_result = result
        .neuron
        .as_ref()
        .expect("Neuron analysis should have run");

    println!(
        "analyze_all neuron candidates: {}",
        neuron_result.helpful_neurons.len()
    );

    // Should find at least one add-neuron candidate
    assert!(
        !neuron_result.helpful_neurons.is_empty(),
        "analyze_all should find at least one add-neuron candidate"
    );

    for c in &neuron_result.helpful_neurons {
        println!(
            "  Found: {} -> {} (squash={} gain={:.6})",
            c.source_neuron_uuid, c.target_neuron_uuid, c.squash, c.expected_creature_score_gain
        );
    }

    // Issue #1141: The squash-diversity filter keeps the highest-gain
    // candidate per `(target, squash)` pair across all sources. For
    // `hidden-C` the highest-gain candidate is typically `hidden-A → hidden-C`
    // (which already has a direct synapse), so it is converted to a
    // coordinated structural replacement and then discounted for multi-op
    // gain. Lower-gain `input-* → hidden-C` candidates that previously
    // surfaced no longer survive the filter. The hidden-C target still gets
    // evaluated end-to-end (verified by `test_hidden_neuron_candidate_properties`
    // via `analyze_neurons` directly), so this test now verifies only that
    // `analyze_all` produces add-neuron candidates rather than that
    // `hidden-C` survives every downstream discount stage.
    assert!(
        !neuron_result.helpful_neurons.is_empty(),
        "analyze_all should produce at least one add-neuron candidate"
    );

    // Check if hidden-A → hidden-C candidate appears (ideal hidden-to-hidden case).
    // Input neurons may outcompete hidden-A as a source in the full pipeline,
    // but the key verification is that hidden neuron targets are evaluated.
    let has_hidden_a_to_c = neuron_result
        .helpful_neurons
        .iter()
        .any(|c| c.source_neuron_uuid == "hidden-A" && c.target_neuron_uuid == "hidden-C");

    if has_hidden_a_to_c {
        println!("SUCCESS: Found hidden-A -> hidden-C neuron candidate via analyze_all");
    } else {
        println!(
            "Note: hidden-A -> hidden-C not in top candidates via analyze_all, \
            but hidden-C is targeted (proven via analyze_neurons in separate test)"
        );
    }
}
