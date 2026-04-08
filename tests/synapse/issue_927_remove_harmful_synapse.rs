//! Issue #927: Verify remove-synapse discovery for harmful connections.
//!
//! NEAT-AI has a scenario test (`DiscoveryScenarioRemoveHarmfulSynapse.ts`) that verifies
//! the Rust discovery engine can identify a harmful synapse that should be removed.
//!
//! Scenario (inverse of add-synapse — the cripple adds a harmful synapse):
//! ```text
//! Whole creature (no harmful synapse):
//!   input-0 --(1.0)--> hidden-A (RELU) --(1.0)--> output-0
//!   input-1 --(1.0)--> hidden-B (TANH) --(1.0)--> output-0
//!
//! Crippled creature (harmful cross-connection added):
//!   input-0 --(1.0)--> hidden-A (RELU) --(1.0)--> output-0
//!   input-1 --(1.0)--> hidden-B (TANH) --(1.0)--> output-0
//!                       hidden-A --(-2.0)--> hidden-B   <-- harmful synapse
//! ```
//!
//! The harmful hidden-A -> hidden-B synapse (weight -2.0) creates interference,
//! suppressing hidden-B's activation. Discovery should identify this synapse as
//! harmful and recommend its removal.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_all, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    AnalyzeAllInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson,
};
use tempfile::NamedTempFile;

/// Skip test if no GPU available.
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Build the "crippled" creature with the harmful cross-connection.
///
/// ```text
/// input-0 --(1.0)--> hidden-A (RELU) --(1.0)--> output-0
/// input-1 --(1.0)--> hidden-B (TANH) --(1.0)--> output-0
///                     hidden-A --(-2.0)--> hidden-B   <-- harmful synapse
/// ```
///
/// Note: hidden-A must appear before hidden-B in the neuron list because
/// there is a synapse from hidden-A to hidden-B (forward-only ordering).
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
                uuid: "hidden-B".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
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
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-B".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-A".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-B".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // The harmful cross-connection
            SynapseJson {
                from_uuid: "hidden-A".to_string(),
                to_uuid: "hidden-B".to_string(),
                weight: -2.0,
                synapse_type: None,
            },
        ],
    }
}

/// Simulate both the whole and crippled creature forward passes to generate
/// discovery records that reveal the harmful synapse's interference.
///
/// Error convention: `actual - target` (positive = neuron output is too high,
/// negative = too low). This matches the GPU harmful shader's convention where
/// `sign(source_activation * weight) == sign(error)` flags a synapse as harmful.
fn generate_harmful_synapse_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    for obs in 0..200u32 {
        // Vary inputs to create diverse activation patterns
        let phase = obs as f32 / 200.0 * std::f32::consts::PI * 4.0;
        let input_0_val = (0.5 + 0.4 * phase.sin()).max(0.0); // Keep positive for RELU clarity
        let input_1_val = 0.4 + 0.3 * (phase * 1.3).cos();

        // --- hidden-A: RELU(input-0 * 1.0) --- same in both creatures
        let hidden_a_value = input_0_val * 1.0;
        let hidden_a_activation = hidden_a_value.max(0.0); // RELU

        // --- Whole creature (ground truth, no harmful synapse) ---
        // hidden-B (whole): TANH(input-1 * 1.0)
        let hidden_b_whole_value = input_1_val * 1.0;
        let hidden_b_whole_activation = hidden_b_whole_value.tanh();

        // output (whole): IDENTITY(hidden-A * 1.0 + hidden-B_whole * 1.0)
        let output_whole = hidden_a_activation + hidden_b_whole_activation;

        // --- Crippled creature (with harmful synapse hidden-A → hidden-B, weight -2.0) ---
        // hidden-B (crippled): TANH(input-1 * 1.0 + hidden-A * (-2.0))
        let hidden_b_crippled_value = input_1_val * 1.0 + hidden_a_activation * (-2.0);
        let hidden_b_crippled_activation = hidden_b_crippled_value.tanh();

        // output (crippled): IDENTITY(hidden-A * 1.0 + hidden-B_crippled * 1.0)
        let output_crippled = hidden_a_activation + hidden_b_crippled_activation;

        // Error = actual - target (negative = output is too low due to harmful synapse)
        let output_error = output_crippled - output_whole;

        // Backpropagate error through the network (actual - target convention)
        // hidden-B error: output_error propagated through weight 1.0
        // Negative because the harmful synapse suppresses hidden-B's activation
        let hidden_b_error = output_error * 1.0;

        // hidden-A error: backprop through both output (weight 1.0) and hidden-B (weight -2.0)
        let hidden_a_error = output_error * 1.0 + hidden_b_error * (-2.0);

        // --- Create records for the crippled creature ---
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

        // hidden-A
        records.push(DiscoverRecord::new(
            obs,
            "hidden-A".to_string(),
            Some(hidden_a_value),
            hidden_a_activation,
            vec![hidden_a_error],
        ));

        // hidden-B (crippled — with harmful synapse interference)
        records.push(DiscoverRecord::new(
            obs,
            "hidden-B".to_string(),
            Some(hidden_b_crippled_value),
            hidden_b_crippled_activation,
            vec![hidden_b_error],
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

/// Issue #927: Discovery should identify the harmful hidden-A -> hidden-B synapse
/// and produce a remove-synapse candidate via `analyze_synapses`.
///
/// The harmful synapse (weight -2.0) suppresses hidden-B's activation by injecting
/// a strong negative signal from hidden-A. Removing it would restore hidden-B's
/// independent contribution and reduce creature error.
#[test]
fn test_remove_harmful_synapse_discovery() {
    skip_without_gpu!();

    let creature = create_crippled_creature();
    let records = generate_harmful_synapse_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    // Focus on output-0 and hidden-B (both affected by the harmful synapse)
    let input = AnalyzeSynapsesInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string(), "hidden-B".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
        temperature: 1.0,
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    println!("\n=== Issue #927: Remove Harmful Synapse Discovery ===");
    println!(
        "Harmful synapse candidates: {}",
        result.harmful_synapses.len()
    );
    for candidate in &result.harmful_synapses {
        println!(
            "  {} -> {}: weight={:.2} impact={:.4} error_reduction={:.6} score_gain={:.6}",
            candidate.from_neuron_uuid,
            candidate.to_neuron_uuid,
            candidate.weight,
            candidate.target_neuron_impact,
            candidate.expected_creature_error_reduction,
            candidate.expected_creature_score_gain,
        );
    }

    println!(
        "Coordinated structural candidates: {}",
        result.coordinated_structural_candidates.len()
    );
    for candidate in &result.coordinated_structural_candidates {
        println!(
            "  ops={:?} gain={:.6} comment={:?}",
            candidate
                .operations
                .iter()
                .map(|op| format!("{op:?}"))
                .collect::<Vec<_>>(),
            candidate.expected_creature_score_gain,
            candidate.comment,
        );
    }

    // The harmful synapse hidden-A -> hidden-B should be detected.
    // It may appear as a harmful_synapses entry or as a coordinated structural
    // candidate with a RemoveSynapse operation.
    let found_in_harmful = result.harmful_synapses.iter().any(|c| {
        c.from_neuron_uuid == "hidden-A"
            && c.to_neuron_uuid == "hidden-B"
            && c.expected_creature_score_gain > 0.0
    });

    let found_in_coordinated = result.coordinated_structural_candidates.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(op,
                neat_ai_discovery::CoordinatedStructuralOpJson::RemoveSynapse {
                    from_neuron_uuid, to_neuron_uuid
                } if from_neuron_uuid == "hidden-A" && to_neuron_uuid == "hidden-B"
            )
        }) && c.expected_creature_score_gain > 0.0
    });

    assert!(
        found_in_harmful || found_in_coordinated,
        "Discovery should identify the harmful hidden-A -> hidden-B synapse \
         (weight -2.0) for removal. Found neither in harmful_synapses nor in \
         coordinated_structural_candidates."
    );

    // If found in harmful_synapses, validate the candidate properties
    if let Some(candidate) = result
        .harmful_synapses
        .iter()
        .find(|c| c.from_neuron_uuid == "hidden-A" && c.to_neuron_uuid == "hidden-B")
    {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Removing the harmful synapse should improve the creature's score, got {:.6}",
            candidate.expected_creature_score_gain
        );
        assert!(
            candidate.expected_creature_error_reduction > 0.0,
            "Removing the harmful synapse should reduce error, got {:.6}",
            candidate.expected_creature_error_reduction
        );
        assert_eq!(
            candidate.weight, -2.0,
            "The harmful synapse weight should be -2.0, got {:.2}",
            candidate.weight
        );
        println!(
            "SUCCESS: Found harmful synapse hidden-A -> hidden-B in harmful_synapses \
             with score_gain={:.6}",
            candidate.expected_creature_score_gain
        );
    }

    // If found in coordinated structural candidates, validate the candidate
    if found_in_coordinated {
        println!(
            "SUCCESS: Found remove-synapse for hidden-A -> hidden-B in coordinated_structural_candidates"
        );
    }

    println!("Test passed: Harmful synapse hidden-A -> hidden-B detected for removal");
}

/// Issue #927: Verify the harmful synapse candidate has correct source and target neurons.
///
/// When the harmful synapse is detected, the candidate must identify:
/// - The correct source neuron (hidden-A)
/// - The correct target neuron (hidden-B)
/// - A positive expected score gain (removal would help)
#[test]
fn test_harmful_synapse_identifies_correct_neurons() {
    skip_without_gpu!();

    let creature = create_crippled_creature();
    let records = generate_harmful_synapse_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string(), "hidden-B".to_string()],
        max_candidates: Some(100),
        analysis_deadline_ms: Some(30000),
        random_seed: None,
        temperature: 1.0,
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // Collect all removal candidates from both result lists
    let mut removal_from_uuids = Vec::new();
    let mut removal_to_uuids = Vec::new();

    for c in &result.harmful_synapses {
        removal_from_uuids.push(c.from_neuron_uuid.as_str());
        removal_to_uuids.push(c.to_neuron_uuid.as_str());
    }

    for c in &result.coordinated_structural_candidates {
        for op in &c.operations {
            if let neat_ai_discovery::CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
            } = op
            {
                removal_from_uuids.push(from_neuron_uuid.as_str());
                removal_to_uuids.push(to_neuron_uuid.as_str());
            }
        }
    }

    println!(
        "Removal candidates found: {} harmful + {} coordinated",
        result.harmful_synapses.len(),
        result.coordinated_structural_candidates.len()
    );

    // At least one removal candidate should target the hidden-A -> hidden-B synapse
    let has_harmful_synapse = removal_from_uuids
        .iter()
        .zip(removal_to_uuids.iter())
        .any(|(from, to)| *from == "hidden-A" && *to == "hidden-B");

    assert!(
        has_harmful_synapse,
        "Should find a removal candidate for the harmful hidden-A -> hidden-B synapse. \
         Removal sources found: {removal_from_uuids:?}, targets found: {removal_to_uuids:?}",
    );
}

/// Issue #927: Verify the full pipeline (`analyze_all`) detects the harmful synapse.
///
/// This ensures the combined synapse + neuron analysis pipeline also identifies
/// the harmful cross-connection for removal.
#[test]
fn test_analyze_all_finds_harmful_synapse() {
    skip_without_gpu!();

    let creature = create_crippled_creature();
    let records = generate_harmful_synapse_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeAllInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string(), "hidden-B".to_string()],
        max_synapse_candidates: Some(100),
        max_neuron_candidates: Some(20),
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(false),
        random_seed: Some(42),
        temperature: 1.0,
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    let synapse_result = result
        .synapse
        .as_ref()
        .expect("Synapse analysis should have run");

    println!(
        "analyze_all harmful synapse candidates: {}",
        synapse_result.harmful_synapses.len()
    );
    println!(
        "analyze_all coordinated structural candidates: {}",
        synapse_result.coordinated_structural_candidates.len()
    );

    // Check harmful_synapses
    let found_in_harmful = synapse_result.harmful_synapses.iter().any(|c| {
        c.from_neuron_uuid == "hidden-A"
            && c.to_neuron_uuid == "hidden-B"
            && c.expected_creature_score_gain > 0.0
    });

    // Check coordinated structural candidates
    let found_in_coordinated = synapse_result
        .coordinated_structural_candidates
        .iter()
        .any(|c| {
            c.operations.iter().any(|op| {
                matches!(op,
                    neat_ai_discovery::CoordinatedStructuralOpJson::RemoveSynapse {
                        from_neuron_uuid, to_neuron_uuid
                    } if from_neuron_uuid == "hidden-A" && to_neuron_uuid == "hidden-B"
                )
            }) && c.expected_creature_score_gain > 0.0
        });

    for c in &synapse_result.harmful_synapses {
        println!(
            "  harmful: {} -> {} gain={:.6}",
            c.from_neuron_uuid, c.to_neuron_uuid, c.expected_creature_score_gain
        );
    }

    assert!(
        found_in_harmful || found_in_coordinated,
        "analyze_all should identify the harmful hidden-A -> hidden-B synapse for removal"
    );

    println!("SUCCESS: analyze_all pipeline detects harmful synapse for removal");
}
