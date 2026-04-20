//! Issue #930: Verify change-squash discovery for suboptimal activation functions.
//!
//! NEAT-AI has a scenario test (`DiscoveryScenarioChangeSquash.ts`) that verifies
//! the Rust discovery engine can identify when a neuron's squash (activation)
//! function has been changed to a suboptimal one.
//!
//! Scenario:
//! ```text
//! Whole creature:
//!   input-0 ──(0.8)──▶ hidden-A (TANH, bias 0.5) ──(1.0)──▶ output-0
//!   input-1 ──(0.6)──▶ hidden-A
//!
//! Crippled creature (squash degraded):
//!   input-0 ──(0.8)──▶ hidden-A (IDENTITY, bias 0.5) ──(1.0)──▶ output-0
//!   input-1 ──(0.6)──▶ hidden-A
//! ```
//!
//! The cripple changes hidden-A's squash from TANH to IDENTITY while keeping all
//! weights and biases identical. The non-linear TANH is clearly better for the
//! test data which requires non-linear separation. Discovery should identify that
//! TANH would be a better activation function.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for neural network computation (Issue #873)

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_all};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    AnalyzeAllInput, CoordinatedStructuralOpJson, CreatureJson, NeuronJson, SynapseJson,
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

/// Build the "crippled" creature with IDENTITY instead of TANH on hidden-A.
///
/// ```text
/// input-0 ──(0.8)──▶ hidden-A (IDENTITY, bias 0.5) ──(1.0)──▶ output-0
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
                squash: "IDENTITY".to_string(),
                bias: 0.5,
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
            SynapseJson {
                from_uuid: "hidden-A".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
    }
}

/// Simulate both the whole and crippled creature forward passes to generate
/// discovery records that reveal the suboptimal activation function.
///
/// The whole creature uses TANH on hidden-A, providing non-linear separation.
/// The crippled creature uses IDENTITY, which passes the raw weighted sum through.
/// The error signals show the difference between these two behaviours.
///
/// Error convention: `actual - target` (positive = neuron output is too high,
/// negative = too low).
fn generate_change_squash_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    let num_observations = 200u32;

    for obs in 0..num_observations {
        // Vary inputs to create diverse activation patterns that benefit from
        // non-linear separation. Use a range that spans both positive and negative
        // pre-activation values so TANH's non-linearity is clearly advantageous.
        let phase = obs as f32 / num_observations as f32 * std::f32::consts::PI * 4.0;
        let input_0_val = 0.5 + 0.5 * phase.sin();
        let input_1_val = 0.3 + 0.4 * (phase * 1.3).cos();

        // --- Pre-activation for hidden-A (same for both creatures) ---
        let hidden_a_pre = input_0_val * 0.8 + input_1_val * 0.6 + 0.5; // bias = 0.5

        // --- Whole creature (ground truth): hidden-A uses TANH ---
        let hidden_a_whole_activation = hidden_a_pre.tanh();
        let output_whole = hidden_a_whole_activation * 1.0; // weight 1.0 to output

        // --- Crippled creature: hidden-A uses IDENTITY ---
        let hidden_a_crippled_activation = hidden_a_pre; // IDENTITY = pass-through
        let output_crippled = hidden_a_crippled_activation * 1.0;

        // Error = actual (crippled) - target (whole)
        let output_error = output_crippled - output_whole;

        // Backpropagate error to hidden-A through output weight 1.0
        let hidden_a_error = output_error * 1.0;

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

        // hidden-A (crippled — with IDENTITY instead of TANH)
        records.push(DiscoverRecord::new(
            obs,
            "hidden-A".to_string(),
            Some(hidden_a_pre),
            hidden_a_crippled_activation,
            vec![hidden_a_error],
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

/// Non-linear squash functions that are acceptable recommendations when
/// replacing IDENTITY. The key is that discovery recognises IDENTITY is
/// suboptimal and recommends something non-linear.
const ACCEPTED_SQUASHES: &[&str] = &[
    "TANH",
    "LOGISTIC",
    "SOFTSIGN",
    "HARD_TANH",
    "CLIPPED",
    "BIPOLAR",
    "BIPOLAR_SIGMOID",
    "MISH",
    "SWISH",
    "GELU",
    "SELU",
    "ELU",
    "RELU",
    "RELU6",
    "LEAKYRELU",
    "SOFTPLUS",
];

/// Helper: check whether any coordinated structural candidate contains a
/// `ChangeSquash` operation for the given neuron recommending one of the
/// accepted squash functions.
fn find_change_squash_candidate(
    candidates: &[neat_ai_discovery::CoordinatedStructuralCandidateJson],
    neuron_uuid: &str,
    accepted_squashes: &[&str],
) -> bool {
    candidates.iter().any(|c| {
        c.operations.iter().any(|op| {
            matches!(op,
                CoordinatedStructuralOpJson::ChangeSquash {
                    neuron_uuid: uuid, squash
                } if uuid == neuron_uuid && accepted_squashes.contains(&squash.as_str())
            )
        }) && c.expected_creature_score_gain > 0.0
    })
}

/// Run `analyze_all` with the crippled creature and return the coordinated
/// structural candidates from the synapse analysis result.
fn run_analysis() -> Vec<neat_ai_discovery::CoordinatedStructuralCandidateJson> {
    let creature = create_crippled_creature();
    let records = generate_change_squash_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeAllInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string(), "hidden-A".to_string()],
        max_synapse_candidates: Some(100),
        max_neuron_candidates: Some(20),
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(false),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    let synapse_result = result
        .synapse
        .as_ref()
        .expect("Synapse analysis should have run");

    synapse_result.coordinated_structural_candidates.clone()
}

/// Issue #930: Discovery should identify that TANH is better than IDENTITY for
/// hidden-A and produce a change-squash candidate via `analyze_all`.
///
/// The crippled creature uses IDENTITY on hidden-A, but the data requires
/// non-linear separation. Multiple detection modules (high-error squash
/// exploration, activation mismatch, activation recommendation, squash
/// weight rescale) may detect this and recommend a non-linear function.
#[test]
fn test_change_squash_discovery_for_suboptimal_activation() {
    skip_without_gpu!();

    let candidates = run_analysis();

    println!("\n=== Issue #930: Change-Squash Discovery ===");
    println!("Coordinated structural candidates: {}", candidates.len());
    for candidate in &candidates {
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

    let found = find_change_squash_candidate(&candidates, "hidden-A", ACCEPTED_SQUASHES);

    assert!(
        found,
        "Discovery should identify that IDENTITY is suboptimal for hidden-A \
         and recommend a non-linear activation function (e.g. TANH). \
         Found {} coordinated structural candidates but none contained a \
         ChangeSquash for hidden-A with a non-linear squash.",
        candidates.len(),
    );

    println!(
        "SUCCESS: Change-squash candidate found recommending non-linear activation for hidden-A"
    );
}

/// Issue #930: Verify the change-squash candidate has correct properties.
///
/// When the change-squash candidate is detected, it must:
/// - Target the correct neuron (hidden-A)
/// - Recommend a different squash from the current one (IDENTITY)
/// - Report positive expected score gain
#[test]
fn test_change_squash_candidate_properties() {
    skip_without_gpu!();

    let candidates = run_analysis();

    // Find all ChangeSquash candidates for hidden-A
    let change_squash_candidates: Vec<_> = candidates
        .iter()
        .filter(|c| {
            c.operations.iter().any(|op| {
                matches!(op,
                    CoordinatedStructuralOpJson::ChangeSquash {
                        neuron_uuid, ..
                    } if neuron_uuid == "hidden-A"
                )
            })
        })
        .collect();

    println!(
        "ChangeSquash candidates for hidden-A: {}",
        change_squash_candidates.len()
    );

    assert!(
        !change_squash_candidates.is_empty(),
        "Should find at least one ChangeSquash candidate for hidden-A"
    );

    for candidate in &change_squash_candidates {
        // Extract the recommended squash
        for op in &candidate.operations {
            if let CoordinatedStructuralOpJson::ChangeSquash {
                neuron_uuid,
                squash,
            } = op
                && neuron_uuid == "hidden-A"
            {
                // Must not recommend the same squash (IDENTITY)
                assert_ne!(
                    squash, "IDENTITY",
                    "Should not recommend the current squash (IDENTITY)"
                );

                println!(
                    "  Recommended: {} (gain={:.6})",
                    squash, candidate.expected_creature_score_gain
                );
            }
        }

        // Must have positive expected score gain
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "ChangeSquash candidate should have positive expected score gain, got {:.6}",
            candidate.expected_creature_score_gain,
        );
    }
}

/// Issue #930: Verify the full pipeline (`analyze_all`) detects the suboptimal
/// activation function end-to-end.
///
/// This test runs both synapse and neuron analysis to ensure the combined
/// pipeline also identifies that hidden-A's IDENTITY squash should be changed
/// to a non-linear function. This mirrors the NEAT-AI integration scenario
/// where the full analysis result is inspected.
#[test]
fn test_analyze_all_finds_change_squash_candidate() {
    skip_without_gpu!();

    let creature = create_crippled_creature();
    let records = generate_change_squash_records();

    let temp_file = NamedTempFile::new().unwrap();
    let parquet_path = temp_file.path().to_str().unwrap();
    write_records_to_parquet(parquet_path, &records).unwrap();

    let input = AnalyzeAllInput {
        parquet_file: parquet_path.to_string(),
        creature,
        focus_neurons: vec!["output-0".to_string(), "hidden-A".to_string()],
        max_synapse_candidates: Some(100),
        max_neuron_candidates: Some(20),
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
    };

    let result = analyze_all(&input).expect("analyze_all should succeed");

    let synapse_result = result
        .synapse
        .as_ref()
        .expect("Synapse analysis should have run");

    println!("\n=== Issue #930: analyze_all Change-Squash Discovery ===");
    println!(
        "Coordinated structural candidates: {}",
        synapse_result.coordinated_structural_candidates.len()
    );
    for candidate in &synapse_result.coordinated_structural_candidates {
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

    let found = find_change_squash_candidate(
        &synapse_result.coordinated_structural_candidates,
        "hidden-A",
        ACCEPTED_SQUASHES,
    );

    assert!(
        found,
        "analyze_all should identify that IDENTITY is suboptimal for hidden-A \
         and recommend a non-linear activation function via a ChangeSquash candidate. \
         Found {} coordinated structural candidates but none matched.",
        synapse_result.coordinated_structural_candidates.len(),
    );

    println!("SUCCESS: analyze_all pipeline detects suboptimal activation for hidden-A");
}
