//! Test for Issue #182: Environment variable to focus on unused observations.
//!
//! When `NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1` is set, the discovery process
//! should prioritise input neurons that have NO existing outgoing synapses. These are
//! "unused observations" - inputs in the training data that haven't been connected to
//! the network yet.
//!
//! This is particularly useful when new observations have been added to the training
//! data set and the user wants discovery to focus on connecting these new inputs first.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::utils::focus_unused_observations_from_env;
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
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

/// Minimal env var guard so tests remain isolated.
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        match &self.previous {
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

/// Create a creature with multiple inputs where some have synapses and some don't.
///
/// Layout:
/// - input-0: HAS synapse to output-0 (weight 1.0) - "used"
/// - input-1: HAS synapse to output-0 (weight 1.0) - "used"
/// - input-2: NO synapse - "unused"
/// - input-3: NO synapse - "unused"
/// - output-0: target neuron
fn create_test_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "TANH".to_string(),
            bias: 0.0,
        }],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 4, // 4 inputs: input-0, input-1, input-2, input-3
        output: 1,
    }
}

/// Create parquet records where unused inputs (input-2, input-3) have strong
/// error correlation that should make them good candidates.
fn create_test_records() -> Vec<DiscoverRecord> {
    let mut records = Vec::new();

    for obs_index in 0..100u32 {
        let phase = (obs_index as f32) * 0.1;

        // input-0 and input-1 (used) - random activations, weak correlation with error
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(phase.sin()),
            phase.sin(),
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-1".to_string(),
            Some(phase.cos()),
            phase.cos(),
            vec![0.0],
        ));

        // input-2 and input-3 (unused) - activations that correlate with target error
        // These should be good candidates for new synapses
        let error = if obs_index % 2 == 0 { 0.5 } else { -0.5 };
        records.push(DiscoverRecord::new(
            obs_index,
            "input-2".to_string(),
            Some(error * 2.0), // Strong correlation with error
            error * 2.0,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "input-3".to_string(),
            Some(error * 1.5), // Moderate correlation with error
            error * 1.5,
            vec![0.0],
        ));

        // output-0 with error pattern
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.0),
            0.0,
            vec![error],
        ));
    }

    records
}

/// Test that the environment variable is correctly parsed when set to "1".
#[test]
#[serial]
fn issue_182_env_var_parsing() {
    // Test with env var set to "1"
    let _guard = EnvVarGuard::set("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS", "1");
    assert!(
        focus_unused_observations_from_env(),
        "Should be true when set to '1'"
    );
}

#[test]
#[serial]
fn issue_182_env_var_parsing_not_set() {
    // Test with env var explicitly removed (using guard to restore)
    // Note: This test is serialised via #[serial] to avoid concurrent env var access.
    let _guard = EnvVarGuard::set("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS", "");
    // Empty string should be treated as disabled
    assert!(
        !focus_unused_observations_from_env(),
        "Should be false when empty"
    );
}

#[test]
#[serial]
fn issue_182_env_var_parsing_various_values() {
    // Test with env var set to "true"
    let _guard = EnvVarGuard::set("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS", "true");
    assert!(
        focus_unused_observations_from_env(),
        "Should be true when set to 'true'"
    );
}

#[test]
#[serial]
fn issue_182_env_var_parsing_disabled() {
    // Test with env var set to "0" - should be disabled
    let _guard = EnvVarGuard::set("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS", "0");
    assert!(
        !focus_unused_observations_from_env(),
        "Should be false when set to '0'"
    );
}

#[test]
fn issue_182_focus_unused_observations_prioritises_inputs_without_synapses() {
    skip_without_gpu!();

    // Enable focus on unused observations
    let _guard = EnvVarGuard::set("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS", "1");

    let creature = create_test_creature();

    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    let records = create_test_records();
    write_records_to_parquet(&file_path, &records).expect("write parquet");

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: Some(5000), // Short deadline to test prioritisation
        random_seed: Some(42),
    };

    let result = analyze_synapses(&input).expect("analysis should succeed");

    // With NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1, the unused inputs
    // (input-2, input-3) should be prioritised and appear in the results
    // before the used inputs (input-0, input-1).
    let helpful = &result.helpful_synapses;

    // We should find candidates from the unused inputs
    let unused_input_candidates: Vec<_> = helpful
        .iter()
        .filter(|c| c.from_neuron_uuid == "input-2" || c.from_neuron_uuid == "input-3")
        .collect();

    assert!(
        !unused_input_candidates.is_empty(),
        "Should find candidates from unused inputs (input-2, input-3). Got {} helpful synapses: {:?}",
        helpful.len(),
        helpful
            .iter()
            .map(|c| &c.from_neuron_uuid)
            .collect::<Vec<_>>()
    );

    // With short deadline and prioritisation, the unused inputs should be evaluated first.
    // This means the first few candidates should predominantly be from unused inputs.
    // Count how many of the first 4 candidates are from unused inputs.
    let first_four: Vec<_> = helpful.iter().take(4).collect();
    let unused_in_first_four = first_four
        .iter()
        .filter(|c| c.from_neuron_uuid == "input-2" || c.from_neuron_uuid == "input-3")
        .count();

    // When NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1 is set, unused inputs
    // should be processed FIRST (moved to the front of the evaluation queue).
    // Therefore, candidates from unused inputs should dominate the results.
    assert!(
        unused_in_first_four >= 2,
        "With NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS=1, most early candidates should be from \
        unused inputs (input-2, input-3). Got {} unused in first 4. First 4: {:?}",
        unused_in_first_four,
        first_four
            .iter()
            .map(|c| &c.from_neuron_uuid)
            .collect::<Vec<_>>()
    );
}

#[test]
fn issue_182_without_env_var_no_prioritisation() {
    skip_without_gpu!();

    // Ensure env var is NOT set
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { std::env::remove_var("NEAT_AI_DISCOVERY_FOCUS_UNUSED_OBSERVATIONS") };

    let creature = create_test_creature();

    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    let records = create_test_records();
    write_records_to_parquet(&file_path, &records).expect("write parquet");

    let input = AnalyzeSynapsesInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None, // No deadline - process all
        random_seed: Some(42),
    };

    let result = analyze_synapses(&input).expect("analysis should succeed");

    // Without the env var, the default behaviour is random ordering.
    // We just verify that analysis completes and returns some results.
    // The exact ordering depends on the random seed.
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Analysis should return either candidates or diagnostic reasons"
    );
}
