//! Regression test for Issue #156 (29-Dec-2025): allow output-only focus targets for add-neuron analysis.
//!
//! This test is intentionally scoped to the **opt-in** mode:
//! - Default behaviour remains backwards compatible (hidden focus targets are analysed with impact discounting).
//! - When `NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY=1` is set, hidden focus targets are filtered
//!   and reported as `HiddenNeuronFiltered`.

use neat_ai_discovery::analysis::shared::NeuronNoCandidateReason;
use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};
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

/// Minimal env var guard so tests remain isolated (env var tests are serialised via `#[serial]`).
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            // SAFETY: Serialised via #[serial] — no concurrent env access.
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

#[test]
#[serial]
fn issue_156_hidden_focus_neurons_are_filtered_when_output_only_mode_is_enabled() {
    skip_without_gpu!();

    // Enable output-only focus targets for add-neuron analysis (production experiment).
    let _guard = EnvVarGuard::set("NEAT_AI_DISCOVERY_NEURON_TARGETS_OUTPUT_ONLY", "1");

    // Minimal creature:
    // - 1 hidden neuron (focus target)
    // - 1 output neuron (exists, but not included in focus list)
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![SynapseJson {
            from_uuid: "hidden-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 1.0,
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    };

    // Create a tiny parquet so the record cache can initialise.
    let mut records = Vec::new();
    for obs_index in 0..20u32 {
        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(0.5),
            0.5,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "hidden-0".to_string(),
            Some(0.1),
            0.1,
            vec![0.0],
        ));
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.2),
            0.2,
            vec![0.01],
        ));
    }

    let temp_file = NamedTempFile::new().expect("temp parquet");
    let file_path = temp_file.path().to_str().expect("temp path").to_string();
    write_records_to_parquet(&file_path, &records).expect("write parquet");

    let input = AnalyzeNeuronsInput {
        parquet_file: file_path,
        creature,
        focus_neurons: vec!["hidden-0".to_string()],
        max_candidates: Some(50),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("analysis should succeed");

    assert!(
        result.helpful_neurons.is_empty(),
        "Hidden-only focus list should return no helpful_neurons when output-only mode is enabled"
    );

    assert!(
        result.no_candidate_reasons.iter().any(|s| {
            s.target_uuid == "hidden-0" && s.reason == NeuronNoCandidateReason::HiddenNeuronFiltered
        }),
        "Expected HiddenNeuronFiltered reason for hidden-0 focus neuron"
    );
}
