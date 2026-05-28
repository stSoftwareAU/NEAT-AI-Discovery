//! Integration tests for Issue #1132: Adapt discovery strategy when recent
//! creature-level success rate is near zero.
//!
//! Verifies that:
//!
//! 1. A `discovery_outcome_log` full of failures causes `analyze_all` to
//!    report `discoveryMode = "conservative"` and the configured rolling
//!    success rate in the response metadata.
//! 2. An empty / successful outcome log leaves the metadata in the default
//!    `"normal"` mode.
//! 3. Conservative mode biases the returned `module_outcome_tracker` in
//!    favour of low-risk modules relative to normal mode (integration check
//!    of the weight shift performed inside orchestration).

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS, DEFAULT_LOW_SUCCESS_RATE_THRESHOLD, DiscoveryMode,
    DiscoveryOutcomeLog, biased_tracker_for_conservative_mode, decide_mode,
};
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};

macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

fn build_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-0".to_string(),
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
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

fn build_records(creature: &CreatureJson, count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..count as u32 {
        let t = obs as f32 / count as f32;
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (act, Some(act), vec![0.05 * (t * 3.0).cos()])
                }
                "output" => {
                    let error = 0.1 * (t * std::f32::consts::PI).cos();
                    (0.5 + 0.2 * t, Some(0.5 + 0.2 * t), vec![error])
                }
                _ => continue,
            };
            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }
    records
}

fn make_input(parquet_file: String, outcome_log: Option<DiscoveryOutcomeLog>) -> AnalyzeAllInput {
    let creature = build_creature();
    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();

    AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
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
        discovery_outcome_log: outcome_log,
        cost_name: None,
    }
}

/// A rolling log of ten consecutive failures causes `analyze_all` to emit
/// `conservative` in the response metadata, with a rolling success rate of 0.
#[test]
fn ten_failures_triggers_conservative_mode_in_metadata() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let outcome_log = DiscoveryOutcomeLog::from_outcomes(vec![false; 10]);
    let input = make_input(parquet_file, Some(outcome_log));

    let result = analyze_all(&input).expect("analyze_all should succeed");
    let synapse = result.synapse.expect("synapse analysis should run");
    let neuron = result.neuron.expect("neuron analysis should run");

    assert_eq!(
        synapse.metadata.discovery_mode,
        DiscoveryMode::Conservative,
        "synapse metadata should report conservative mode"
    );
    assert!(
        (synapse.metadata.rolling_success_rate).abs() < 1e-6,
        "rolling success rate should be 0.0, got {}",
        synapse.metadata.rolling_success_rate
    );

    assert_eq!(
        neuron.metadata.discovery_mode,
        DiscoveryMode::Conservative,
        "neuron metadata should report conservative mode"
    );
    assert!(
        (neuron.metadata.rolling_success_rate).abs() < 1e-6,
        "rolling success rate should be 0.0, got {}",
        neuron.metadata.rolling_success_rate
    );
}

/// Absence of an outcome log leaves the metadata in the default normal mode
/// with a neutral rolling rate of 1.0.
#[test]
fn no_outcome_log_keeps_metadata_in_normal_mode() {
    skip_without_gpu!();

    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("test.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet");

    let input = make_input(parquet_file, None);
    let result = analyze_all(&input).expect("analyze_all should succeed");
    let synapse = result.synapse.expect("synapse analysis should run");
    let neuron = result.neuron.expect("neuron analysis should run");

    assert_eq!(synapse.metadata.discovery_mode, DiscoveryMode::Normal);
    assert!(
        (synapse.metadata.rolling_success_rate - 1.0).abs() < 1e-6,
        "rolling success rate should default to 1.0"
    );
    assert_eq!(neuron.metadata.discovery_mode, DiscoveryMode::Normal);
    assert!((neuron.metadata.rolling_success_rate - 1.0).abs() < 1e-6);
}

/// The decision function itself respects the rolling window and the cooldown
/// — this guards the transition semantics without needing a GPU.
#[test]
fn mode_decision_transitions_at_documented_boundaries() {
    // A healthy log stays in Normal mode.
    let healthy = DiscoveryOutcomeLog::from_outcomes(vec![true; 10]);
    assert_eq!(
        decide_mode(
            &healthy,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS
        ),
        DiscoveryMode::Normal
    );

    // 9 failures + 1 success over last 10 passes = 0.1 < 0.2 threshold.
    let mut outcomes = vec![false; 9];
    outcomes.push(true);
    let struggling = DiscoveryOutcomeLog::from_outcomes(outcomes);
    assert_eq!(
        decide_mode(
            &struggling,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS
        ),
        DiscoveryMode::Conservative
    );

    // Consecutive failure streak exceeds max_epochs: revert to Normal.
    let long_drought = DiscoveryOutcomeLog::from_outcomes(vec![false; 30]);
    assert_eq!(
        decide_mode(
            &long_drought,
            DEFAULT_LOW_SUCCESS_RATE_THRESHOLD,
            DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS
        ),
        DiscoveryMode::Normal
    );
}

/// Biasing a tracker with identical low- and high-risk histories must produce
/// a tracker where low-risk modules score strictly above high-risk modules.
/// This is an integration-level cross-check of the weight-shift helper.
#[test]
fn conservative_bias_raises_low_risk_above_high_risk() {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..10 {
        tracker.record("bias-drift", true);
        tracker.record("bias-drift", false);
        tracker.record("add-neurons", true);
        tracker.record("add-neurons", false);
    }

    let biased = biased_tracker_for_conservative_mode(&tracker);
    let low = biased.stats("bias-drift").success_rate();
    let high = biased.stats("add-neurons").success_rate();
    assert!(
        low > high,
        "conservative bias should raise low-risk modules above high-risk: low={low}, high={high}"
    );
}
