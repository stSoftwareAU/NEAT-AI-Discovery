//! Issue #953: Tests for deadline-aware GPU work queue.
//!
//! Verifies that the GPU work queue propagates analysis deadlines to
//! `GpuEvaluator` trait calls, preventing liveness stalls when the GPU
//! is slow and rayon threads are blocked on 5-minute default timeouts.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use tempfile::tempdir;

/// Skip test if no GPU available
macro_rules! skip_without_gpu {
    () => {
        if !GpuAnalyzer::gpu_is_available() {
            eprintln!("Skipping test: no GPU available");
            return;
        }
    };
}

/// Helper to create a minimal test parquet file with correlated data.
fn create_test_parquet(parquet_file: &str) {
    let mut records = Vec::new();
    for obs_index in 0..30u32 {
        let input_activation = (obs_index as f32 - 15.0) / 15.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        let error = if input_activation > 0.0 { 0.2 } else { -0.2 };
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

    write_records_to_parquet(parquet_file, &records).expect("Failed to write parquet");
}

fn create_test_creature() -> CreatureJson {
    CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::new(),
    }
}

/// Issue #953: Neuron analysis with a deadline completes without hanging.
///
/// Previously, `GpuEvaluator` trait calls used `None` for deadline, giving each
/// GPU operation a 5-minute timeout. This test verifies that deadlines are
/// propagated correctly by running analysis with a generous deadline and
/// confirming it completes without stalling.
#[test]
fn neuron_analysis_with_deadline_completes() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    create_test_parquet(&parquet_file);

    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature: create_test_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        // Issue #953: Set a 2-minute deadline — the queue should use this for
        // adaptive timeouts instead of the 5-minute maximum.
        analysis_deadline_ms: Some(120_000),
        random_seed: Some(42),
        temperature: 1.0,
    };

    let result = analyze_neurons(&input).expect("Neuron analysis with deadline should succeed");
    assert!(
        !result.helpful_neurons.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should produce candidates or diagnostics"
    );
}

/// Issue #953: Synapse analysis with a deadline completes without hanging.
#[test]
fn synapse_analysis_with_deadline_completes() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    create_test_parquet(&parquet_file);

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature: create_test_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        // Issue #953: Set a 2-minute deadline
        analysis_deadline_ms: Some(120_000),
        random_seed: Some(42),
        temperature: 1.0,
    };

    let result = analyze_synapses(&input).expect("Synapse analysis with deadline should succeed");
    assert!(
        !result.helpful_synapses.is_empty() || !result.no_candidate_reasons.is_empty(),
        "Should produce candidates or diagnostics"
    );
}

/// Issue #953: Analysis with an expired deadline returns promptly rather than
/// hanging for 5 minutes on GPU timeouts.
///
/// When the analysis deadline has already passed, the GPU queue should use the
/// minimum timeout (60s) rather than the maximum (5 min). The analysis should
/// detect the expired deadline at the per-target checkpoint and return quickly
/// with partial or empty results instead of stalling.
#[test]
fn analysis_with_expired_deadline_returns_promptly() {
    skip_without_gpu!();

    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    create_test_parquet(&parquet_file);

    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature: create_test_creature(),
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        // Set a deadline of 1ms — effectively already expired by the time
        // analysis begins. The analysis should return quickly (not hang).
        analysis_deadline_ms: Some(1),
        random_seed: Some(42),
        temperature: 1.0,
    };

    let start = std::time::Instant::now();
    // Analysis should either succeed with partial/empty results or return an error.
    // The key assertion is that it returns promptly.
    let _result = analyze_neurons(&input);
    let elapsed = start.elapsed();

    // Should return well within 60 seconds (the minimum GPU timeout).
    // In practice this should be nearly instant since the deadline check
    // fires before any GPU work is submitted.
    assert!(
        elapsed.as_secs() < 60,
        "Analysis with expired deadline took {elapsed:?} — should return promptly"
    );
}
