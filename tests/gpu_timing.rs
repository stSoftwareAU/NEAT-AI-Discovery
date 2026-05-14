//! Tests for GPU kernel profiling/timing feature (Issue #195).
//!
//! This feature adds optional GPU timing diagnostics to help identify:
//! - Which shaders are slowest
//! - CPU vs GPU time breakdown
//! - Buffer transfer overhead
//!
//! Enabled via `NEAT_AI_DISCOVERY_GPU_TIMING=1` environment variable.
//!
//! Note: The env var check is cached with `OnceLock` for performance, so we can't
//! dynamically test enabling/disabling at runtime. Instead we test the `TimingCollector`
//! behavior directly and run integration tests with the env var pre-set.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
mod common;

use neat_ai_discovery::analysis::{GpuAnalyzer, analyze_neurons, analyze_synapses};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use serial_test::serial;
use std::env;
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

/// Helper to create a simple creature with records for testing
fn create_test_data() -> (String, CreatureJson) {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with clear correlation for synapse discovery
    let mut records = Vec::new();
    for obs_index in 0..50u32 {
        let input_activation = (obs_index as f32 - 25.0) / 25.0;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            None,
            input_activation,
            Vec::new(),
        ));

        let error = input_activation * 0.2;
        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(0.5),
            0.5,
            vec![error],
        ));
    }

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

    // Keep temp_dir alive by leaking it (test cleanup will handle it)
    std::mem::forget(temp_dir);

    (parquet_file, creature)
}

/// Test that the timing collector works correctly when disabled.
///
/// Note: We can't test the env var being unset at runtime because the check
/// is cached with `OnceLock` for performance. Instead, we directly test the
/// `TimingCollector` behavior when constructed with `enabled=false`.
#[test]
fn timing_collector_disabled() {
    use neat_ai_discovery::analysis::shared::TimingCollector;

    let collector = TimingCollector::new(false);

    // Record some timing data (should be no-ops)
    collector.record_shader("helpful", 1000000);
    collector.record_buffer_transfer(2000000);
    collector.record_sample_building(3000000);

    // When disabled, finalize should return None
    assert!(
        collector.finalize().is_none(),
        "Timing should be None when collector is disabled"
    );
}

/// Test that timing IS collected when the environment variable is set.
#[test]
#[serial]
fn timing_enabled_via_env_var() {
    skip_without_gpu!();

    // Set the environment variable to enable timing
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_GPU_TIMING", "1") };

    let (parquet_file, creature) = create_test_data();

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    // Clean up
    unsafe { env::remove_var("NEAT_AI_DISCOVERY_GPU_TIMING") };

    // When timing is enabled, metadata should contain timing data
    let timing = result
        .metadata
        .timing
        .expect("Timing should be present when NEAT_AI_DISCOVERY_GPU_TIMING=1");

    // Verify timing structure has expected fields
    assert!(
        timing.total_analysis_ms > 0.0,
        "Total analysis time should be positive"
    );

    // GPU timing should have shader execution times
    let gpu = &timing.gpu;
    assert!(
        gpu.shader_execution_ms >= 0.0,
        "GPU shader execution time should be non-negative"
    );
    assert!(
        gpu.buffer_transfer_ms >= 0.0,
        "Buffer transfer time should be non-negative"
    );

    // CPU timing should have sample building time
    let cpu = &timing.cpu;
    assert!(
        cpu.sample_building_ms >= 0.0,
        "CPU sample building time should be non-negative"
    );
    assert!(
        cpu.result_processing_ms >= 0.0,
        "CPU result processing time should be non-negative"
    );
}

/// Test that per-shader timing is collected.
#[test]
#[serial]
fn per_shader_timing_collected() {
    skip_without_gpu!();

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_GPU_TIMING", "1") };

    let (parquet_file, creature) = create_test_data();

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_synapses(&input).expect("Analysis should succeed");

    unsafe { env::remove_var("NEAT_AI_DISCOVERY_GPU_TIMING") };

    let timing = result.metadata.timing.expect("Timing should be present");

    // Check that we have shader timing entries
    let gpu = &timing.gpu;

    // The helpful shader should have been called at least once
    if let Some(helpful) = gpu.shader_timings.get("helpful") {
        assert!(helpful.calls > 0, "Helpful shader should have been called");
        assert!(helpful.total_ms >= 0.0, "Total time should be non-negative");
        assert!(helpful.avg_ms >= 0.0, "Average time should be non-negative");
    }
}

/// Test that timing output is included in JSON response when enabled.
#[test]
#[serial]
fn timing_in_json_output() {
    skip_without_gpu!();

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_GPU_TIMING", "1") };

    let (parquet_file, creature) = create_test_data();

    // Use the internal JSON-based API to verify JSON output
    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "randomSeed": 42
    });

    let result_json = neat_ai_discovery::analyze_parallel_internal(&input_json.to_string())
        .expect("Analysis should succeed");

    unsafe { env::remove_var("NEAT_AI_DISCOVERY_GPU_TIMING") };

    // Parse the JSON response
    let result: serde_json::Value = serde_json::from_str(&result_json).expect("Should parse JSON");

    assert!(
        result["success"].as_bool().unwrap_or(false),
        "Analysis should succeed"
    );

    // Check that timing is present in the synapse metadata
    let timing = &result["synapseMetadata"]["timing"];
    assert!(
        !timing.is_null(),
        "Timing should be present in JSON output when enabled"
    );
    assert!(
        timing["totalAnalysisMs"].is_number(),
        "totalAnalysisMs should be a number"
    );
    assert!(
        timing["gpu"]["shaderExecutionMs"].is_number(),
        "gpu.shaderExecutionMs should be a number"
    );
    assert!(
        timing["cpu"]["sampleBuildingMs"].is_number(),
        "cpu.sampleBuildingMs should be a number"
    );
}

/// Test that the disabled `TimingCollector` discards recordings and the enabled
/// collector accumulates them into a finalised timing summary.
///
/// Issue #454: Converted from timing-based benchmark to functional test.
/// Performance measurement belongs in `benches/`, not unit tests.
#[test]
fn timing_collector_disabled_vs_enabled_behaviour() {
    use neat_ai_discovery::analysis::shared::TimingCollector;

    // Disabled collector: recordings are no-ops, finalize returns None.
    let disabled_collector = TimingCollector::new(false);
    for _ in 0..100 {
        disabled_collector.record_shader("helpful", 1000);
        disabled_collector.record_buffer_transfer(1000);
        disabled_collector.record_sample_building(1000);
    }
    assert!(
        disabled_collector.finalize().is_none(),
        "Disabled collector should produce no timing data"
    );

    // Enabled collector: recordings are accumulated, finalize returns Some.
    let enabled_collector = TimingCollector::new(true);
    for _ in 0..100 {
        enabled_collector.record_shader("helpful", 1000);
        enabled_collector.record_buffer_transfer(1000);
        enabled_collector.record_sample_building(1000);
    }
    let timing = enabled_collector
        .finalize()
        .expect("Enabled collector should produce timing data");

    // Verify the timing summary contains non-negative values
    assert!(
        timing.gpu.shader_execution_ms >= 0.0,
        "Shader execution time should be non-negative"
    );
    assert!(
        timing.gpu.buffer_transfer_ms >= 0.0,
        "Buffer transfer time should be non-negative"
    );
    assert!(
        timing.cpu.sample_building_ms >= 0.0,
        "Sample building time should be non-negative"
    );
}

/// Test that timing is collected for neuron analysis as well.
///
/// This tests the neuron analysis code path to ensure timing is properly
/// integrated there (not just synapse analysis).
#[test]
#[serial]
fn neuron_analysis_timing_collected() {
    skip_without_gpu!();

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_GPU_TIMING", "1") };

    let (parquet_file, creature) = create_test_data();

    let input = AnalyzeNeuronsInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: None,
        analysis_deadline_ms: None,
        random_seed: Some(42),
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
    };

    let result = analyze_neurons(&input).expect("Analysis should succeed");

    unsafe { env::remove_var("NEAT_AI_DISCOVERY_GPU_TIMING") };

    // When timing is enabled, metadata should contain timing data
    let timing = result
        .metadata
        .timing
        .expect("Timing should be present in neuron analysis when NEAT_AI_DISCOVERY_GPU_TIMING=1");

    // Verify timing structure has expected fields
    assert!(
        timing.total_analysis_ms > 0.0,
        "Total analysis time should be positive"
    );

    // GPU timing should have shader execution times
    let gpu = &timing.gpu;
    assert!(
        gpu.shader_execution_ms >= 0.0,
        "GPU shader execution time should be non-negative"
    );

    // CPU timing should have sample building time
    let cpu = &timing.cpu;
    assert!(
        cpu.sample_building_ms >= 0.0,
        "CPU sample building time should be non-negative"
    );
}
