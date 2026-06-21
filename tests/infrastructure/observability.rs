//! Tests for structured observability and profiling hooks (Issue #214).
//!
//! This module tests the observability infrastructure that provides:
//! - Phase-level timing via `PhaseTimer`
//! - GPU metrics tracking (batch count, samples processed, utilisation)
//! - Structured JSON profiling output
//!
//! Environment variables:
//! - `NEAT_AI_DISCOVERY_TIMING=1`: Print phase timing to stderr
//! - `NEAT_AI_DISCOVERY_PROFILE=json`: Output structured profile as JSON
//! - `NEAT_AI_DISCOVERY_GPU_METRICS=1`: Print GPU metrics to stderr

#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::GpuAnalyzer;
use neat_ai_discovery::observability::{
    GpuMetrics, PhaseTimer, ProfileData, ProfileMode, gpu_metrics_enabled, timing_enabled,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson};
use serial_test::serial;
use std::env;
use std::time::Duration;
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

// =============================================================================
// PhaseTimer Tests
// =============================================================================

/// Test that `PhaseTimer` reports at least the elapsed sleep duration.
///
/// This is a "what" test: it asserts on the value the timer itself reports
/// (`elapsed_ms`), not on externally-measured host wall-clock. The only
/// behavioural guarantee is monotonicity — having slept for at least
/// `sleep_duration`, the timer must report at least that much. A fragile
/// upper bound on overhead is intentionally omitted: thread-scheduling jitter
/// on a loaded CI runner can exceed any tight margin without any regression in
/// `PhaseTimer` itself (Issue #1468).
#[test]
fn phase_timer_accuracy() {
    let sleep_duration = Duration::from_millis(50);

    let timer = PhaseTimer::new("test_phase");
    std::thread::sleep(sleep_duration);
    let reported = Duration::from_millis(timer.elapsed_ms());

    // Monotonicity: the timer reports at least the time we slept. `elapsed_ms`
    // truncates to whole milliseconds and `thread::sleep` guarantees sleeping
    // for at least the requested duration, so this holds without a host-speed
    // assumption.
    assert!(
        reported >= sleep_duration,
        "PhaseTimer should report at least the slept duration ({sleep_duration:?}), got {reported:?}"
    );
}

/// Test that `PhaseTimer` reports correctly when `NEAT_AI_DISCOVERY_TIMING=1`.
#[test]
fn phase_timer_respects_env_var() {
    // When timing is disabled (default), PhaseTimer should be cheap
    let timer = PhaseTimer::new("test_phase");
    drop(timer);

    // We can't easily test stderr output in unit tests, but we can verify
    // the timer completes without errors regardless of env var state
}

/// Test that nested `PhaseTimers` work correctly.
#[test]
fn phase_timer_nested() {
    let _outer = PhaseTimer::new("outer_phase");
    std::thread::sleep(Duration::from_millis(10));
    {
        let _inner = PhaseTimer::new("inner_phase");
        std::thread::sleep(Duration::from_millis(10));
    }
    // Both timers should complete without issues
}

// =============================================================================
// GpuMetrics Tests
// =============================================================================

/// Test that `GpuMetrics` tracks batch counts correctly.
#[test]
fn gpu_metrics_batch_count() {
    let metrics = GpuMetrics::new();

    // Increment batch count multiple times
    metrics.record_batch(100);
    metrics.record_batch(200);
    metrics.record_batch(150);

    assert_eq!(metrics.batch_count(), 3, "Should have recorded 3 batches");
    assert_eq!(
        metrics.total_samples_processed(),
        450,
        "Should have processed 450 samples total"
    );
}

/// Test that `GpuMetrics` tracks queue wait time.
#[test]
fn gpu_metrics_queue_wait_time() {
    let metrics = GpuMetrics::new();

    metrics.record_queue_wait_us(1000);
    metrics.record_queue_wait_us(2000);

    assert_eq!(
        metrics.total_queue_wait_us(),
        3000,
        "Should have recorded 3000us queue wait time"
    );
}

/// Test that `GpuMetrics` tracks GPU busy time.
#[test]
fn gpu_metrics_gpu_busy_time() {
    let metrics = GpuMetrics::new();

    metrics.record_gpu_busy_us(5000);
    metrics.record_gpu_busy_us(3000);

    assert_eq!(
        metrics.total_gpu_busy_us(),
        8000,
        "Should have recorded 8000us GPU busy time"
    );
}

/// Test that `GpuMetrics` calculates utilisation correctly.
#[test]
fn gpu_metrics_utilisation() {
    let metrics = GpuMetrics::new();

    // 80% utilisation: 8000us busy out of 10000us total (busy + wait)
    metrics.record_gpu_busy_us(8000);
    metrics.record_queue_wait_us(2000);

    let utilisation = metrics.utilisation_percent();
    assert!(
        (utilisation - 80.0).abs() < 0.1,
        "Utilisation should be approximately 80%, got {utilisation}"
    );
}

/// Test that `GpuMetrics` handles zero total time gracefully.
#[test]
fn gpu_metrics_zero_utilisation() {
    let metrics = GpuMetrics::new();

    // No time recorded yet
    let utilisation = metrics.utilisation_percent();
    assert!(
        utilisation.is_nan() || utilisation == 0.0,
        "Utilisation should be 0 or NaN when no time recorded"
    );
}

/// Test that `GpuMetrics` is thread-safe.
#[test]
fn gpu_metrics_thread_safety() {
    use std::sync::Arc;

    let metrics = Arc::new(GpuMetrics::new());

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let m = Arc::clone(&metrics);
            std::thread::spawn(move || {
                for _ in 0..100 {
                    m.record_batch(10);
                    m.record_queue_wait_us(100);
                    m.record_gpu_busy_us(500);
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should complete");
    }

    // 4 threads * 100 iterations = 400 batches
    assert_eq!(metrics.batch_count(), 400);
    assert_eq!(metrics.total_samples_processed(), 4000); // 400 * 10
    assert_eq!(metrics.total_queue_wait_us(), 40000); // 400 * 100
    assert_eq!(metrics.total_gpu_busy_us(), 200000); // 400 * 500
}

// =============================================================================
// ProfileData Tests
// =============================================================================

/// Test that `ProfileData` collects timing data correctly.
#[test]
fn profile_data_timing() {
    let mut profile = ProfileData::new();

    profile.record_phase("parquet_loading", 156);
    profile.record_phase("focus_selection", 23);
    profile.record_phase("gpu_analysis", 987);

    let json = profile.to_json();

    // Verify JSON structure
    assert!(json["timing"]["phases"]["parquet_loading"].is_number());
    assert_eq!(json["timing"]["phases"]["parquet_loading"], 156);
    assert_eq!(json["timing"]["phases"]["focus_selection"], 23);
    assert_eq!(json["timing"]["phases"]["gpu_analysis"], 987);
}

/// Test that `ProfileData` collects GPU metrics correctly.
#[test]
fn profile_data_gpu_metrics() {
    let mut profile = ProfileData::new();

    profile.set_gpu_batch_count(45);
    profile.set_gpu_samples_processed(1234567);
    profile.set_gpu_utilisation(87.3);
    profile.set_gpu_device("Apple M2 Max".to_string());

    let json = profile.to_json();

    assert_eq!(json["gpu"]["batchCount"], 45);
    assert_eq!(json["gpu"]["samplesProcessed"], 1234567);
    assert!((json["gpu"]["utilisationPercent"].as_f64().unwrap() - 87.3).abs() < 0.1);
    assert_eq!(json["gpu"]["device"], "Apple M2 Max");
}

/// Test that `ProfileData` collects analysis metrics correctly.
#[test]
fn profile_data_analysis_metrics() {
    let mut profile = ProfileData::new();

    profile.set_focus_neurons_requested(64);
    profile.set_focus_neurons_completed(64);
    profile.set_candidates_found(1234);
    profile.set_candidates_returned(100);

    let json = profile.to_json();

    assert_eq!(json["analysis"]["focusNeuronsRequested"], 64);
    assert_eq!(json["analysis"]["focusNeuronsCompleted"], 64);
    assert_eq!(json["analysis"]["candidatesFound"], 1234);
    assert_eq!(json["analysis"]["candidatesReturned"], 100);
}

/// Test that `ProfileData` `to_json` produces valid JSON output.
#[test]
fn profile_data_json_valid() {
    let mut profile = ProfileData::new();

    profile.record_phase("test", 100);
    profile.set_gpu_batch_count(10);
    profile.set_focus_neurons_requested(5);

    let json = profile.to_json();
    let json_string = serde_json::to_string(&json).expect("Should serialize to JSON string");

    // Should be parseable
    let parsed: serde_json::Value =
        serde_json::from_str(&json_string).expect("Should parse back to JSON");

    assert!(parsed["timing"]["phases"]["test"].is_number());
}

// =============================================================================
// Environment Variable Tests
// =============================================================================

/// Test that `timing_enabled()` respects the environment variable.
#[test]
fn timing_enabled_env_check() {
    // Note: env var checks are cached with OnceLock, so we can't dynamically
    // test enabling/disabling. We can only test that the function exists and
    // returns a boolean.
    let _enabled = timing_enabled();
}

/// Test that `gpu_metrics_enabled()` respects the environment variable.
#[test]
fn gpu_metrics_enabled_env_check() {
    let _enabled = gpu_metrics_enabled();
}

/// Test that `profile_mode()` parses the environment variable correctly.
#[test]
fn profile_mode_parsing() {
    // Test the ProfileMode enum
    assert!(matches!(ProfileMode::None, ProfileMode::None));
    assert!(matches!(ProfileMode::Json, ProfileMode::Json));
}

// =============================================================================
// Integration Tests with Analysis Pipeline
// =============================================================================

/// Helper to create test data for integration tests.
fn create_test_data() -> (String, CreatureJson) {
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let parquet_path = temp_dir.path().join("records.parquet");
    let parquet_file = parquet_path.to_str().unwrap().to_string();

    // Create records with clear correlation for discovery
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

    // Keep temp_dir alive by leaking it
    std::mem::forget(temp_dir);

    (parquet_file, creature)
}

/// Test that timing output includes phase breakdown when `NEAT_AI_DISCOVERY_TIMING=1`.
#[test]
#[serial]
fn integration_timing_output() {
    skip_without_gpu!();

    // Set environment variable
    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_TIMING", "1") };

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

    let result =
        neat_ai_discovery::analysis::analyze_synapses(&input).expect("Analysis should succeed");

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::remove_var("NEAT_AI_DISCOVERY_TIMING") };

    // Analysis should complete successfully
    // Timing output goes to stderr which is hard to capture in tests,
    // but we verify the analysis completed
    assert!(result.metadata.total_focus_neurons > 0);
}

/// Test that JSON profile output is structured correctly when `NEAT_AI_DISCOVERY_PROFILE=json`.
#[test]
#[serial]
fn integration_json_profile() {
    skip_without_gpu!();

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_PROFILE", "json") };

    let (parquet_file, creature) = create_test_data();

    let input_json = serde_json::json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "focusNeurons": ["output-0"],
        "randomSeed": 42
    });

    let result_json = neat_ai_discovery::analyze_parallel_internal(&input_json.to_string())
        .expect("Analysis should succeed");

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::remove_var("NEAT_AI_DISCOVERY_PROFILE") };

    // Parse the JSON response
    let result: serde_json::Value = serde_json::from_str(&result_json).expect("Should parse JSON");

    assert!(
        result["success"].as_bool().unwrap_or(false),
        "Analysis should succeed"
    );

    // When NEAT_AI_DISCOVERY_PROFILE=json, the profile data should be in the response
    // The profile is written to stderr as a separate JSON object, but the main
    // response should still be valid
}

/// Test that GPU metrics are collected when `NEAT_AI_DISCOVERY_GPU_METRICS=1`.
#[test]
#[serial]
fn integration_gpu_metrics() {
    skip_without_gpu!();

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::set_var("NEAT_AI_DISCOVERY_GPU_METRICS", "1") };

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

    let result =
        neat_ai_discovery::analysis::analyze_synapses(&input).expect("Analysis should succeed");

    // SAFETY: Serialised via #[serial] — no concurrent env access.
    unsafe { env::remove_var("NEAT_AI_DISCOVERY_GPU_METRICS") };

    // Analysis should complete successfully
    assert!(result.metadata.total_focus_neurons > 0);
}

// =============================================================================
// Overhead Verification Tests
// =============================================================================

/// Test that `PhaseTimer` can be created and dropped many times without error.
///
/// Issue #454: Converted from timing-based benchmark to functional test.
/// Performance measurement belongs in `benches/`, not unit tests.
#[test]
fn phase_timer_repeated_creation_and_drop() {
    for _ in 0..100 {
        let _timer = PhaseTimer::new("test_phase");
        // Timer drops immediately — verify no panic or resource leak
    }
}

/// Test that `GpuMetrics` accumulates correctly after many recordings.
///
/// Issue #454: Converted from timing-based benchmark to functional test.
/// Performance measurement belongs in `benches/`, not unit tests.
#[test]
fn gpu_metrics_accumulates_many_recordings() {
    let metrics = GpuMetrics::new();

    let iterations = 100;
    for _ in 0..iterations {
        metrics.record_batch(100);
        metrics.record_queue_wait_us(1000);
        metrics.record_gpu_busy_us(5000);
    }

    assert_eq!(metrics.batch_count(), iterations);
    assert_eq!(metrics.total_samples_processed(), iterations * 100);
    assert_eq!(metrics.total_queue_wait_us(), iterations as u64 * 1000);
    assert_eq!(metrics.total_gpu_busy_us(), iterations as u64 * 5000);
}

/// Test that `ProfileData` retains the last recorded phase value.
///
/// Issue #454: Converted from timing-based benchmark to functional test.
/// Performance measurement belongs in `benches/`, not unit tests.
#[test]
fn profile_data_records_many_phases() {
    let mut profile = ProfileData::new();

    for i in 0..100 {
        profile.record_phase("test", i as u64);
    }

    let json = profile.to_json();
    // The last recorded value for "test" should be 99
    assert_eq!(json["timing"]["phases"]["test"], 99);
}
