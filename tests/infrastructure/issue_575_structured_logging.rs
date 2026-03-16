//! Integration tests for structured logging with the tracing crate (Issue #575).
//!
//! These tests verify that the tracing subscriber can be initialised and that
//! tracing macros execute without panicking. They exercise the real
//! `init_tracing()` function from `observability`.

use neat_ai_discovery::observability;

#[test]
fn init_tracing_is_idempotent() {
    // Smoke test: init_tracing sets a global tracing subscriber via try_init().
    // The only observable contract is that repeated calls do not panic — there is
    // no return value or queryable state to assert on.
    observability::init_tracing();
    observability::init_tracing();
    observability::init_tracing();
}

#[test]
fn phase_timer_records_elapsed_time() {
    observability::init_tracing();

    let timer = observability::PhaseTimer::new("test_tracing_phase");
    std::thread::sleep(std::time::Duration::from_millis(5));
    let elapsed = timer.elapsed_ms();
    drop(timer);

    assert!(
        elapsed >= 1,
        "PhaseTimer should record non-zero elapsed time after sleep, got {elapsed}ms"
    );
}

#[test]
fn gpu_metrics_tracks_recorded_values() {
    observability::init_tracing();

    let metrics = observability::GpuMetrics::new();
    metrics.record_batch(42);
    metrics.record_gpu_busy_us(1000);
    metrics.record_queue_wait_us(200);

    assert_eq!(metrics.batch_count(), 1, "should record one batch");
    assert_eq!(
        metrics.total_samples_processed(),
        42,
        "should record 42 samples"
    );
    assert_eq!(
        metrics.total_gpu_busy_us(),
        1000,
        "should record 1000µs GPU busy time"
    );
    assert_eq!(
        metrics.total_queue_wait_us(),
        200,
        "should record 200µs queue wait time"
    );

    // Utilisation = 1000 / (1000 + 200) ≈ 83.3%
    let utilisation = metrics.utilisation_percent();
    assert!(
        (utilisation - 83.3).abs() < 1.0,
        "utilisation should be ~83.3%, got {utilisation:.1}%"
    );

    // report() emits tracing output — verify it does not panic
    metrics.report();
}

#[test]
fn profile_data_records_phases_and_produces_valid_json() {
    observability::init_tracing();

    let mut profile = observability::ProfileData::new();
    profile.record_phase("test_phase", 100);
    profile.set_gpu_batch_count(5);

    let json = profile.to_json();
    assert_eq!(
        json["timing"]["phases"]["test_phase"], 100,
        "recorded phase should appear in JSON output"
    );
    assert_eq!(
        json["gpu"]["batchCount"], 5,
        "GPU batch count should appear in JSON output"
    );

    // report() emits tracing output — verify it does not panic
    profile.report();
}

#[test]
fn log_version_once_initialises_tracing() {
    // log_version_once is pub(crate), so we exercise it through the public API.
    // get_library_version_internal calls log_version_once internally.
    let result = neat_ai_discovery::get_library_version_internal();
    assert!(
        result.is_ok(),
        "get_library_version_internal should succeed"
    );
    let json_str = result.unwrap();
    let json: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(json["success"], true);
    assert!(json["version"].is_string());
}
