//! Integration tests for structured logging with the tracing crate (Issue #575).
//!
//! These tests verify that the tracing subscriber can be initialised and that
//! tracing macros execute without panicking. They exercise the real
//! `init_tracing()` function from `observability`.

use neat_ai_discovery::observability;

#[test]
fn init_tracing_is_idempotent() {
    // Calling init_tracing multiple times must not panic.
    observability::init_tracing();
    observability::init_tracing();
    observability::init_tracing();
}

#[test]
fn phase_timer_uses_tracing_without_panic() {
    observability::init_tracing();

    // Create and drop a PhaseTimer — should emit a tracing event (not panic).
    let timer = observability::PhaseTimer::new("test_tracing_phase");
    std::thread::sleep(std::time::Duration::from_millis(1));
    drop(timer);
}

#[test]
fn gpu_metrics_report_uses_tracing_without_panic() {
    observability::init_tracing();

    let metrics = observability::GpuMetrics::new();
    metrics.record_batch(42);
    metrics.record_gpu_busy_us(1000);
    metrics.record_queue_wait_us(200);

    // report() now uses tracing::info! instead of eprintln! — must not panic.
    metrics.report();
}

#[test]
fn profile_data_report_uses_tracing_without_panic() {
    observability::init_tracing();

    let mut profile = observability::ProfileData::new();
    profile.record_phase("test_phase", 100);
    profile.set_gpu_batch_count(5);

    // report() now uses tracing::info! instead of eprintln! — must not panic.
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
