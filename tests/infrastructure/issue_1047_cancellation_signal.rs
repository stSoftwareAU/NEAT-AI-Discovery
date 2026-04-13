//! Tests for the cancellation signal mechanism (Issue #1047).
//!
//! Verifies that:
//! - The `cancel_analysis()` FFI export sets the global cancellation flag
//! - `reset_cancellation()` clears the flag
//! - `deadline_passed()` returns `true` when cancelled
//! - The `DiscoveryErrorKind::Cancelled` variant is correctly classified
//! - The `cancelled` field appears in `AnalyzeParallelOutput` JSON

use neat_ai_discovery::DiscoveryErrorKind;
use neat_ai_discovery::cancellation;
use serial_test::serial;

// ============================================================================
// Cancellation flag unit behaviour
// ============================================================================

#[test]
#[serial]
fn cancellation_flag_starts_false() {
    cancellation::reset_cancellation();
    assert!(
        !cancellation::is_cancelled(),
        "flag should be false after reset"
    );
}

#[test]
#[serial]
fn request_cancellation_sets_flag() {
    cancellation::reset_cancellation();
    cancellation::request_cancellation();
    assert!(
        cancellation::is_cancelled(),
        "flag should be true after request_cancellation"
    );
    cancellation::reset_cancellation();
}

#[test]
#[serial]
fn reset_clears_cancellation() {
    cancellation::request_cancellation();
    assert!(cancellation::is_cancelled());
    cancellation::reset_cancellation();
    assert!(
        !cancellation::is_cancelled(),
        "flag should be false after reset"
    );
}

// ============================================================================
// deadline_passed integration
// ============================================================================

#[test]
#[serial]
fn deadline_passed_returns_true_when_cancelled() {
    cancellation::reset_cancellation();
    // With no deadline, deadline_passed normally returns false.
    let no_deadline: Option<std::time::SystemTime> = None;
    assert!(
        !neat_ai_discovery::analysis::utils::deadline_passed(&no_deadline),
        "should be false before cancellation"
    );

    // After requesting cancellation, deadline_passed should return true
    // even without a real deadline.
    cancellation::request_cancellation();
    assert!(
        neat_ai_discovery::analysis::utils::deadline_passed(&no_deadline),
        "should be true after cancellation request"
    );
    cancellation::reset_cancellation();
}

#[test]
#[serial]
fn deadline_passed_returns_true_when_cancelled_with_future_deadline() {
    cancellation::reset_cancellation();
    let future_deadline = Some(std::time::SystemTime::now() + std::time::Duration::from_secs(3600));

    assert!(
        !neat_ai_discovery::analysis::utils::deadline_passed(&future_deadline),
        "should be false before cancellation"
    );

    cancellation::request_cancellation();
    assert!(
        neat_ai_discovery::analysis::utils::deadline_passed(&future_deadline),
        "should be true after cancellation even with future deadline"
    );
    cancellation::reset_cancellation();
}

// ============================================================================
// Error classification
// ============================================================================

#[test]
fn cancelled_error_kind_is_not_retryable() {
    assert!(
        !DiscoveryErrorKind::Cancelled.is_retryable(),
        "cancelled is not a retryable error"
    );
}

#[test]
fn cancelled_error_kind_is_cancelled() {
    assert!(DiscoveryErrorKind::Cancelled.is_cancelled());
    assert!(!DiscoveryErrorKind::Timeout.is_cancelled());
    assert!(!DiscoveryErrorKind::Unknown.is_cancelled());
}

#[test]
fn cancelled_error_kind_serialises_correctly() {
    let json = serde_json::to_string(&DiscoveryErrorKind::Cancelled).unwrap();
    assert_eq!(json, "\"cancelled\"");
}

#[test]
fn discovery_error_cancelled_has_correct_kind() {
    let err = neat_ai_discovery::DiscoveryError::Cancelled;
    assert_eq!(err.error_kind(), DiscoveryErrorKind::Cancelled);
    assert_eq!(err.to_string(), "Analysis cancelled by host");
}

#[test]
fn string_classification_detects_cancellation() {
    assert_eq!(
        neat_ai_discovery::classify_error("Analysis cancelled by host"),
        DiscoveryErrorKind::Cancelled,
    );
    assert_eq!(
        neat_ai_discovery::classify_error("cancelled by host process"),
        DiscoveryErrorKind::Cancelled,
    );
}

// ============================================================================
// FFI output struct includes cancelled field
// ============================================================================

#[test]
fn analyze_parallel_output_includes_cancelled_field() {
    let output = neat_ai_discovery::AnalyzeParallelOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
        helpful_synapses: None,
        harmful_synapses: None,
        synapse_diagnostics: None,
        synapse_gpu_used: None,
        synapse_metadata: None,
        helpful_neurons: None,
        synapse_weight_updates: None,
        coordinated_structural_candidates: None,
        candidate_clusters: None,
        neuron_diagnostics: None,
        neuron_gpu_used: None,
        neuron_metadata: None,
        neuron_fingerprints: None,
        fingerprint_cache_hits: None,
        fingerprint_cache_misses: None,
        module_outcome_tracker: None,
        memory_budget_exceeded: None,
        cancelled: Some(true),
        memory_pressure_cancelled: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json_str = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(
        parsed.get("cancelled").and_then(serde_json::Value::as_bool),
        Some(true),
        "cancelled should be true in serialised JSON"
    );
}

#[test]
fn analyze_parallel_output_omits_cancelled_when_none() {
    let output = neat_ai_discovery::AnalyzeParallelOutput {
        success: true,
        schema_version: neat_ai_discovery::SCHEMA_VERSION.to_string(),
        helpful_synapses: None,
        harmful_synapses: None,
        synapse_diagnostics: None,
        synapse_gpu_used: None,
        synapse_metadata: None,
        helpful_neurons: None,
        synapse_weight_updates: None,
        coordinated_structural_candidates: None,
        candidate_clusters: None,
        neuron_diagnostics: None,
        neuron_gpu_used: None,
        neuron_metadata: None,
        neuron_fingerprints: None,
        fingerprint_cache_hits: None,
        fingerprint_cache_misses: None,
        module_outcome_tracker: None,
        memory_budget_exceeded: None,
        cancelled: None,
        memory_pressure_cancelled: None,
        error: None,
        error_kind: None,
        retryable: None,
    };

    let json_str = serde_json::to_string(&output).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert!(
        parsed.get("cancelled").is_none(),
        "cancelled should be omitted from JSON when None"
    );
}
