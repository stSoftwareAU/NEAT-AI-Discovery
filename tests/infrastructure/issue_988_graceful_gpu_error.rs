//! Integration tests for graceful GPU unavailable error handling (Issue #988).
//!
//! Verifies that when `DiscoveryError::GpuUnavailable` is returned from analysis
//! functions, it propagates through the FFI layer as a structured JSON error with
//! `errorKind: "gpu_permanent"` instead of panicking the thread.

use neat_ai_discovery::ffi_types::{
    DiscoveryError, DiscoveryErrorKind, classify_anyhow_error, error_fields_from_anyhow,
};

// ============================================================================
// GpuUnavailable error propagates correctly through anyhow
// ============================================================================

#[test]
fn gpu_unavailable_error_propagates_as_gpu_permanent_through_anyhow() {
    let typed_err = DiscoveryError::GpuUnavailable {
        reason: "No compatible GPU adapter found on this system".to_string(),
    };
    let anyhow_err: anyhow::Error = typed_err.into();

    let kind = classify_anyhow_error(&anyhow_err);
    assert_eq!(kind, DiscoveryErrorKind::GpuPermanent);
    assert!(
        !kind.is_retryable(),
        "GPU permanent errors should not be retryable"
    );
}

#[test]
fn gpu_unavailable_error_fields_produce_correct_triple() {
    let typed_err = DiscoveryError::GpuUnavailable {
        reason: "No compatible GPU adapter found on this system".to_string(),
    };
    let anyhow_err: anyhow::Error = typed_err.into();

    let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&anyhow_err);
    assert!(
        err_msg.contains("GPU unavailable"),
        "Error message should contain 'GPU unavailable', got: {err_msg}"
    );
    assert_eq!(error_kind, Some(DiscoveryErrorKind::GpuPermanent));
    assert_eq!(retryable, Some(false));
}

// ============================================================================
// analyze_parallel_internal returns structured error when GPU is unavailable
// ============================================================================

/// On machines without a GPU, `analyze_parallel_internal` must return a
/// structured JSON error with `errorKind: "gpu_permanent"` instead of panicking.
///
/// On machines with a GPU, the analysis proceeds normally (success or a
/// different error), so the test only validates the no-GPU path when
/// `GpuAnalyzer::gpu_is_available()` returns false.
#[test]
fn analyze_parallel_returns_structured_error_when_no_gpu() {
    use neat_ai_discovery::analysis::GpuAnalyzer;

    // Only test the no-GPU path on machines that actually lack a GPU.
    // On GPU-equipped machines this test is a no-op (the fix is validated
    // by the error propagation tests above).
    if GpuAnalyzer::gpu_is_available() {
        return;
    }

    // Minimal valid input that would previously trigger the panic
    let input_json = serde_json::json!({
        "parquetFile": "/tmp/nonexistent.parquet",
        "creature": {
            "neurons": [
                {"uuid": "input-1", "neuronType": "input", "squash": "IDENTITY", "bias": 0.0},
                {"uuid": "output-1", "neuronType": "output", "squash": "LOGISTIC", "bias": 0.0}
            ],
            "synapses": [
                {"fromUuid": "input-1", "toUuid": "output-1", "weight": 1.0}
            ]
        },
        "focusNeurons": ["output-1"],
        "maxSynapseCandidates": 10,
        "maxNeuronCandidates": 10,
        "analysisDeadlineMs": 5000,
        "randomSeed": 42
    });

    let result = neat_ai_discovery::analyze_parallel_internal(&input_json.to_string());
    let json_str = result.expect("Should return Ok with error JSON, not panic");
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(value["success"], false, "Should report failure");
    assert_eq!(
        value["errorKind"], "gpu_permanent",
        "Should classify as gpu_permanent, got: {:?}",
        value["errorKind"]
    );
    assert_eq!(
        value["retryable"], false,
        "GPU permanent errors should not be retryable"
    );
    assert!(
        value["error"]
            .as_str()
            .unwrap_or("")
            .contains("GPU unavailable"),
        "Error message should mention GPU unavailable, got: {:?}",
        value["error"]
    );
}
