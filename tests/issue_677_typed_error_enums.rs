//! Integration tests for typed error enums (Issue #677).
//!
//! Verifies that `DiscoveryError` typed enums classify correctly via pattern
//! matching instead of string matching, and that the external JSON format
//! remains backward-compatible.

use neat_ai_discovery::ffi_types::{
    DiscoveryError, DiscoveryErrorKind, classify_anyhow_error, classify_error,
};

// ============================================================================
// DiscoveryError variants map to correct DiscoveryErrorKind
// ============================================================================

#[test]
fn gpu_unavailable_error_classifies_as_gpu_permanent() {
    let err = DiscoveryError::GpuUnavailable {
        reason: "No Metal device found".to_string(),
    };
    assert_eq!(err.error_kind(), DiscoveryErrorKind::GpuPermanent);
    assert!(!err.error_kind().is_retryable());
}

#[test]
fn gpu_device_lost_error_classifies_as_gpu_transient() {
    let err = DiscoveryError::GpuDeviceLost {
        detail: "Device is lost".to_string(),
    };
    assert_eq!(err.error_kind(), DiscoveryErrorKind::GpuTransient);
    assert!(err.error_kind().is_retryable());
}

#[test]
fn invalid_input_error_classifies_as_data_validation() {
    let err = DiscoveryError::InvalidInput {
        detail: "Missing required field 'creature'".to_string(),
    };
    assert_eq!(err.error_kind(), DiscoveryErrorKind::DataValidation);
    assert!(!err.error_kind().is_retryable());
}

#[test]
fn timeout_error_classifies_as_timeout() {
    let err = DiscoveryError::Timeout {
        deadline_ms: 30_000,
    };
    assert_eq!(err.error_kind(), DiscoveryErrorKind::Timeout);
    assert!(err.error_kind().is_retryable());
}

#[test]
fn memory_exhausted_error_classifies_as_memory_exhausted() {
    let err = DiscoveryError::MemoryExhausted {
        detail: "GPU buffer allocation failed".to_string(),
    };
    assert_eq!(err.error_kind(), DiscoveryErrorKind::MemoryExhausted);
    assert!(err.error_kind().is_retryable());
}

#[test]
fn io_error_classifies_as_io_error() {
    let err = DiscoveryError::Io {
        detail: "File not found: /tmp/records.parquet".to_string(),
    };
    assert_eq!(err.error_kind(), DiscoveryErrorKind::IoError);
    assert!(err.error_kind().is_retryable());
}

// ============================================================================
// classify_anyhow_error — downcast to DiscoveryError first
// ============================================================================

#[test]
fn classify_anyhow_error_downcasts_typed_error() {
    let typed_err = DiscoveryError::Timeout { deadline_ms: 5_000 };
    let anyhow_err: anyhow::Error = typed_err.into();
    let kind = classify_anyhow_error(&anyhow_err);
    assert_eq!(kind, DiscoveryErrorKind::Timeout);
}

#[test]
fn classify_anyhow_error_falls_back_to_string_matching() {
    // Create a non-DiscoveryError anyhow error with a known message pattern
    let anyhow_err = anyhow::anyhow!("Device is lost during operation");
    let kind = classify_anyhow_error(&anyhow_err);
    assert_eq!(kind, DiscoveryErrorKind::GpuTransient);
}

#[test]
fn classify_anyhow_error_unknown_for_unrecognised_error() {
    let anyhow_err = anyhow::anyhow!("Something completely unexpected");
    let kind = classify_anyhow_error(&anyhow_err);
    assert_eq!(kind, DiscoveryErrorKind::Unknown);
}

// ============================================================================
// DiscoveryError Display messages are human-readable
// ============================================================================

#[test]
fn discovery_error_display_messages_are_descriptive() {
    let err = DiscoveryError::GpuUnavailable {
        reason: "No Metal device".to_string(),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("No Metal device"),
        "Error message should include the reason"
    );

    let err = DiscoveryError::Timeout {
        deadline_ms: 30_000,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("30000"),
        "Timeout message should include the deadline"
    );
}

// ============================================================================
// Backward compatibility — string-based classify_error still works
// ============================================================================

#[test]
fn string_based_classify_error_still_works_after_typed_errors() {
    // Existing string-based classification must continue to work
    assert_eq!(
        classify_error("Device is lost"),
        DiscoveryErrorKind::GpuTransient
    );
    assert_eq!(
        classify_error("No GPU available"),
        DiscoveryErrorKind::GpuPermanent
    );
    assert_eq!(
        classify_error("Failed to parse input JSON"),
        DiscoveryErrorKind::DataValidation
    );
    assert_eq!(
        classify_error("Analysis deadline exceeded"),
        DiscoveryErrorKind::Timeout
    );
    assert_eq!(
        classify_error("Out of memory"),
        DiscoveryErrorKind::MemoryExhausted
    );
    assert_eq!(
        classify_error("parquet read error"),
        DiscoveryErrorKind::IoError
    );
}

// ============================================================================
// FFI response shape — typed errors produce same JSON as string-matched errors
// ============================================================================

#[test]
fn typed_error_produces_same_json_classification_as_string_error() {
    // A typed InvalidInput error and a string "Failed to parse input JSON" should
    // both produce DataValidation classification
    let typed = DiscoveryError::InvalidInput {
        detail: "bad JSON".to_string(),
    };
    assert_eq!(typed.error_kind(), DiscoveryErrorKind::DataValidation);
    assert_eq!(
        classify_error("Failed to parse input JSON: bad JSON"),
        DiscoveryErrorKind::DataValidation
    );
}

#[test]
fn analyze_parallel_with_invalid_json_still_returns_data_validation() {
    // Existing FFI behaviour must be preserved
    let result = neat_ai_discovery::analyze_parallel_internal("not valid json");
    let json_str = result.expect("Should return Ok with error JSON");
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(value["success"], false);
    assert_eq!(value["errorKind"], "data_validation");
    assert_eq!(value["retryable"], false);
}
