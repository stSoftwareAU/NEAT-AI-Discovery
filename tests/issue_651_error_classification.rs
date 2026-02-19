//! Integration tests for structured error classification (Issue #651).
//!
//! Verifies that FFI error responses include `error_kind` and `retryable` fields
//! with correct classification for different error scenarios.

use neat_ai_discovery::ffi_types::{
    DiscoveryErrorKind, classify_error, classify_panic, error_fields, no_error_fields,
};

// ============================================================================
// classify_error — GPU transient errors
// ============================================================================

#[test]
fn gpu_device_lost_is_classified_as_gpu_transient() {
    let kind = classify_error("Device is lost");
    assert_eq!(kind, DiscoveryErrorKind::GpuTransient);
    assert!(
        kind.is_retryable(),
        "GPU transient errors should be retryable"
    );
}

#[test]
fn gpu_device_poll_error_is_classified_as_gpu_transient() {
    let kind = classify_error("GPU device poll error (warm-up): device was lost");
    assert_eq!(kind, DiscoveryErrorKind::GpuTransient);
}

#[test]
fn gpu_command_buffer_overflow_is_classified_as_gpu_transient() {
    let kind = classify_error("Too many command buffers in flight");
    assert_eq!(kind, DiscoveryErrorKind::GpuTransient);
}

#[test]
fn gpu_driver_unresponsive_is_classified_as_gpu_transient() {
    let kind = classify_error("GPU driver may be unresponsive");
    assert_eq!(kind, DiscoveryErrorKind::GpuTransient);
}

// ============================================================================
// classify_error — GPU permanent errors
// ============================================================================

#[test]
fn no_gpu_available_is_classified_as_gpu_permanent() {
    let kind = classify_error("No GPU available");
    assert_eq!(kind, DiscoveryErrorKind::GpuPermanent);
    assert!(
        !kind.is_retryable(),
        "GPU permanent errors should not be retryable"
    );
}

#[test]
fn gpu_required_but_unavailable_is_classified_as_gpu_permanent() {
    let kind = classify_error("GPU is required but not available on this machine");
    assert_eq!(kind, DiscoveryErrorKind::GpuPermanent);
}

// ============================================================================
// classify_error — Data validation errors
// ============================================================================

#[test]
fn parse_failure_is_classified_as_data_validation() {
    let kind = classify_error("Failed to parse input JSON: expected value at line 1");
    assert_eq!(kind, DiscoveryErrorKind::DataValidation);
    assert!(
        !kind.is_retryable(),
        "Data validation errors should not be retryable"
    );
}

#[test]
fn null_input_is_classified_as_data_validation() {
    let kind = classify_error("Null input pointer");
    assert_eq!(kind, DiscoveryErrorKind::DataValidation);
}

#[test]
fn invalid_utf8_is_classified_as_data_validation() {
    let kind = classify_error("Invalid UTF-8 in input");
    assert_eq!(kind, DiscoveryErrorKind::DataValidation);
}

// ============================================================================
// classify_error — Timeout errors
// ============================================================================

#[test]
fn deadline_exceeded_is_classified_as_timeout() {
    let kind = classify_error("Analysis deadline exceeded after 30000ms");
    assert_eq!(kind, DiscoveryErrorKind::Timeout);
    assert!(kind.is_retryable(), "Timeout errors should be retryable");
}

#[test]
fn timed_out_is_classified_as_timeout() {
    let kind = classify_error("Operation timed out");
    assert_eq!(kind, DiscoveryErrorKind::Timeout);
}

// ============================================================================
// classify_error — Memory exhaustion
// ============================================================================

#[test]
fn out_of_memory_is_classified_as_memory_exhausted() {
    let kind = classify_error("Out of memory allocating GPU buffer");
    assert_eq!(kind, DiscoveryErrorKind::MemoryExhausted);
    assert!(kind.is_retryable(), "Memory exhaustion should be retryable");
}

#[test]
fn allocation_failed_is_classified_as_memory_exhausted() {
    let kind = classify_error("allocation failed: not enough memory");
    assert_eq!(kind, DiscoveryErrorKind::MemoryExhausted);
}

// ============================================================================
// classify_error — I/O errors
// ============================================================================

#[test]
fn parquet_error_is_classified_as_io() {
    let kind = classify_error("Failed to read parquet file: corrupted footer");
    assert_eq!(kind, DiscoveryErrorKind::IoError);
    assert!(kind.is_retryable(), "I/O errors should be retryable");
}

#[test]
fn file_not_found_is_classified_as_io() {
    let kind = classify_error("File not found: /tmp/records.parquet");
    assert_eq!(kind, DiscoveryErrorKind::IoError);
}

// ============================================================================
// classify_error — Internal panic
// ============================================================================

#[test]
fn panic_classification_returns_internal_panic() {
    let kind = classify_panic();
    assert_eq!(kind, DiscoveryErrorKind::InternalPanic);
    assert!(
        !kind.is_retryable(),
        "Internal panics should not be retryable"
    );
}

// ============================================================================
// classify_error — Unknown errors
// ============================================================================

#[test]
fn unrecognised_error_is_classified_as_unknown() {
    let kind = classify_error("Something completely unexpected happened");
    assert_eq!(kind, DiscoveryErrorKind::Unknown);
    assert!(
        !kind.is_retryable(),
        "Unknown errors should not be retryable by default"
    );
}

// ============================================================================
// error_fields and no_error_fields helpers
// ============================================================================

#[test]
fn error_fields_returns_classified_kind_and_retryable() {
    let (kind, retryable) = error_fields("Device is lost");
    assert_eq!(kind, Some(DiscoveryErrorKind::GpuTransient));
    assert_eq!(retryable, Some(true));
}

#[test]
fn error_fields_returns_non_retryable_for_validation() {
    let (kind, retryable) = error_fields("Failed to parse input JSON: unexpected token");
    assert_eq!(kind, Some(DiscoveryErrorKind::DataValidation));
    assert_eq!(retryable, Some(false));
}

#[test]
fn no_error_fields_returns_none_for_both() {
    let (kind, retryable) = no_error_fields();
    assert_eq!(kind, None);
    assert_eq!(retryable, None);
}

// ============================================================================
// JSON serialisation of error kind
// ============================================================================

#[test]
fn error_kind_serialises_as_snake_case_json() {
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::GpuTransient).unwrap(),
        "\"gpu_transient\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::GpuPermanent).unwrap(),
        "\"gpu_permanent\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::DataValidation).unwrap(),
        "\"data_validation\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::Timeout).unwrap(),
        "\"timeout\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::MemoryExhausted).unwrap(),
        "\"memory_exhausted\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::IoError).unwrap(),
        "\"io_error\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::InternalPanic).unwrap(),
        "\"internal_panic\""
    );
    assert_eq!(
        serde_json::to_string(&DiscoveryErrorKind::Unknown).unwrap(),
        "\"unknown\""
    );
}

// ============================================================================
// FFI response shape — verify error fields appear in JSON output
// ============================================================================

#[test]
fn analyze_parallel_error_response_includes_error_classification() {
    // Call analyze_parallel_internal with invalid JSON to trigger an error
    let result = neat_ai_discovery::analyze_parallel_internal("not valid json");
    let json_str = result.expect("Should return Ok with error JSON, not Err");
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(value["success"], false);
    assert!(value["error"].is_string(), "error field should be present");
    assert_eq!(
        value["errorKind"], "data_validation",
        "Parse errors should be classified as data_validation"
    );
    assert_eq!(
        value["retryable"], false,
        "Data validation errors should not be retryable"
    );
}

#[test]
fn success_response_omits_error_classification_fields() {
    // Use get_library_version_internal which always succeeds — verifies that
    // success responses do not include error_kind or retryable fields.
    let result = neat_ai_discovery::get_library_version_internal();
    let json_str = result.expect("Version query should succeed");
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(value["success"], true);
    // Success: error_kind and retryable should be absent (skip_serializing_if = None)
    assert!(
        value.get("errorKind").is_none() || value["errorKind"].is_null(),
        "Success response should not include errorKind"
    );
    assert!(
        value.get("retryable").is_none() || value["retryable"].is_null(),
        "Success response should not include retryable"
    );
}

#[test]
fn rank_focus_neurons_error_response_includes_error_classification() {
    let result = neat_ai_discovery::rank_focus_neurons_internal("{invalid}");
    let json_str = result.expect("Should return Ok with error JSON");
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    assert_eq!(value["success"], false);
    assert_eq!(
        value["errorKind"], "data_validation",
        "Parse errors should be classified as data_validation"
    );
    assert_eq!(value["retryable"], false);
}

#[test]
fn case_insensitive_error_classification() {
    // Verify classification works regardless of case
    assert_eq!(
        classify_error("DEVICE IS LOST"),
        DiscoveryErrorKind::GpuTransient
    );
    assert_eq!(
        classify_error("OUT OF MEMORY"),
        DiscoveryErrorKind::MemoryExhausted
    );
    assert_eq!(classify_error("TIMED OUT"), DiscoveryErrorKind::Timeout);
}

// ============================================================================
// Backward compatibility — existing fields are preserved
// ============================================================================

#[test]
fn existing_ffi_contract_preserved_with_additive_fields() {
    // Verify that the error response still has the existing fields (success, error)
    // and the new fields are additive.
    let result = neat_ai_discovery::analyze_parallel_internal("{}");
    let json_str = result.expect("Should return Ok with error JSON");
    let value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Existing contract fields must still be present
    assert!(
        value.get("success").is_some(),
        "success field must be present"
    );
    assert!(
        value.get("error").is_some(),
        "error field must be present on failure"
    );

    // New fields are additive
    assert!(
        value.get("errorKind").is_some(),
        "errorKind field should be present on failure"
    );
    assert!(
        value.get("retryable").is_some(),
        "retryable field should be present on failure"
    );
}
