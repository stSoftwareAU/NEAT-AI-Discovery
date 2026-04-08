//! Integration tests for GPU device-lost recovery (Issue #647)
//!
//! Tests that the device-lost error detection and retry configuration work
//! correctly. Actual GPU recovery is tested implicitly via the GPU thread
//! loop, but cannot be reliably simulated in unit tests without a real GPU
//! device-lost event.

use neat_ai_discovery::analysis::gpu::{
    DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS, DEFAULT_GPU_RETRY_LIMIT,
    GPU_RETRY_LIMIT_ENV, backoff_delay_ms, is_device_lost_error,
};

// =============================================================================
// Device-lost error detection
// =============================================================================

#[test]
fn test_device_lost_detection_recognises_wgpu_patterns() {
    // These error messages are produced by wgpu when the GPU device is lost
    let device_lost_messages = vec![
        "Device is lost",
        "wgpu: the GPU device was lost during compute",
        "GPU device poll error (helpful-pipeline): internal error",
        "Out of memory when allocating staging buffer",
        "Too many command buffers in flight",
        "GPU driver may be unresponsive after 30.0s",
    ];

    for msg in device_lost_messages {
        let err = anyhow::anyhow!("{msg}");
        assert!(
            is_device_lost_error(&err),
            "Expected device-lost detection for: {msg}"
        );
    }
}

#[test]
fn test_device_lost_detection_ignores_normal_errors() {
    // These are normal operational errors that should NOT trigger recovery
    let normal_errors = vec![
        "Invalid sample count: 0",
        "Neuron UUID not found in records",
        "Batch evaluation returned empty results",
        "Channel disconnected",
        "Timeout waiting for response",
        "No focus neurons to analyse",
    ];

    for msg in normal_errors {
        let err = anyhow::anyhow!("{msg}");
        assert!(
            !is_device_lost_error(&err),
            "Should NOT detect device-lost for normal error: {msg}"
        );
    }
}

#[test]
fn test_device_lost_detection_is_case_insensitive() {
    let variations = vec![
        "DEVICE IS LOST",
        "Device Is Lost",
        "device is lost",
        "OUT OF MEMORY",
    ];

    for msg in variations {
        let err = anyhow::anyhow!("{msg}");
        assert!(
            is_device_lost_error(&err),
            "Expected case-insensitive detection for: {msg}"
        );
    }
}

#[test]
fn test_device_lost_detection_with_nested_errors() {
    // anyhow errors can be chained; the {:#} format shows the full chain
    let inner = anyhow::anyhow!("device is lost");
    let outer = anyhow::anyhow!("GPU operation failed").context(inner);
    assert!(is_device_lost_error(&outer));
}

// =============================================================================
// Retry limit configuration
// =============================================================================

#[test]
fn test_default_retry_limit_is_three() {
    assert_eq!(DEFAULT_GPU_RETRY_LIMIT, 3);
}

#[test]
fn test_retry_limit_env_var_name_is_correct() {
    assert_eq!(GPU_RETRY_LIMIT_ENV, "NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT");
}

// =============================================================================
// Recovery module integration
// =============================================================================

#[test]
fn test_device_lost_error_with_buffer_mapping_context() {
    // Simulate a buffer mapping failure that indicates device loss
    let err =
        anyhow::anyhow!("GPU device poll error (helpful-pipeline): device lost after timeout");
    assert!(is_device_lost_error(&err));
}

#[test]
fn test_device_lost_error_with_allocation_failure() {
    // Simulate a memory allocation failure
    let err = anyhow::anyhow!(
        "Failed to create staging buffer: Out of memory (requested 256MB, available 0)"
    );
    assert!(is_device_lost_error(&err));
}

#[test]
fn test_device_lost_error_detects_allocation_failed_pattern() {
    // Issue #1038: Verify the corrected "allocation failed" pattern is detected
    let err = anyhow::anyhow!("GPU allocation failed for compute buffer");
    assert!(
        is_device_lost_error(&err),
        "Expected device-lost detection for 'allocation failed'"
    );
}

// =============================================================================
// Exponential backoff (Issue #1038)
// =============================================================================

#[test]
fn test_backoff_delay_doubles_exponentially() {
    let delays: Vec<u64> = (1..=7)
        .map(|a| backoff_delay_ms(a, DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS))
        .collect();
    assert_eq!(delays, vec![10, 20, 40, 80, 160, 320, 640]);
}

#[test]
fn test_backoff_delay_caps_at_max() {
    // Attempt 8 with 10ms initial = 10 * 128 = 1280, capped at 1000
    let delay = backoff_delay_ms(8, DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS);
    assert_eq!(delay, DEFAULT_BACKOFF_MAX_MS);
}

#[test]
fn test_backoff_delay_does_not_overflow_on_large_attempt() {
    // Very large attempt number must not panic
    let delay = backoff_delay_ms(200, DEFAULT_BACKOFF_INITIAL_MS, DEFAULT_BACKOFF_MAX_MS);
    assert_eq!(delay, DEFAULT_BACKOFF_MAX_MS);
}

#[test]
fn test_backoff_constants_are_sensible() {
    assert_eq!(DEFAULT_BACKOFF_INITIAL_MS, 10);
    assert_eq!(DEFAULT_BACKOFF_MAX_MS, 1_000);
    const { assert!(DEFAULT_BACKOFF_INITIAL_MS < DEFAULT_BACKOFF_MAX_MS) };
}
