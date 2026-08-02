//! Issue #1932 — a wedged GPU is non-retryable at the FFI boundary.
//!
//! A GPU that is present but no longer answering used to surface as a bare
//! `anyhow!` timeout string ("GPU helpful batch evaluation timed out after
//! 300s"), which classified as [`DiscoveryErrorKind::Timeout`] — documented as
//! "retryable with a longer deadline". The host took that advice literally and
//! kept extending the analysis deadline for a device that would never answer.
//!
//! These tests pin the replacement signal: the breaker trip sites and the
//! submission timeout arms construct a typed
//! [`DiscoveryError::GpuWedged`], which classifies as
//! [`DiscoveryErrorKind::GpuWedged`] — never retryable — and survives the FFI
//! JSON boundary as `"errorKind": "gpu_wedged"`, `"retryable": false`.

use neat_ai_discovery::AnalyzeParallelOutput;
use neat_ai_discovery::analysis::gpu::breaker::{GpuCircuitBreaker, GpuTripReason};
use neat_ai_discovery::analysis::gpu::queue::submission::{batch_timeout_error, queue_full_error};
use neat_ai_discovery::ffi_types::{
    DiscoveryError, DiscoveryErrorKind, classify_anyhow_error, classify_error,
    error_fields_from_anyhow,
};

/// Serialise a failed analysis response the way `analyze_parallel_internal`
/// does, then parse it back as the Deno host would.
fn host_json(error: &anyhow::Error) -> serde_json::Value {
    let output = AnalyzeParallelOutput::failure(error);
    let json = serde_json::to_string(&output).expect("failure response must serialise");
    serde_json::from_str(&json).expect("failure response must be valid JSON")
}

// ============================================================================
// (a) The new kind is never retryable
// ============================================================================

#[test]
fn gpu_wedged_kind_is_not_retryable() {
    assert!(
        !DiscoveryErrorKind::GpuWedged.is_retryable(),
        "a wedged GPU cannot be recovered by retrying inside this process"
    );
    assert!(
        !DiscoveryErrorKind::GpuWedged.is_cancelled(),
        "a wedged GPU is a failure, not a host-requested cancellation"
    );
}

#[test]
fn gpu_wedged_kind_serialises_as_snake_case() {
    let json = serde_json::to_string(&DiscoveryErrorKind::GpuWedged).expect("kind serialises");
    assert_eq!(json, "\"gpu_wedged\"");
}

#[test]
fn typed_gpu_wedged_error_maps_to_the_gpu_wedged_kind() {
    let typed = DiscoveryError::GpuWedged {
        detail: "GPU helpful batch evaluation timed out after 300s".to_string(),
        abandoned_threads: 2,
    };

    assert_eq!(typed.error_kind(), DiscoveryErrorKind::GpuWedged);

    let message = typed.to_string();
    assert!(
        message.contains("abandoned GPU threads: 2"),
        "the abandoned-thread count must reach the host: {message}"
    );
    assert!(
        message.contains("remainder of this process"),
        "the message must state the GPU is wedged for the rest of the process: {message}"
    );
    assert!(
        message.contains("restart"),
        "the message must tell the operator the worker is restarted externally: {message}"
    );
    assert!(
        !message.contains("reducing batch size"),
        "the old advice to reduce batch size must be gone: {message}"
    );
}

// ============================================================================
// (b) Real trip sites classify as GpuWedged, not Timeout
// ============================================================================

/// The exact failure from the #1926 incident log: the caller burned its full
/// 300s wait and the GPU never answered.
#[test]
fn a_submission_batch_timeout_classifies_as_gpu_wedged_not_timeout() {
    let breaker = GpuCircuitBreaker::new();
    let err = batch_timeout_error(&breaker, "helpful batch evaluation", 300);

    assert_eq!(
        classify_anyhow_error(&err),
        DiscoveryErrorKind::GpuWedged,
        "a wedged-GPU batch timeout must not classify as a plain Timeout"
    );
    assert!(
        breaker.is_tripped(),
        "the batch timeout must still trip the breaker"
    );
    assert_eq!(breaker.trip_reason(), Some(GpuTripReason::BatchTimeout));
}

#[test]
fn a_full_queue_send_timeout_classifies_as_gpu_wedged() {
    let breaker = GpuCircuitBreaker::new();
    let err = queue_full_error(&breaker, 60);

    assert_eq!(classify_anyhow_error(&err), DiscoveryErrorKind::GpuWedged);
    assert!(
        format!("{err:#}").contains("GPU work queue full"),
        "the caller keeps the diagnostic detail: {err:#}"
    );
}

/// Every call suppressed by the tripped breaker must carry the same verdict —
/// otherwise the host resumes extending the deadline on the second attempt.
#[test]
fn a_suppressed_call_on_a_tripped_breaker_classifies_as_gpu_wedged() {
    let breaker = GpuCircuitBreaker::new();
    breaker.trip(GpuTripReason::AbandonedThread);

    let err = breaker.check().expect_err("a tripped breaker refuses work");

    assert_eq!(classify_anyhow_error(&err), DiscoveryErrorKind::GpuWedged);
    let (msg, kind, retryable) = error_fields_from_anyhow(&err);
    assert_eq!(kind, Some(DiscoveryErrorKind::GpuWedged));
    assert_eq!(retryable, Some(false));
    assert!(
        msg.contains("GPU circuit breaker tripped"),
        "the trip reason must survive: {msg}"
    );
}

/// Defence in depth: a wedged-GPU error that reaches the host as text only —
/// no typed error to downcast — must still classify as wedged rather than
/// falling through to the retryable `Timeout` branch.
#[test]
fn stringified_wedged_errors_classify_without_the_typed_downcast() {
    for message in [
        "GPU wedged: GPU helpful batch evaluation timed out after 300s (abandoned GPU threads: 1)",
        "GPU circuit breaker tripped: a GPU batch submission timed out",
    ] {
        assert_eq!(
            classify_error(message),
            DiscoveryErrorKind::GpuWedged,
            "string fallback must not classify a wedged GPU as retryable: {message}"
        );
    }
}

// ============================================================================
// (c) The kind survives the FFI JSON boundary
// ============================================================================

#[test]
fn the_host_json_carries_gpu_wedged_and_retryable_false() {
    let breaker = GpuCircuitBreaker::new();
    let err = batch_timeout_error(&breaker, "helpful batch evaluation", 300);

    let json = host_json(&err);

    assert_eq!(json["success"], false);
    assert_eq!(
        json["errorKind"], "gpu_wedged",
        "the host must receive the wedged kind, not `timeout`: {json}"
    );
    assert_eq!(
        json["retryable"], false,
        "the host must be told not to retry: {json}"
    );

    let error = json["error"].as_str().expect("an error message is present");
    assert!(
        error.contains("abandoned GPU threads:"),
        "the abandoned-thread count must reach the host: {error}"
    );
    assert!(
        error.contains("timed out after 300s"),
        "the original diagnostic must survive: {error}"
    );
}

// ============================================================================
// Regression guards — the existing taxonomy is unchanged
// ============================================================================

#[test]
fn gpu_unavailable_and_device_lost_classification_is_unchanged() {
    let unavailable = DiscoveryError::GpuUnavailable {
        reason: "No GPU adapter found".to_string(),
    };
    assert_eq!(unavailable.error_kind(), DiscoveryErrorKind::GpuPermanent);
    assert!(!unavailable.error_kind().is_retryable());

    let device_lost = DiscoveryError::GpuDeviceLost {
        detail: "device was lost".to_string(),
    };
    assert_eq!(device_lost.error_kind(), DiscoveryErrorKind::GpuTransient);
    assert!(device_lost.error_kind().is_retryable());

    // The string fallbacks for both are byte-for-byte the same as before.
    assert_eq!(
        classify_error("No GPU available on this system"),
        DiscoveryErrorKind::GpuPermanent
    );
    assert_eq!(
        classify_error("GPU device poll error (warm-up): device was lost"),
        DiscoveryErrorKind::GpuTransient
    );
}

/// A genuine deadline overrun — one the host *can* fix with a longer deadline —
/// must keep classifying as the retryable `Timeout`.
#[test]
fn an_ordinary_deadline_overrun_is_still_a_retryable_timeout() {
    assert_eq!(
        classify_error("Analysis deadline exceeded"),
        DiscoveryErrorKind::Timeout
    );
    assert!(DiscoveryErrorKind::Timeout.is_retryable());
}
