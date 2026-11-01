//! Issue #2440 — classify and expose the complete anyhow cause chain.

use neat_ai_discovery::AnalyzeParallelOutput;
use neat_ai_discovery::ffi_types::{DiscoveryError, DiscoveryErrorKind, classify_error};

fn host_json(error: &anyhow::Error) -> serde_json::Value {
    let output = AnalyzeParallelOutput::failure(error);
    let json = serde_json::to_string(&output).expect("failure response must serialise");
    serde_json::from_str(&json).expect("failure response must be valid JSON")
}

#[test]
fn wrapped_analysis_failures_preserve_context_and_classify_the_root_cause() {
    let cases = [
        (
            "GPU device lost and recovery failed after 3 attempts: Device is lost",
            "gpu_transient",
            true,
        ),
        ("analysis deadline exceeded", "timeout", true),
        (
            "out of memory allocating GPU buffer",
            "memory_exhausted",
            true,
        ),
        ("failed to read parquet file", "io_error", true),
        ("unexpected analysis failure", "unknown", false),
    ];

    for (root_cause, expected_kind, expected_retryable) in cases {
        let error = anyhow::anyhow!(root_cause)
            .context("failed during synapse analysis phase")
            .context("failed during analysis dispatch");

        let json = host_json(&error);

        assert_eq!(json["success"], false);
        assert_eq!(json["errorKind"], expected_kind, "cause: {root_cause}");
        assert_eq!(json["retryable"], expected_retryable, "cause: {root_cause}");
        let message = json["error"]
            .as_str()
            .expect("failure response must include an error message");
        assert!(
            message.contains("failed during analysis dispatch"),
            "outer context must reach the host: {message}"
        );
        assert!(
            message.contains("failed during synapse analysis phase"),
            "inner context must reach the host: {message}"
        );
        assert!(
            message.contains(root_cause),
            "root cause must reach the host: {message}"
        );
    }
}

#[test]
fn context_wrapped_typed_gpu_wedged_error_keeps_typed_precedence() {
    let typed = DiscoveryError::GpuWedged {
        detail: "GPU helpful batch evaluation timed out after 300s".to_string(),
        abandoned_threads: 1,
    };
    let error =
        anyhow::Error::new(typed).context("analysis dispatch timed out while GPU device was lost");

    let json = host_json(&error);

    assert_eq!(json["success"], false);
    assert_eq!(json["errorKind"], "gpu_wedged");
    assert_eq!(json["retryable"], false);
    let message = json["error"]
        .as_str()
        .expect("failure response must include an error message");
    assert!(message.contains("analysis dispatch timed out"));
    assert!(message.contains("GPU wedged:"));
    assert!(message.contains("abandoned GPU threads: 1"));
}

#[test]
fn typed_timeout_takes_precedence_over_conflicting_device_lost_context() {
    let error = anyhow::Error::new(DiscoveryError::Timeout { deadline_ms: 300 })
        .context("analysis dispatch failed: Device is lost");

    assert_eq!(
        classify_error(&format!("{error:#}")),
        DiscoveryErrorKind::GpuTransient,
        "fixture must classify differently when only its text is considered"
    );

    let json = host_json(&error);

    assert_eq!(json["success"], false);
    assert_eq!(json["errorKind"], "timeout");
    assert_eq!(json["retryable"], true);
    let message = json["error"]
        .as_str()
        .expect("failure response must include an error message");
    assert!(message.contains("analysis dispatch failed: Device is lost"));
    assert!(message.contains("Deadline exceeded after 300ms"));
}
