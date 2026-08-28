//! Internal business-logic functions for GPU probe FFI entry points.

use anyhow::Result;

use crate::analysis;
use crate::analysis::gpu::GpuAvailabilityResult;
use crate::ffi_types::*;

/// Probes whether a compatible GPU is available.
///
/// Returns JSON output for easy integration with TypeScript/DenoJS.
pub fn check_gpu_available_internal() -> Result<String> {
    let result = analysis::GpuAnalyzer::check_gpu_availability();
    let output = build_check_gpu_output(result);
    Ok(serde_json::to_string(&output)?)
}

/// Build the FFI capability verdict from a GPU availability probe result.
///
/// Issue #1419: a GPU-less host must receive a *distinct, structured*
/// verdict it can branch on **before** starting a discovery pass. The crate
/// hard-requires a GPU for synapse/neuron analysis, so without this verdict a
/// GPU-less host runs every pass to a guaranteed `0 candidates` result that is
/// indistinguishable from genuine search exhaustion.
///
/// The mapping covers three cases:
/// - **Hard error** (`is_error`, e.g. macOS where Metal should always work):
///   `success = false` with a `GpuUnavailable` error and `gpu_permanent` kind.
/// - **GPU-less host** (probe succeeded, no usable GPU — common on headless
///   Linux): `success = true`, `gpuAvailable = false`, with the unavailability
///   reason *classified* into a structured `error_kind`/`retryable` pair so the
///   caller can tell a permanent "discovery-unsupported-on-host" skip from a
///   transient error worth retrying.
/// - **GPU available**: `success = true`, `gpuAvailable = true`, no error
///   classification.
pub(crate) fn build_check_gpu_output(result: GpuAvailabilityResult) -> CheckGpuOutput {
    if result.is_error {
        // On macOS, missing GPU is an error (Metal should always work).
        let typed = DiscoveryError::GpuUnavailable {
            reason: "GPU required but not available".to_string(),
        };
        let kind = typed.error_kind();
        CheckGpuOutput {
            success: false,
            gpu_available: false,
            reason: result.reason,
            error: Some(typed.to_string()),
            error_kind: Some(kind),
            retryable: Some(kind.is_retryable()),
        }
    } else if !result.available {
        // GPU-less host (common on headless Linux servers). The probe itself
        // succeeded, but discovery is unsupported here. Classify the reason so
        // the caller can branch on a permanent skip vs a transient retry
        // instead of running a guaranteed 0-candidate pass (Issue #1419).
        let kind = classify_gpu_unavailable_reason(result.reason.as_deref());
        CheckGpuOutput {
            success: true,
            gpu_available: false,
            reason: result.reason,
            error: None,
            error_kind: Some(kind),
            retryable: Some(kind.is_retryable()),
        }
    } else {
        let (error_kind, retryable) = no_error_fields();
        CheckGpuOutput {
            success: true,
            gpu_available: true,
            reason: result.reason,
            error: None,
            error_kind,
            retryable,
        }
    }
}

/// Classify *why* a GPU is unavailable into a structured [`DiscoveryErrorKind`].
///
/// Defaults to [`DiscoveryErrorKind::GpuPermanent`] when the reason does not
/// match a more specific transient (device lost) or memory pattern — a host
/// with no usable GPU is unsupported, not worth retrying.
fn classify_gpu_unavailable_reason(reason: Option<&str>) -> DiscoveryErrorKind {
    match reason {
        Some(text) => match classify_error(text) {
            DiscoveryErrorKind::Unknown => DiscoveryErrorKind::GpuPermanent,
            kind => kind,
        },
        None => DiscoveryErrorKind::GpuPermanent,
    }
}

/// Returns the library version string as JSON.
///
/// Returns JSON output for easy integration with TypeScript/DenoJS.
pub fn get_library_version_internal() -> Result<String> {
    let (error_kind, retryable) = no_error_fields();
    let output = GetVersionOutput {
        success: true,
        version: crate::LIB_VERSION.to_string(),
        schema_version: SCHEMA_VERSION.to_string(),
        error: None,
        error_kind,
        retryable,
    };
    Ok(serde_json::to_string(&output)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A GPU-less host (graceful disable, no hard error) must yield a distinct
    /// structured verdict — `gpuAvailable = false` with a non-retryable
    /// `gpu_permanent` classification — never a silent "available" answer that
    /// would let the caller run a guaranteed 0-candidate pass (Issue #1419).
    #[test]
    fn gpu_less_host_yields_distinct_permanent_verdict() {
        let result = GpuAvailabilityResult {
            available: false,
            reason: Some("No GPU adapter found. Discovery disabled on this machine.".to_string()),
            is_error: false,
        };

        let output = build_check_gpu_output(result);

        assert!(
            output.success,
            "probe itself succeeded — only the GPU is missing"
        );
        assert!(!output.gpu_available, "no usable GPU on this host");
        assert_eq!(
            output.error_kind,
            Some(DiscoveryErrorKind::GpuPermanent),
            "a GPU-less host is a permanent, non-retryable verdict"
        );
        assert_eq!(output.retryable, Some(false));
        assert!(
            output.reason.is_some(),
            "the verdict must carry a human-readable reason"
        );
    }

    /// An operator-disabled GPU (GRQ#4405) must reach the caller as a
    /// **permanent, non-retryable** verdict that still names the variable —
    /// never a hard error, and never something the host retries as though the
    /// driver had blipped.
    #[test]
    fn operator_disabled_gpu_is_a_permanent_non_error_verdict() {
        let output = build_check_gpu_output(crate::analysis::gpu::gpu_disabled_result());

        assert!(output.success, "the probe answered; it did not fail");
        assert!(!output.gpu_available);
        assert_eq!(output.error_kind, Some(DiscoveryErrorKind::GpuPermanent));
        assert_eq!(output.retryable, Some(false));
        assert!(
            output.error.is_none(),
            "a configuration choice is not an error"
        );
        assert!(
            output
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("NEAT_AI_DISCOVERY_GPU=off"),
            "the verdict must name the variable that caused it"
        );
    }

    /// A transient GPU failure (device lost / creation failure) must classify
    /// as retryable so the caller retries rather than skipping the host.
    #[test]
    fn transient_gpu_failure_is_retryable() {
        let result = GpuAvailabilityResult {
            available: false,
            reason: Some("GPU device creation failed: device was lost".to_string()),
            is_error: false,
        };

        let output = build_check_gpu_output(result);

        assert!(!output.gpu_available);
        assert_eq!(output.error_kind, Some(DiscoveryErrorKind::GpuTransient));
        assert_eq!(output.retryable, Some(true));
    }

    /// A hard error (e.g. macOS where Metal should always work) keeps the
    /// `success = false` error verdict with a permanent classification.
    #[test]
    fn hard_error_reports_unsuccessful_permanent_verdict() {
        let result = GpuAvailabilityResult {
            available: false,
            reason: Some("Metal should always be available".to_string()),
            is_error: true,
        };

        let output = build_check_gpu_output(result);

        assert!(!output.success);
        assert!(!output.gpu_available);
        assert_eq!(output.error_kind, Some(DiscoveryErrorKind::GpuPermanent));
        assert_eq!(output.retryable, Some(false));
        assert!(
            output.error.is_some(),
            "hard error must carry an error message"
        );
    }

    /// When a usable GPU is present, the verdict is a clean success with no
    /// error classification.
    #[test]
    fn available_gpu_yields_clean_success() {
        let result = GpuAvailabilityResult {
            available: true,
            reason: None,
            is_error: false,
        };

        let output = build_check_gpu_output(result);

        assert!(output.success);
        assert!(output.gpu_available);
        assert!(output.error_kind.is_none());
        assert!(output.retryable.is_none());
        assert!(output.error.is_none());
    }

    /// A missing reason still produces a permanent (non-retryable) verdict so
    /// the caller never treats an unexplained no-GPU host as retryable.
    #[test]
    fn missing_reason_defaults_to_permanent() {
        let result = GpuAvailabilityResult {
            available: false,
            reason: None,
            is_error: false,
        };

        let output = build_check_gpu_output(result);

        assert_eq!(output.error_kind, Some(DiscoveryErrorKind::GpuPermanent));
        assert_eq!(output.retryable, Some(false));
    }
}
