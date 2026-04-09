//! Structured error classification for FFI responses (Issue #651, #677).
//!
//! Provides both **typed error enums** (`DiscoveryError`) and string-based
//! classification (`classify_error`) so the NEAT-AI controller can make informed
//! retry decisions — retrying transient GPU errors but not retrying data
//! validation failures.
//!
//! Prefer constructing a `DiscoveryError` variant when the error category is
//! known at the call site. The string-based fallback handles third-party errors
//! (wgpu, parquet, etc.) whose messages we cannot control.

use serde::Serialize;
use thiserror::Error;

/// Classification of discovery errors for retry decision-making.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DiscoveryErrorKind {
    /// Transient GPU error (device lost, driver reset) — retryable.
    GpuTransient,
    /// Permanent GPU error (no GPU available, unsupported hardware) — not retryable.
    GpuPermanent,
    /// Data validation error (malformed input, missing fields) — not retryable.
    DataValidation,
    /// Deadline/timeout exceeded — retryable with a longer deadline.
    Timeout,
    /// Memory exhaustion — retryable after freeing resources.
    MemoryExhausted,
    /// File I/O error (parquet read/write failure) — may be retryable.
    IoError,
    /// Internal panic caught at the FFI boundary — not retryable.
    InternalPanic,
    /// Analysis was cancelled via `cancel_analysis()` FFI call (Issue #1047).
    /// Not an error — the host requested graceful shutdown.
    Cancelled,
    /// Unclassified error — check the error message for details.
    Unknown,
}

impl DiscoveryErrorKind {
    /// Whether errors of this kind are typically worth retrying.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::GpuTransient | Self::Timeout | Self::MemoryExhausted | Self::IoError
        )
    }

    /// Whether this kind represents a host-requested cancellation (Issue #1047).
    pub fn is_cancelled(self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

// ============================================================================
// Typed domain error enum (Issue #677)
// ============================================================================

/// Typed domain errors for the discovery library.
///
/// Each variant maps to a `DiscoveryErrorKind` via `error_kind()`, removing the
/// need for string matching when the error category is known at construction.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// GPU is permanently unavailable (no device, unsupported hardware).
    #[error("GPU unavailable: {reason}")]
    GpuUnavailable { reason: String },

    /// Transient GPU failure (device lost, driver reset).
    #[error("GPU device lost: {detail}")]
    GpuDeviceLost { detail: String },

    /// Invalid input data (parse failure, missing fields, bad format).
    #[error("Invalid input: {detail}")]
    InvalidInput { detail: String },

    /// Analysis deadline exceeded.
    #[error("Deadline exceeded after {deadline_ms}ms")]
    Timeout { deadline_ms: u64 },

    /// Memory exhaustion (GPU buffer allocation, system OOM).
    #[error("Memory exhausted: {detail}")]
    MemoryExhausted { detail: String },

    /// File I/O failure (parquet, file system).
    #[error("I/O error: {detail}")]
    Io { detail: String },

    /// Analysis cancelled by host via `cancel_analysis()` (Issue #1047).
    #[error("Analysis cancelled by host")]
    Cancelled,
}

impl DiscoveryError {
    /// Return the `DiscoveryErrorKind` for this typed error via pattern matching.
    pub fn error_kind(&self) -> DiscoveryErrorKind {
        match self {
            Self::GpuUnavailable { .. } => DiscoveryErrorKind::GpuPermanent,
            Self::GpuDeviceLost { .. } => DiscoveryErrorKind::GpuTransient,
            Self::InvalidInput { .. } => DiscoveryErrorKind::DataValidation,
            Self::Timeout { .. } => DiscoveryErrorKind::Timeout,
            Self::MemoryExhausted { .. } => DiscoveryErrorKind::MemoryExhausted,
            Self::Io { .. } => DiscoveryErrorKind::IoError,
            Self::Cancelled => DiscoveryErrorKind::Cancelled,
        }
    }
}

// ============================================================================
// Classify from anyhow::Error — downcast first, then fall back to strings
// ============================================================================

/// Classify an `anyhow::Error` by first attempting to downcast to a typed
/// `DiscoveryError`, then falling back to string-based classification.
pub fn classify_anyhow_error(err: &anyhow::Error) -> DiscoveryErrorKind {
    if let Some(discovery_err) = err.downcast_ref::<DiscoveryError>() {
        return discovery_err.error_kind();
    }
    classify_error(&err.to_string())
}

/// Return `(error_msg, error_kind, retryable)` from an `anyhow::Error`,
/// preferring typed downcast over string matching.
pub fn error_fields_from_anyhow(
    err: &anyhow::Error,
) -> (String, Option<DiscoveryErrorKind>, Option<bool>) {
    let kind = classify_anyhow_error(err);
    (err.to_string(), Some(kind), Some(kind.is_retryable()))
}

// ============================================================================
// String-based classification (backward-compatible fallback)
// ============================================================================

/// Classify an error message into a `DiscoveryErrorKind`.
///
/// Inspects the error string for known patterns from GPU (wgpu), parquet I/O,
/// deadline, and memory subsystems.
pub fn classify_error(error_msg: &str) -> DiscoveryErrorKind {
    let lower = error_msg.to_lowercase();

    // Host-requested cancellation (Issue #1047)
    if lower.contains("cancelled by host") || lower.contains("analysis cancelled") {
        return DiscoveryErrorKind::Cancelled;
    }

    // GPU transient errors (device lost, driver issues)
    if is_gpu_transient_pattern(&lower) {
        return DiscoveryErrorKind::GpuTransient;
    }

    // GPU permanent errors (no GPU, unsupported)
    if is_gpu_permanent_pattern(&lower) {
        return DiscoveryErrorKind::GpuPermanent;
    }

    // Timeout / deadline exceeded
    if is_timeout_pattern(&lower) {
        return DiscoveryErrorKind::Timeout;
    }

    // Memory exhaustion
    if is_memory_pattern(&lower) {
        return DiscoveryErrorKind::MemoryExhausted;
    }

    // Data validation errors
    if is_data_validation_pattern(&lower) {
        return DiscoveryErrorKind::DataValidation;
    }

    // I/O errors (parquet, file system)
    if is_io_pattern(&lower) {
        return DiscoveryErrorKind::IoError;
    }

    DiscoveryErrorKind::Unknown
}

fn is_gpu_transient_pattern(lower: &str) -> bool {
    lower.contains("device is lost")
        || lower.contains("device lost")
        || lower.contains("device was lost")
        || lower.contains("gpu device poll error")
        || lower.contains("command buffer")
        || lower.contains("too many command buffers")
        || lower.contains("gpu driver")
        || lower.contains("driver may be unresponsive")
        || lower.contains("device creation failed")
        || lower.contains("internal error in gpu")
}

fn is_gpu_permanent_pattern(lower: &str) -> bool {
    lower.contains("no gpu")
        || lower.contains("gpu unavailable")
        || lower.contains("gpu is required")
        || lower.contains("no compatible gpu")
        || lower.contains("unsupported gpu")
        || lower.contains("metal should always be available")
}

fn is_timeout_pattern(lower: &str) -> bool {
    lower.contains("deadline")
        || lower.contains("timed out")
        || lower.contains("timeout")
        || lower.contains("time limit")
        || lower.contains("exceeded time")
        || lower.contains("aborted after")
}

fn is_memory_pattern(lower: &str) -> bool {
    lower.contains("out of memory")
        || lower.contains("memory exhausted")
        || lower.contains("allocation failed")
        || lower.contains("insufficient memory")
        || lower.contains("memory pressure")
}

fn is_data_validation_pattern(lower: &str) -> bool {
    lower.contains("invalid input")
        || lower.contains("failed to parse")
        || lower.contains("missing field")
        || lower.contains("invalid utf-8")
        || lower.contains("null input")
        || lower.contains("invalid json")
        || lower.contains("deserialization")
        || lower.contains("deserialisation")
        || lower.contains("validation failed")
        || lower.contains("expected ") && (lower.contains("found") || lower.contains("but got"))
}

fn is_io_pattern(lower: &str) -> bool {
    lower.contains("parquet")
        || lower.contains("file not found")
        || lower.contains("permission denied")
        || lower.contains("no such file")
        || lower.contains("i/o error")
        || lower.contains("io error")
        || lower.contains("broken pipe")
        || lower.contains("disk full")
}

/// Classify a panic message as `InternalPanic`.
pub fn classify_panic() -> DiscoveryErrorKind {
    DiscoveryErrorKind::InternalPanic
}

/// Classify an error and return `(error_kind, retryable)` fields ready for
/// inclusion in an FFI response struct.
pub fn error_fields(error_msg: &str) -> (Option<DiscoveryErrorKind>, Option<bool>) {
    let kind = classify_error(error_msg);
    (Some(kind), Some(kind.is_retryable()))
}

/// Return `(error_kind, retryable)` fields for a caught panic.
pub fn panic_error_fields() -> (Option<DiscoveryErrorKind>, Option<bool>) {
    let kind = classify_panic();
    (Some(kind), Some(kind.is_retryable()))
}

/// Return `(None, None)` — no error classification for success responses.
pub fn no_error_fields() -> (Option<DiscoveryErrorKind>, Option<bool>) {
    (None, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_transient_device_lost() {
        assert_eq!(
            classify_error("Device is lost"),
            DiscoveryErrorKind::GpuTransient
        );
        assert!(classify_error("Device is lost").is_retryable());
    }

    #[test]
    fn test_gpu_transient_device_poll() {
        assert_eq!(
            classify_error("GPU device poll error (warm-up): device was lost"),
            DiscoveryErrorKind::GpuTransient
        );
    }

    #[test]
    fn test_gpu_permanent_no_gpu() {
        assert_eq!(
            classify_error("No GPU available on this system"),
            DiscoveryErrorKind::GpuPermanent
        );
        assert!(!classify_error("No GPU available on this system").is_retryable());
    }

    #[test]
    fn test_data_validation_parse_error() {
        assert_eq!(
            classify_error("Failed to parse input JSON: expected value"),
            DiscoveryErrorKind::DataValidation
        );
        assert!(!classify_error("Failed to parse input JSON").is_retryable());
    }

    #[test]
    fn test_data_validation_null_input() {
        assert_eq!(
            classify_error("Null input pointer"),
            DiscoveryErrorKind::DataValidation
        );
    }

    #[test]
    fn test_timeout_deadline() {
        assert_eq!(
            classify_error("Analysis deadline exceeded"),
            DiscoveryErrorKind::Timeout
        );
        assert!(classify_error("Analysis deadline exceeded").is_retryable());
    }

    #[test]
    fn test_timeout_timed_out() {
        assert_eq!(
            classify_error("Operation timed out after 30s"),
            DiscoveryErrorKind::Timeout
        );
    }

    #[test]
    fn test_memory_exhausted() {
        assert_eq!(
            classify_error("Out of memory allocating GPU buffer"),
            DiscoveryErrorKind::MemoryExhausted
        );
        assert!(classify_error("Out of memory").is_retryable());
    }

    #[test]
    fn test_io_parquet() {
        assert_eq!(
            classify_error("Failed to read parquet file"),
            DiscoveryErrorKind::IoError
        );
        assert!(classify_error("parquet error").is_retryable());
    }

    #[test]
    fn test_io_file_not_found() {
        assert_eq!(
            classify_error("File not found: /tmp/data.parquet"),
            DiscoveryErrorKind::IoError
        );
    }

    #[test]
    fn test_internal_panic() {
        let kind = classify_panic();
        assert_eq!(kind, DiscoveryErrorKind::InternalPanic);
        assert!(!kind.is_retryable());
    }

    #[test]
    fn test_unknown_error() {
        assert_eq!(
            classify_error("Something completely unexpected"),
            DiscoveryErrorKind::Unknown
        );
        assert!(!classify_error("Unknown situation").is_retryable());
    }

    #[test]
    fn test_case_insensitive_classification() {
        assert_eq!(
            classify_error("DEVICE IS LOST"),
            DiscoveryErrorKind::GpuTransient
        );
        assert_eq!(
            classify_error("OUT OF MEMORY"),
            DiscoveryErrorKind::MemoryExhausted
        );
    }

    #[test]
    fn test_invalid_utf8_is_data_validation() {
        assert_eq!(
            classify_error("Invalid UTF-8 in input"),
            DiscoveryErrorKind::DataValidation
        );
    }

    #[test]
    fn test_gpu_transient_command_buffer() {
        assert_eq!(
            classify_error("Too many command buffers in flight"),
            DiscoveryErrorKind::GpuTransient
        );
    }

    #[test]
    fn test_gpu_transient_driver_unresponsive() {
        assert_eq!(
            classify_error("GPU driver may be unresponsive"),
            DiscoveryErrorKind::GpuTransient
        );
    }

    #[test]
    fn test_serialise_error_kind_snake_case() {
        let json = serde_json::to_string(&DiscoveryErrorKind::GpuTransient).unwrap();
        assert_eq!(json, "\"gpu_transient\"");

        let json = serde_json::to_string(&DiscoveryErrorKind::DataValidation).unwrap();
        assert_eq!(json, "\"data_validation\"");

        let json = serde_json::to_string(&DiscoveryErrorKind::MemoryExhausted).unwrap();
        assert_eq!(json, "\"memory_exhausted\"");
    }
}
