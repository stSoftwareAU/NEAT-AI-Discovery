//! Internal business-logic functions for GPU probe FFI entry points.

use anyhow::Result;

use crate::analysis;
use crate::ffi_types::*;

pub fn check_gpu_available_internal() -> Result<String> {
    let result = analysis::GpuAnalyzer::check_gpu_availability();

    // On macOS, missing GPU is an error (Metal should always work).
    // On Linux, missing GPU gracefully disables discovery (common on headless servers).
    let output = if result.is_error {
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
    } else {
        let (error_kind, retryable) = no_error_fields();
        CheckGpuOutput {
            success: true,
            gpu_available: result.available,
            reason: result.reason,
            error: None,
            error_kind,
            retryable,
        }
    };
    Ok(serde_json::to_string(&output)?)
}

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
