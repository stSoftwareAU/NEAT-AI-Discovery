//! Internal business-logic functions for GPU probe FFI entry points.

use anyhow::Result;

use crate::analysis;
use crate::ffi_types::*;

pub fn check_gpu_available_internal() -> Result<String> {
    let result = analysis::GpuAnalyzer::check_gpu_availability();

    // On macOS, missing GPU is an error (Metal should always work).
    // On Linux, missing GPU gracefully disables discovery (common on headless servers).
    let output = if result.is_error {
        let err_msg = "GPU required but not available".to_string();
        let (error_kind, retryable) = error_fields(&err_msg);
        CheckGpuOutput {
            success: false,
            gpu_available: false,
            reason: result.reason,
            error: Some(err_msg),
            error_kind,
            retryable,
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
        error: None,
        error_kind,
        retryable,
    };
    Ok(serde_json::to_string(&output)?)
}
