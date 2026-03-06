//! GPU probe FFI entry points.

use super::helpers::{ffi_error_literal, panic_to_ffi_json, to_ffi_json};
use crate::ffi_types::*;
use crate::log_version_once;

#[unsafe(no_mangle)]
pub extern "C" fn check_gpu_available() -> *mut std::ffi::c_char {
    use std::ffi::CString;
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let json_result = match crate::check_gpu_available_internal() {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = CheckGpuOutput {
                    success: false,
                    gpu_available: false,
                    reason: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                return to_ffi_json(&output);
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => ffi_error_literal(
                r#"{"success":false,"gpuAvailable":false,"error":"Failed to create output string"}"#,
            ),
        }
    }))
    .unwrap_or_else(panic_to_ffi_json)
}
