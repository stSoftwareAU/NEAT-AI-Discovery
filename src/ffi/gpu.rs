//! GPU probe FFI entry points.

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
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    r#"{"success":false,"gpuAvailable":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"gpuAvailable":false,"error":"Failed to create output string"}"#;
                CString::new(error).unwrap().into_raw()
            }
        }
    }))
    .unwrap_or_else(|panic_info| {
        // If panic occurred, create a safe error response
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"gpuAvailable\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"gpuAvailable":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}
