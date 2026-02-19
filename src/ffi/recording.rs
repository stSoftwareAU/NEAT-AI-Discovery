//! Recording FFI entry points — single-call and streaming.

use crate::ffi_types::*;
use crate::{log_version_once, streaming};

// ============================================================================
// Recording — single-call
// ============================================================================

/// FFI export for recording discovery data
///
/// # Safety
/// This function is unsafe because it deals with raw C strings.
/// The caller must ensure:
/// - input_json is a valid null-terminated C string
/// - The returned pointer is freed using free_discovery_result
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn record_discovery(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        // Read input C string
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        // Call the Rust function with original string
        let json_result = match crate::record_discovery_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                // Properly serialize error message to avoid JSON injection issues
                let err_msg = e.to_string();
                let (error_kind, retryable) = error_fields(&err_msg);
                let output = RecordDiscoveryOutput {
                    success: false,
                    temp_dir: None,
                    file: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    // Fallback if serialization fails (shouldn't happen)
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

        // Return as C string - this is also protected by catch_unwind
        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                let error = r#"{"success":false,"error":"Failed to create output string"}"#;
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
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        // This should never fail, but if it does, we return null pointer
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

// ============================================================================
// Streaming Recording API — FFI Entry Points
// ============================================================================
// These functions support incremental recording to avoid JavaScript
// "Invalid string length" errors when serialising large datasets.
//
// Usage pattern from TypeScript:
//   1. start_discovery_session() → returns session_id
//   2. Loop: append_discovery_records() as data accumulates
//   3. finish_discovery_session() → finalises Parquet file
//
// Benefits:
//   - Each append call is small enough to serialise (e.g., ~50MB)
//   - Unlimited total sample size
//   - Partial data preserved if process crashes

/// Start a new streaming discovery session.
///
/// Creates a Parquet file and returns a session ID for subsequent append/finish calls.
///
/// Input JSON:
/// ```json
/// {
///   "creature": { ... },
///   "tempDir": "/path/to/temp/dir"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "sessionId": "uuid-string"
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn start_discovery_session(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: StartSessionInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let err_msg = format!("Failed to parse input JSON: {e}");
                let (error_kind, retryable) = error_fields(&err_msg);
                let output = StartSessionOutput {
                    success: false,
                    session_id: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        let output = match streaming::start_session(input.creature, input.temp_dir) {
            Ok(session_id) => {
                let (error_kind, retryable) = no_error_fields();
                StartSessionOutput {
                    success: true,
                    session_id: Some(session_id),
                    error: None,
                    error_kind,
                    retryable,
                }
            }
            Err(e) => {
                let err_msg = e.to_string();
                let (error_kind, retryable) = error_fields(&err_msg);
                StartSessionOutput {
                    success: false,
                    session_id: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                }
            }
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Append records to an existing streaming session.
///
/// Input JSON:
/// ```json
/// {
///   "sessionId": "uuid-string",
///   "observations": [
///     {
///       "obsIndex": 0,
///       "neuronData": [{ "neuronUuid": "...", "activation": 0.5, "value": 0.4, "errors": [0.1] }],
///       "inputs": [0.1, 0.2, 0.3]
///     }
///   ]
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "recordsWritten": 42
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn append_discovery_records(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: AppendRecordsInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let err_msg = format!("Failed to parse input JSON: {e}");
                let (error_kind, retryable) = error_fields(&err_msg);
                let output = AppendRecordsOutput {
                    success: false,
                    records_written: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        // Convert observations to the internal format
        let batches: Vec<(u32, Vec<NeuronData>, Vec<f32>)> = input
            .observations
            .into_iter()
            .map(|obs| (obs.obs_index, obs.neuron_data, obs.inputs))
            .collect();

        let output = match streaming::append_records(&input.session_id, batches) {
            Ok(records_written) => {
                let (error_kind, retryable) = no_error_fields();
                AppendRecordsOutput {
                    success: true,
                    records_written: Some(records_written),
                    error: None,
                    error_kind,
                    retryable,
                }
            }
            Err(e) => {
                let err_msg = e.to_string();
                let (error_kind, retryable) = error_fields(&err_msg);
                AppendRecordsOutput {
                    success: false,
                    records_written: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                }
            }
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Finish a streaming session and finalise the Parquet file.
///
/// Input JSON:
/// ```json
/// {
///   "sessionId": "uuid-string"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "tempDir": "/path/to/temp/dir",
///   "file": "discovery_data.parquet",
///   "totalRecords": 12345
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn finish_discovery_session(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: FinishSessionInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let err_msg = format!("Failed to parse input JSON: {e}");
                let (error_kind, retryable) = error_fields(&err_msg);
                let output = FinishSessionOutput {
                    success: false,
                    temp_dir: None,
                    file: None,
                    total_records: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        let output = match streaming::finish_session(&input.session_id) {
            Ok((temp_dir, file, total_records)) => {
                let (error_kind, retryable) = no_error_fields();
                FinishSessionOutput {
                    success: true,
                    temp_dir: Some(temp_dir),
                    file: Some(file),
                    total_records: Some(total_records),
                    error: None,
                    error_kind,
                    retryable,
                }
            }
            Err(e) => {
                let err_msg = e.to_string();
                let (error_kind, retryable) = error_fields(&err_msg);
                FinishSessionOutput {
                    success: false,
                    temp_dir: None,
                    file: None,
                    total_records: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                }
            }
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}

/// Cancel a streaming session without finalising.
///
/// Use this to clean up if recording fails or is cancelled.
///
/// Input JSON:
/// ```json
/// {
///   "sessionId": "uuid-string"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true
/// }
/// ```
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn cancel_discovery_session(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let input_str = unsafe {
            if input_json.is_null() {
                let error = r#"{"success":false,"error":"Null input pointer"}"#;
                return CString::new(error).unwrap().into_raw();
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                    return CString::new(error).unwrap().into_raw();
                }
            }
        };

        let input: CancelSessionInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let err_msg = format!("Failed to parse input JSON: {e}");
                let (error_kind, retryable) = error_fields(&err_msg);
                let output = CancelSessionOutput {
                    success: false,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                let json = serde_json::to_string(&output).unwrap();
                return CString::new(json).unwrap().into_raw();
            }
        };

        let output = match streaming::cancel_session(&input.session_id) {
            Ok(()) => {
                let (error_kind, retryable) = no_error_fields();
                CancelSessionOutput {
                    success: true,
                    error: None,
                    error_kind,
                    retryable,
                }
            }
            Err(e) => {
                let err_msg = e.to_string();
                let (error_kind, retryable) = error_fields(&err_msg);
                CancelSessionOutput {
                    success: false,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                }
            }
        };

        let json = serde_json::to_string(&output).unwrap();
        CString::new(json).unwrap().into_raw()
    }))
    .unwrap_or_else(|panic_info| {
        let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = panic_info.downcast_ref::<String>() {
            s.clone()
        } else {
            "Unknown panic".to_string()
        };
        let error_json = format!(
            "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
        );
        CString::new(error_json)
            .unwrap_or_else(|_| {
                CString::new(r#"{"success":false,"error":"Failed to create panic error string"}"#)
                    .unwrap()
            })
            .into_raw()
    })
}
