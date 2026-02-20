//! Analysis FFI entry points — `rank_focus_neurons` and `analyze_parallel`.

use crate::ffi_types::*;
use crate::log_version_once;

// ============================================================================
// Rank focus neurons
// ============================================================================

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn rank_focus_neurons(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
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

        let json_result = match crate::rank_focus_neurons_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = RankFocusNeuronsOutput {
                    success: false,
                    neurons: None,
                    removal_candidates: None,
                    constant_neuron_removals: None,
                    max_output_error: None,
                    processed_neurons: None,
                    total_neurons: None,
                    duration_ms: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    r#"{"success":false,"error":"Failed to serialize output"}"#.to_string()
                })
            }
        };

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
// Analyze parallel
// ============================================================================

#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[unsafe(no_mangle)]
pub extern "C" fn analyze_parallel(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
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

        let json_result = match crate::analyze_parallel_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = AnalyzeParallelOutput {
                    success: false,
                    helpful_synapses: None,
                    harmful_synapses: None,
                    synapse_diagnostics: None,
                    synapse_gpu_used: None,
                    synapse_metadata: None,
                    helpful_neurons: None,
                    synapse_weight_updates: None,
                    coordinated_structural_candidates: None,
                    candidate_clusters: None,
                    neuron_diagnostics: None,
                    neuron_gpu_used: None,
                    neuron_metadata: None,
                    neuron_fingerprints: None,
                    fingerprint_cache_hits: None,
                    fingerprint_cache_misses: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                serde_json::to_string(&output).unwrap_or_else(|_| {
                    r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
                })
            }
        };

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
