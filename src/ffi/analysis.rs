//! Analysis FFI entry points — `rank_focus_neurons` and `analyze_parallel`.

use super::helpers::{ffi_error_literal, panic_to_ffi_json, to_ffi_json};
use crate::ffi_types::*;
use crate::log_version_once;

// ============================================================================
// Rank focus neurons
// ============================================================================

/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rank_focus_neurons(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        // SAFETY: caller must provide a valid, non-null pointer to a
        // null-terminated C string. We validate null and UTF-8 before use.
        let input_str = unsafe {
            if input_json.is_null() {
                return ffi_error_literal(r#"{"success":false,"error":"Null input pointer"}"#);
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    return ffi_error_literal(
                        r#"{"success":false,"error":"Invalid UTF-8 in input"}"#,
                    );
                }
            }
        };

        let json_result = match crate::rank_focus_neurons_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = RankFocusNeuronsOutput {
                    success: false,
                    schema_version: SCHEMA_VERSION.to_string(),
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
                return to_ffi_json(&output);
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                ffi_error_literal(r#"{"success":false,"error":"Failed to create output string"}"#)
            }
        }
    }))
    .unwrap_or_else(panic_to_ffi_json)
}

// ============================================================================
// Analyse parallel
// ============================================================================

/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn analyze_parallel(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        // SAFETY: caller must provide a valid, non-null pointer to a
        // null-terminated C string. We validate null and UTF-8 before use.
        let input_str = unsafe {
            if input_json.is_null() {
                return ffi_error_literal(r#"{"success":false,"error":"Null input pointer"}"#);
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    return ffi_error_literal(
                        r#"{"success":false,"error":"Invalid UTF-8 in input"}"#,
                    );
                }
            }
        };

        let json_result = match crate::analyze_parallel_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = AnalyzeParallelOutput {
                    success: false,
                    schema_version: SCHEMA_VERSION.to_string(),
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
                    module_outcome_tracker: None,
                    memory_budget_exceeded: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                return to_ffi_json(&output);
            }
        };

        match CString::new(json_result) {
            Ok(c_string) => c_string.into_raw(),
            Err(_) => {
                ffi_error_literal(r#"{"success":false,"error":"Failed to create output string"}"#)
            }
        }
    }))
    .unwrap_or_else(panic_to_ffi_json)
}
