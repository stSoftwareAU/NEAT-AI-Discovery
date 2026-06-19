//! Analysis FFI entry points — `rank_focus_neurons`, `analyze_parallel`,
//! `cancel_analysis`, and `reset_cancellation`.

use super::helpers::{ffi_error_literal, panic_to_ffi_json, to_ffi_json};
use crate::ffi_types::*;
use crate::log_version_once;

// ============================================================================
// Cancellation control (Issue #1047)
// ============================================================================

/// Signal in-flight analysis to stop gracefully.
///
/// Call this from the host process when SIGTERM is received. The Rust
/// analysis pipeline checks the flag at every deadline-check point and
/// at parquet batch boundaries, returning a distinguishable cancellation
/// result instead of an error.
///
/// # Safety
///
/// This function is safe to call from any thread at any time (the flag
/// is an `AtomicBool`). No pointer arguments.
#[unsafe(no_mangle)]
pub extern "C" fn cancel_analysis() {
    crate::cancellation::request_cancellation();
}

/// Signal in-flight analysis to stop due to CRITICAL memory pressure (Issue #1099).
///
/// Call this from the host process when the memory monitor detects CRITICAL
/// memory pressure (e.g., ≥85% heap usage). This sets both the general
/// cancellation flag and the memory-pressure-specific flag, so the analysis
/// pipeline can report the specific reason and the host can take additional
/// recovery actions (e.g., clearing WASM caches, evicting discovery buffers).
///
/// # Safety
///
/// This function is safe to call from any thread at any time (the flags
/// are `AtomicBool`). No pointer arguments.
#[unsafe(no_mangle)]
pub extern "C" fn cancel_analysis_memory_pressure() {
    crate::cancellation::request_cancellation_memory_pressure();
}

/// Clear a previous cancellation request.
///
/// The analysis pipeline calls this automatically at the start of each
/// `analyze_parallel` / `rank_focus_neurons` invocation, but the host
/// may also call it explicitly.
///
/// # Safety
///
/// No pointer arguments; safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn reset_cancellation() {
    crate::cancellation::reset_cancellation();
}

// ============================================================================
// Analysis lifecycle tracking (Issue #1048)
// ============================================================================

/// Check whether any analysis invocation is currently in-flight.
///
/// The host process must call this before deleting the parquet temp
/// directory. If it returns `true` (non-zero), the host should either:
/// - Wait for the analysis FFI call to return, **or**
/// - Call `cancel_analysis()` first and then wait.
///
/// Returns `1` if at least one analysis is active, `0` otherwise.
///
/// # Safety
///
/// No pointer arguments; safe to call from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn is_analysis_active() -> i32 {
    if crate::cancellation::is_analysis_active() {
        1
    } else {
        0
    }
}

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
                    focus_selection: None,
                    removal_candidates: None,
                    constant_neuron_removals: None,
                    max_output_error: None,
                    processed_neurons: None,
                    total_neurons: None,
                    duration_ms: None,
                    rejection_breakdown: None,
                    loading_mode: None,
                    lazy_reason: None,
                    budget_mb: None,
                    projected_mb: None,
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
                    cancelled: None,
                    memory_pressure_cancelled: None,
                    environmentally_disabled: None,
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
