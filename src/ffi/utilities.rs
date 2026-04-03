//! Utility FFI entry points — merge, read, export, and version.

use super::helpers::{ffi_error_literal, panic_to_ffi_json, to_ffi_json};
use crate::ffi_types::*;
use crate::log_version_once;

// ============================================================================
// Merge Parquet
// ============================================================================

/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn merge_discovery_parquet(
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

        let json_result = match crate::merge_discovery_parquet_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = MergeParquetOutput {
                    success: false,
                    output_file: None,
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
// Read discovery records
// ============================================================================

/// FFI export for reading discovery records.
///
/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn read_discovery_records_ffi(
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

        let json_result = match crate::read_discovery_records(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = ReadDiscoveryOutput {
                    success: false,
                    records: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                };
                return to_ffi_json(&output);
            }
        };

        // Return as C string - this is also protected by catch_unwind
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
// Export visualisation snapshot
// ============================================================================

/// Export a visualisation snapshot to JSON for debugging with NEAT-AI-Explore.
///
/// Input JSON:
/// ```json
/// {
///   "parquetFile": "/path/to/records.parquet",
///   "creature": { ... },
///   "outFile": "/path/to/snapshot.json",
///   "includePerSynapseSeries": true,
///   "includeReconstructionChecks": true,
///   "maxObs": null,
///   "topKWorstSamples": 20
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "outFile": "/path/to/snapshot.json",
///   "stats": { "obsCount": 1000, "neuronCount": 50, "synapseCount": 200, "outputCount": 1 }
/// }
/// ```
/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn export_visualisation_snapshot(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

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

        let json_result = match crate::export_visualisation_snapshot_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = ExportVisualisationSnapshotOutput {
                    success: false,
                    out_file: None,
                    stats: None,
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
// Calibration Summary (Issue #605)
// ============================================================================

/// FFI export for querying calibration summary from discovery history.
///
/// Input JSON:
/// ```json
/// {
///   "discoveryHistory": "<serialised DiscoveryHistory JSON string>"
/// }
/// ```
///
/// Output JSON:
/// ```json
/// {
///   "success": true,
///   "calibrationSummary": [
///     {
///       "moduleName": "saturation",
///       "candidateType": "addSynapse",
///       "sampleCount": 10,
///       "meanAbsoluteError": 0.02,
///       "bias": 0.01,
///       "calibrationFactor": 0.95
///     }
///   ]
/// }
/// ```
///
/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_calibration_summary(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        // SAFETY: caller must provide a valid, non-null pointer to a
        // null-terminated C string. We validate null and UTF-8 before use.
        let input_str = unsafe {
            if input_json.is_null() {
                return ffi_error_literal(
                    r#"{"success":false,"calibrationSummary":[],"error":"Null input pointer"}"#,
                );
            }
            match CStr::from_ptr(input_json).to_str() {
                Ok(s) => s,
                Err(_) => {
                    return ffi_error_literal(
                        r#"{"success":false,"calibrationSummary":[],"error":"Invalid UTF-8 in input"}"#,
                    );
                }
            }
        };

        let json_result = match crate::get_calibration_summary_internal(input_str) {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = crate::ffi_types::CalibrationSummaryOutput {
                    success: false,
                    calibration_summary: vec![],
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
                r#"{"success":false,"calibrationSummary":[],"error":"Failed to create output string"}"#,
            ),
        }
    }))
    .unwrap_or_else(panic_to_ffi_json)
}

// ============================================================================
// Library cleanup (Issue #994)
// ============================================================================

/// Shut down background threads spawned by the library (Issue #994).
///
/// Call this function before process exit to cleanly stop the deadlock-detector
/// and signal-handler threads. Without this call, those threads may keep the
/// host process alive after all FFI work has finished.
///
/// This function is safe to call multiple times and from any thread.
#[unsafe(no_mangle)]
pub extern "C" fn cleanup_discovery_lib() {
    use std::panic;

    let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        crate::debug::shutdown_debug_handlers();
    }));
}

// ============================================================================
// Library version
// ============================================================================

/// FFI export for querying the library version
///
/// Returns the version string that was embedded in the binary at compile time.
/// This allows callers to verify they're using the expected version.
///
/// # Safety
/// The returned pointer must be freed using `free_discovery_result`
#[unsafe(no_mangle)]
pub extern "C" fn get_library_version() -> *mut std::ffi::c_char {
    use std::ffi::CString;
    use std::panic;

    // Catch any panics to prevent unwinding across FFI boundary
    // This includes the final CString::new() conversion to catch any panics there
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        log_version_once();

        let json_result = match crate::get_library_version_internal() {
            Ok(json) => json,
            Err(e) => {
                let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
                let output = GetVersionOutput {
                    success: false,
                    version: String::new(),
                    schema_version: SCHEMA_VERSION.to_string(),
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
                r#"{"success":false,"version":"","error":"Failed to create output string"}"#,
            ),
        }
    }))
    .unwrap_or_else(panic_to_ffi_json)
}
