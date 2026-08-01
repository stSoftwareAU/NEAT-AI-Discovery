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
// Memory usage (Issue #1027)
// ============================================================================

/// Return the current Rust-side heap allocation in bytes.
///
/// This function is designed for periodic polling (every 5–30 seconds) by the
/// Deno-side memory watchdog. It reads a single atomic counter maintained by
/// the tracking allocator, so overhead is negligible.
///
/// The returned value reflects memory allocated through Rust's global
/// allocator. It does **not** include V8/Deno heap usage — callers should
/// combine this with `Deno.memoryUsage().heapUsed` for total process memory.
#[unsafe(no_mangle)]
pub extern "C" fn discovery_memory_usage_bytes() -> u64 {
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        #[allow(clippy::cast_possible_truncation)]
        let bytes = crate::ALLOCATOR.allocated() as u64;
        bytes
    }))
    .unwrap_or(0)
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
// Discovery directory cleanup (Issue #1100)
// ============================================================================

/// Atomically clean up a discovery temp directory (Issue #1100).
///
/// Removes the entire directory tree in a single recursive call so that the
/// lock file is never absent while the directory still exists. If the directory
/// has already been removed by another actor, the response reports
/// `alreadyGone: true` with `success: true` (no error).
///
/// Input JSON:
/// ```json
/// { "tempDir": "/path/to/.discovery/abc123" }
/// ```
///
/// Output JSON:
/// ```json
/// { "success": true, "alreadyGone": false }
/// ```
///
/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cleanup_discovery_dir(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::CStr;
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
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

        let input: CleanupDiscoveryDirInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let typed = DiscoveryError::InvalidInput {
                    detail: format!("Failed to parse input JSON: {e}"),
                };
                let kind = typed.error_kind();
                let output = CleanupDiscoveryDirOutput {
                    success: false,
                    already_gone: None,
                    error: Some(typed.to_string()),
                    error_kind: Some(kind),
                    retryable: Some(kind.is_retryable()),
                };
                return to_ffi_json(&output);
            }
        };

        let output = match crate::discovery_cleanup::cleanup_discovery_dir(&input.temp_dir) {
            Ok(outcome) => {
                let is_already_gone =
                    outcome == crate::discovery_cleanup::CleanupOutcome::AlreadyGone;
                let (error_kind, retryable) = no_error_fields();
                CleanupDiscoveryDirOutput {
                    success: true,
                    already_gone: Some(is_already_gone),
                    error: None,
                    error_kind,
                    retryable,
                }
            }
            // A path rejected by the discovery-directory guard (Issue #1866)
            // is a caller bug, not a transient fault: report it as
            // non-retryable `invalid_input` so hosts do not retry it.
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => {
                let typed = DiscoveryError::InvalidInput {
                    detail: e.to_string(),
                };
                let kind = typed.error_kind();
                CleanupDiscoveryDirOutput {
                    success: false,
                    already_gone: None,
                    error: Some(typed.to_string()),
                    error_kind: Some(kind),
                    retryable: Some(kind.is_retryable()),
                }
            }
            Err(e) => {
                let (err_msg, error_kind, retryable) =
                    error_fields_from_anyhow(&anyhow::anyhow!(e));
                CleanupDiscoveryDirOutput {
                    success: false,
                    already_gone: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                }
            }
        };

        to_ffi_json(&output)
    }))
    .unwrap_or_else(panic_to_ffi_json)
}

/// Scan a base directory for orphaned discovery directories and remove them
/// (Issue #1100).
///
/// A subdirectory is considered orphaned when it has no `discovery.lock` file.
/// `NotFound` errors are suppressed because the async cleanup actor may have
/// removed the directory between the orphan check and the removal call.
///
/// Input JSON:
/// ```json
/// { "baseDir": "/path/to/.discovery" }
/// ```
///
/// Output JSON:
/// ```json
/// { "success": true, "removed": 2, "alreadyGone": 0, "removalErrors": [] }
/// ```
///
/// # Safety
///
/// - `input_json` must be a valid, non-null pointer to a null-terminated C
///   string containing valid UTF-8 JSON.
/// - The returned pointer must be freed using `free_discovery_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn clean_orphaned_discovery_dirs(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::CStr;
    use std::panic;

    panic::catch_unwind(panic::AssertUnwindSafe(|| {
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

        let input: CleanOrphanedDirsInput = match serde_json::from_str(input_str) {
            Ok(input) => input,
            Err(e) => {
                let typed = DiscoveryError::InvalidInput {
                    detail: format!("Failed to parse input JSON: {e}"),
                };
                let kind = typed.error_kind();
                let output = CleanOrphanedDirsOutput {
                    success: false,
                    removed: None,
                    already_gone: None,
                    removal_errors: None,
                    error: Some(typed.to_string()),
                    error_kind: Some(kind),
                    retryable: Some(kind.is_retryable()),
                };
                return to_ffi_json(&output);
            }
        };

        let output = match crate::discovery_cleanup::clean_orphaned_discovery_dirs(&input.base_dir)
        {
            Ok(result) => {
                let removal_errors = if result.errors.is_empty() {
                    None
                } else {
                    Some(result.errors)
                };
                let (error_kind, retryable) = no_error_fields();
                CleanOrphanedDirsOutput {
                    success: true,
                    removed: Some(result.removed),
                    already_gone: Some(result.already_gone),
                    removal_errors,
                    error: None,
                    error_kind,
                    retryable,
                }
            }
            Err(e) => {
                let (err_msg, error_kind, retryable) =
                    error_fields_from_anyhow(&anyhow::anyhow!(e));
                CleanOrphanedDirsOutput {
                    success: false,
                    removed: None,
                    already_gone: None,
                    removal_errors: None,
                    error: Some(err_msg),
                    error_kind,
                    retryable,
                }
            }
        };

        to_ffi_json(&output)
    }))
    .unwrap_or_else(panic_to_ffi_json)
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
