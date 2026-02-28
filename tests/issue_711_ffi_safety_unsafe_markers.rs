//! Issue #711 — FFI safety: verify unsafe markers and null-pointer handling.
//!
//! These tests verify that all FFI functions that accept raw pointers handle
//! null pointers gracefully by returning a well-formed JSON error response
//! with `success: false`.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Parse the JSON response from an FFI function and assert it contains
/// `"success": false` with an error message, then free the result.
fn assert_null_pointer_error(ptr: *mut c_char) {
    assert!(!ptr.is_null(), "FFI function returned null for null input");
    // SAFETY: we just confirmed the pointer is non-null, and the FFI functions
    // always return valid null-terminated C strings.
    let response = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("response should be valid UTF-8");
    let parsed: serde_json::Value =
        serde_json::from_str(response).expect("response should be valid JSON");
    assert_eq!(
        parsed["success"], false,
        "null input should yield success=false, got: {response}"
    );
    assert!(
        parsed["error"].as_str().is_some(),
        "null input should yield an error message, got: {response}"
    );
    // SAFETY: pointer was allocated by the FFI function via CString::into_raw.
    unsafe {
        neat_ai_discovery::ffi::free_discovery_result(ptr);
    }
}

/// Parse the JSON response and assert it contains `"success": false`, then
/// free the result.
fn assert_error_response(ptr: *mut c_char) {
    assert!(!ptr.is_null(), "FFI function returned null");
    // SAFETY: we just confirmed the pointer is non-null, and the FFI functions
    // always return valid null-terminated C strings.
    let response = unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .expect("response should be valid UTF-8");
    let parsed: serde_json::Value =
        serde_json::from_str(response).expect("response should be valid JSON");
    assert_eq!(
        parsed["success"], false,
        "invalid input should yield success=false, got: {response}"
    );
    // SAFETY: pointer was allocated by the FFI function via CString::into_raw.
    unsafe {
        neat_ai_discovery::ffi::free_discovery_result(ptr);
    }
}

// ============================================================================
// Null pointer tests — each FFI function must handle null gracefully
// ============================================================================

#[test]
fn rank_focus_neurons_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::rank_focus_neurons(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn analyze_parallel_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::analyze_parallel(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn record_discovery_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::record_discovery(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn start_discovery_session_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::start_discovery_session(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn append_discovery_records_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::append_discovery_records(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn finish_discovery_session_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::finish_discovery_session(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn cancel_discovery_session_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::cancel_discovery_session(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn merge_discovery_parquet_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::merge_discovery_parquet(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn read_discovery_records_ffi_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::read_discovery_records_ffi(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn export_visualisation_snapshot_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::export_visualisation_snapshot(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn get_calibration_summary_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully.
    let result = unsafe { neat_ai_discovery::ffi::get_calibration_summary(std::ptr::null()) };
    assert_null_pointer_error(result);
}

#[test]
fn free_discovery_result_handles_null_pointer() {
    // SAFETY: testing that null input is handled gracefully (no-op).
    unsafe {
        neat_ai_discovery::ffi::free_discovery_result(std::ptr::null_mut());
    }
}

// ============================================================================
// Invalid JSON input tests — verify graceful error handling
// ============================================================================

#[test]
fn rank_focus_neurons_handles_invalid_json() {
    let input = CString::new("not valid json").unwrap();
    // SAFETY: input is a valid null-terminated C string.
    let result = unsafe { neat_ai_discovery::ffi::rank_focus_neurons(input.as_ptr()) };
    assert_error_response(result);
}

#[test]
fn analyze_parallel_handles_invalid_json() {
    let input = CString::new("not valid json").unwrap();
    // SAFETY: input is a valid null-terminated C string.
    let result = unsafe { neat_ai_discovery::ffi::analyze_parallel(input.as_ptr()) };
    assert_error_response(result);
}

#[test]
fn merge_discovery_parquet_handles_invalid_json() {
    let input = CString::new("not valid json").unwrap();
    // SAFETY: input is a valid null-terminated C string.
    let result = unsafe { neat_ai_discovery::ffi::merge_discovery_parquet(input.as_ptr()) };
    assert_error_response(result);
}

#[test]
fn get_calibration_summary_handles_invalid_json() {
    let input = CString::new("not valid json").unwrap();
    // SAFETY: input is a valid null-terminated C string.
    let result = unsafe { neat_ai_discovery::ffi::get_calibration_summary(input.as_ptr()) };
    assert_error_response(result);
}
