//! Issue #2045 — Shared null-pointer / invalid-UTF-8 input guard.
//!
//! Every FFI entry point taking a `*const c_char` must reject a null pointer
//! and reject invalid UTF-8 with the same JSON error response. The guard now
//! lives in one helper; these tests pin the observable responses of all
//! thirteen entry points so the shared helper cannot silently change them.

use std::ffi::{CStr, CString};

/// Every FFI entry point that accepts a `*const c_char` input pointer.
type InputEntryPoint = unsafe extern "C" fn(*const std::ffi::c_char) -> *mut std::ffi::c_char;

/// The twelve entry points whose error responses carry only `success` and
/// `error`, plus `get_calibration_summary`, which also carries
/// `calibrationSummary`.
fn entry_points() -> Vec<(&'static str, InputEntryPoint)> {
    use neat_ai_discovery::ffi;
    vec![
        ("rank_focus_neurons", ffi::rank_focus_neurons as _),
        ("analyze_parallel", ffi::analyze_parallel as _),
        ("record_discovery", ffi::record_discovery as _),
        ("start_discovery_session", ffi::start_discovery_session as _),
        (
            "append_discovery_records",
            ffi::append_discovery_records as _,
        ),
        (
            "finish_discovery_session",
            ffi::finish_discovery_session as _,
        ),
        (
            "cancel_discovery_session",
            ffi::cancel_discovery_session as _,
        ),
        ("merge_discovery_parquet", ffi::merge_discovery_parquet as _),
        (
            "read_discovery_records_ffi",
            ffi::read_discovery_records_ffi as _,
        ),
        (
            "export_visualisation_snapshot",
            ffi::export_visualisation_snapshot as _,
        ),
        ("get_calibration_summary", ffi::get_calibration_summary as _),
        ("cleanup_discovery_dir", ffi::cleanup_discovery_dir as _),
        (
            "clean_orphaned_discovery_dirs",
            ffi::clean_orphaned_discovery_dirs as _,
        ),
    ]
}

/// Call an entry point, read the response as an owned `String`, and free the
/// FFI allocation.
fn call_and_read(entry: InputEntryPoint, input: *const std::ffi::c_char) -> String {
    // SAFETY: `input` is either null or a valid NUL-terminated C string owned
    // by the caller for the duration of the call.
    let ptr = unsafe { entry(input) };
    assert!(!ptr.is_null(), "entry point returned a null response");
    // SAFETY: `ptr` was allocated by `CString::into_raw` inside the entry point.
    let response = unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: `ptr` came from an FFI entry point and is freed exactly once.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(ptr) };
    response
}

/// The extra response-shape fields each entry point's error response carries
/// beyond `success` and `error`.
fn expects_calibration_summary(name: &str) -> bool {
    name == "get_calibration_summary"
}

fn assert_guard_response(name: &str, response: &str, expected_error: &str) {
    let parsed: serde_json::Value = serde_json::from_str(response)
        .unwrap_or_else(|e| panic!("{name}: response must be valid JSON ({e}): {response}"));
    assert_eq!(
        parsed["success"], false,
        "{name}: guard response must report success=false: {response}"
    );
    assert_eq!(
        parsed["error"], expected_error,
        "{name}: unexpected guard error message: {response}"
    );
    if expects_calibration_summary(name) {
        assert_eq!(
            parsed["calibrationSummary"],
            serde_json::json!([]),
            "{name}: response shape must keep an empty calibrationSummary: {response}"
        );
    } else {
        assert!(
            parsed.get("calibrationSummary").is_none(),
            "{name}: response shape must not gain a calibrationSummary field: {response}"
        );
    }
}

#[test]
fn every_entry_point_rejects_a_null_input_pointer() {
    for (name, entry) in entry_points() {
        let response = call_and_read(entry, std::ptr::null());
        assert_guard_response(name, &response, "Null input pointer");
    }
}

#[test]
fn every_entry_point_rejects_invalid_utf8_input() {
    // Lone continuation bytes — a NUL-free byte sequence that is not UTF-8.
    let invalid = CString::new(vec![0xffu8, 0xfe, 0x7b]).expect("no interior NUL");
    for (name, entry) in entry_points() {
        let response = call_and_read(entry, invalid.as_ptr());
        assert_guard_response(name, &response, "Invalid UTF-8 in input");
    }
}

#[test]
fn valid_utf8_input_passes_the_guard_to_the_entry_point() {
    use neat_ai_discovery::ffi;

    // Well-formed UTF-8 that is not valid input: the guard must let it through
    // so the entry point produces its own parse error, never a guard message.
    // Only read-only entry points are exercised here — the guard is shared, so
    // one pass-through per response shape is enough.
    let input =
        CString::new(r#"{"unexpected":"\u00fcn\u00efc\u00f6d\u00e9"}"#).expect("no interior NUL");
    let read_only: Vec<(&str, InputEntryPoint)> = vec![
        (
            "read_discovery_records_ffi",
            ffi::read_discovery_records_ffi as _,
        ),
        ("get_calibration_summary", ffi::get_calibration_summary as _),
    ];
    for (name, entry) in read_only {
        let response = call_and_read(entry, input.as_ptr());
        let parsed: serde_json::Value = serde_json::from_str(&response)
            .unwrap_or_else(|e| panic!("{name}: response must be valid JSON ({e}): {response}"));
        let error = parsed["error"].as_str().unwrap_or_default();
        assert!(
            error != "Null input pointer" && error != "Invalid UTF-8 in input",
            "{name}: valid UTF-8 input must not trip the input guard: {response}"
        );
    }
}
