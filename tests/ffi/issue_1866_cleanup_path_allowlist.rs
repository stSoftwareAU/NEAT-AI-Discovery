//! Issue #1866 — `cleanup_discovery_dir` must not recursively delete an
//! arbitrary caller-supplied path.
//!
//! The FFI entry point takes a `tempDir` string from the Deno host and passes
//! it to `fs::remove_dir_all`. These tests exercise the exported symbol end to
//! end and assert that a path outside the discovery-directory allowlist is
//! refused — with a non-retryable `data_validation` error — and that the
//! directory and its contents survive.

use std::ffi::{CStr, CString};
use std::fs::{self, File};

fn call_cleanup(temp_dir: &str) -> serde_json::Value {
    let input = serde_json::json!({ "tempDir": temp_dir }).to_string();
    let input_c = CString::new(input).expect("input contains no interior NUL");
    // SAFETY: `input_c` is a valid, null-terminated UTF-8 C string that
    // outlives the call.
    let response_ptr = unsafe { neat_ai_discovery::ffi::cleanup_discovery_dir(input_c.as_ptr()) };
    assert!(
        !response_ptr.is_null(),
        "FFI must return a non-null pointer"
    );
    // SAFETY: the pointer was just produced by the FFI call and is valid.
    let response = unsafe { CStr::from_ptr(response_ptr) }
        .to_string_lossy()
        .into_owned();
    // SAFETY: `response_ptr` came from `cleanup_discovery_dir`, matching the
    // contract for `free_discovery_result`.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(response_ptr) };
    serde_json::from_str(&response).expect("response must be valid JSON")
}

#[test]
fn cleanup_ffi_refuses_non_discovery_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let victim = temp.path().join("Documents");
    fs::create_dir(&victim).expect("create victim dir");
    let precious = victim.join("important.dat");
    File::create(&precious).expect("create victim file");

    let response = call_cleanup(victim.to_str().unwrap());

    assert_eq!(
        response["success"], false,
        "an unvalidated path must be refused: {response}"
    );
    assert_eq!(response["errorKind"], "data_validation", "{response}");
    assert_eq!(response["retryable"], false, "{response}");
    assert!(victim.exists(), "victim directory must survive");
    assert!(precious.exists(), "victim contents must survive");
}

#[test]
fn cleanup_ffi_removes_genuine_discovery_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).expect("create discovery root");
    let session = root.join("session-abc123");
    fs::create_dir(&session).expect("create session dir");
    File::create(session.join("discovery.lock")).expect("create lock file");

    let response = call_cleanup(session.to_str().unwrap());

    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["alreadyGone"], false, "{response}");
    assert!(!session.exists(), "session directory must be removed");
}

#[test]
fn cleanup_ffi_reports_already_gone_for_missing_directory() {
    let temp = tempfile::tempdir().expect("create temp dir");
    let missing = temp.path().join(".discovery").join("never-created");

    let response = call_cleanup(missing.to_str().unwrap());

    assert_eq!(response["success"], true, "{response}");
    assert_eq!(response["alreadyGone"], true, "{response}");
}
