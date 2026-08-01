//! Issue #1866 — the exported `cleanup_discovery_dir` FFI entry point must
//! reject caller paths that are not discovery directories.
//!
//! The guard lives in `discovery_cleanup::cleanup_discovery_dir`; these tests
//! confirm it is actually reached through the FFI boundary and surfaces as a
//! structured `success: false` response rather than a silent recursive delete.

use std::ffi::{CStr, CString};
use std::fs::{self, File};
use tempfile::TempDir;

/// Call the exported entry point and return its JSON response.
fn call_cleanup(input_json: &str) -> serde_json::Value {
    let input = CString::new(input_json).unwrap();
    // SAFETY: `input` is a valid null-terminated C string that outlives the
    // call, and the returned pointer is freed below via `free_discovery_result`.
    let raw = unsafe { neat_ai_discovery::ffi::cleanup_discovery_dir(input.as_ptr()) };
    assert!(!raw.is_null(), "FFI must never return a null pointer");
    // SAFETY: `raw` is a non-null pointer to a null-terminated C string
    // allocated by the library.
    let json = unsafe { CStr::from_ptr(raw) }.to_str().unwrap().to_string();
    // SAFETY: `raw` came from this library and is freed exactly once.
    unsafe { neat_ai_discovery::ffi::free_discovery_result(raw) };
    serde_json::from_str(&json).unwrap()
}

#[test]
fn ffi_cleanup_rejects_unrelated_directory() {
    let temp = TempDir::new().unwrap();
    let victim = temp.path().join("Documents");
    fs::create_dir(&victim).unwrap();
    let precious = victim.join("thesis.txt");
    File::create(&precious).unwrap();

    let response = call_cleanup(&format!(
        r#"{{"tempDir": {}}}"#,
        serde_json::to_string(victim.to_str().unwrap()).unwrap()
    ));

    assert_eq!(
        response["success"], false,
        "unrelated path must be refused: {response}"
    );
    // A rejected path is a caller bug, so it must not be advertised as
    // retryable — otherwise a host could loop on it forever.
    assert_eq!(response["errorKind"], "data_validation", "{response}");
    assert_eq!(response["retryable"], false, "{response}");
    assert!(victim.exists(), "unrelated directory must survive");
    assert!(precious.exists(), "unrelated files must survive");
}

#[test]
fn ffi_cleanup_removes_genuine_discovery_dir() {
    let temp = TempDir::new().unwrap();
    let root = temp.path().join(".discovery");
    fs::create_dir(&root).unwrap();
    let session = root.join("session-abc123");
    fs::create_dir(&session).unwrap();
    File::create(session.join("discovery.lock")).unwrap();

    let response = call_cleanup(&format!(
        r#"{{"tempDir": {}}}"#,
        serde_json::to_string(session.to_str().unwrap()).unwrap()
    ));

    assert_eq!(response["success"], true, "response: {response}");
    assert_eq!(response["alreadyGone"], false, "response: {response}");
    assert!(!session.exists());
}
