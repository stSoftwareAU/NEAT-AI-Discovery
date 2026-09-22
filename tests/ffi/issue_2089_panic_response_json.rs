//! Issue #2089 (security sweep chunk 2) — the FFI panic response must be
//! valid JSON.
//!
//! `panic_to_ffi_json` is the last-resort error channel of every `extern "C"`
//! entry point: `catch_unwind` hands it the caught payload and its output is
//! what the Deno host receives. It used to hand-roll the JSON string and escape
//! only `\` and `"`, so any payload carrying a control character produced a
//! response the host cannot parse — and a newline is exactly what every
//! `assert!` / `assert_eq!` panic message carries. The documented
//! `{"success":false,"error":…}` contract that every caller branches on
//! silently degraded into unparsable text at the one moment it matters.

use std::any::Any;
use std::ffi::CString;

use neat_ai_discovery::ffi::helpers::panic_to_ffi_json;

/// Drive a panic payload through the FFI panic formatter and read the response
/// back as an owned string, reclaiming the allocation exactly once.
fn panic_response(payload: impl Any + Send) -> String {
    let boxed: Box<dyn Any + Send> = Box::new(payload);
    let ptr = panic_to_ffi_json(boxed);
    assert!(!ptr.is_null(), "panic_to_ffi_json must never return null");
    // SAFETY: `ptr` was produced by `CString::into_raw` inside
    // `panic_to_ffi_json` and is reclaimed here exactly once.
    let owned = unsafe { CString::from_raw(ptr) };
    owned
        .to_str()
        .expect("FFI response must be valid UTF-8")
        .to_string()
}

/// Parse the response, failing with the raw bytes so a malformed payload is
/// legible rather than a bare serde error.
fn parse(response: &str) -> serde_json::Value {
    serde_json::from_str(response).unwrap_or_else(|e| {
        panic!("FFI panic response must be valid JSON ({e}): {response:?}");
    })
}

/// The realistic trigger: a standard `assert_eq!` failure message. Its
/// embedded newlines made the hand-rolled response unparsable.
#[test]
fn assertion_style_panic_message_yields_parsable_json() {
    let message = "assertion `left == right` failed\n  left: 1\n right: 2";
    let response = panic_response(message.to_string());
    let json = parse(&response);

    assert_eq!(json["success"], false);
    let error = json["error"].as_str().expect("error must be a string");
    assert!(
        error.contains("left: 1") && error.contains("right: 2"),
        "the panic message must survive escaping intact: {error:?}"
    );
}

/// Every control character a payload can carry must be escaped, not emitted
/// raw — tab, carriage return, form feed and a bare C0 byte alike.
#[test]
fn control_characters_in_a_panic_payload_are_escaped() {
    let message = "tab\there\rreturn\u{000c}form\u{0001}c0-byte\nnewline";
    let response = panic_response(message.to_string());
    let json = parse(&response);

    assert_eq!(json["success"], false);
    let error = json["error"].as_str().expect("error must be a string");
    assert!(
        error.ends_with(message),
        "the payload must round-trip byte-for-byte through the escape: {error:?}"
    );
}

/// A `&'static str` payload (`panic!("literal")`) takes the other downcast arm
/// and must be escaped identically.
#[test]
fn static_str_payload_with_a_newline_yields_parsable_json() {
    let response = panic_response("first line\nsecond line");
    let json = parse(&response);

    assert_eq!(json["success"], false);
    assert!(
        json["error"]
            .as_str()
            .expect("error must be a string")
            .ends_with("first line\nsecond line")
    );
}

/// Quotes and backslashes kept working — the fix must not regress the escaping
/// the hand-rolled formatter did get right.
#[test]
fn quotes_and_backslashes_still_round_trip() {
    let message = r#"path "C:\temp\x" not found"#;
    let response = panic_response(message.to_string());
    let json = parse(&response);

    assert_eq!(json["success"], false);
    assert!(
        json["error"]
            .as_str()
            .expect("error must be a string")
            .ends_with(message)
    );
}

/// An oversized payload is still truncated — and the truncated result is still
/// parsable, including when the truncation lands next to a control character.
#[test]
fn oversized_payload_is_truncated_and_still_parsable() {
    let message = format!("{}\ntail", "line\n".repeat(4096));
    let response = panic_response(message);
    let json = parse(&response);

    assert_eq!(json["success"], false);
    let error = json["error"].as_str().expect("error must be a string");
    assert!(
        error.len() < 16_384,
        "the embedded payload must stay bounded, got {} bytes",
        error.len()
    );
}

/// A payload of an unexpected type still produces a well-formed response.
#[test]
fn unknown_payload_type_yields_parsable_json() {
    let response = panic_response(42_i32);
    let json = parse(&response);

    assert_eq!(json["success"], false);
    assert!(
        json["error"]
            .as_str()
            .expect("error must be a string")
            .contains("Unknown panic")
    );
}
