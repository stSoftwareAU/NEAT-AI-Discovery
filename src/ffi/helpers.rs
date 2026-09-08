//! Shared FFI helpers for safe JSON-to-C-string conversion (Issue #772).
//!
//! These helpers eliminate `unwrap()` calls at the FFI boundary by handling
//! serialisation failures and null-byte edge cases gracefully, returning a
//! JSON error response instead of panicking.

use serde::Serialize;
use std::ffi::{CStr, CString};

/// Serialise a value to JSON and return it as an FFI-safe `*mut c_char`.
///
/// On serialisation or `CString` creation failure, returns a JSON error
/// string instead of panicking. The caller must free the returned pointer
/// with `free_discovery_result`.
pub fn to_ffi_json<T: Serialize>(value: &T) -> *mut std::ffi::c_char {
    let json = match serde_json::to_string(value) {
        Ok(j) => j,
        Err(_) => {
            return ffi_error_literal(
                r#"{"success":false,"error":"Failed to serialise response"}"#,
            );
        }
    };
    match CString::new(json) {
        Ok(c) => c.into_raw(),
        Err(_) => ffi_error_literal(r#"{"success":false,"error":"Response contains null byte"}"#),
    }
}

/// Convert a static error string literal to an FFI-safe `*mut c_char`.
///
/// The input **must not** contain null bytes (compile-time string literals
/// that are valid JSON never do). If `CString::new` somehow fails, returns
/// a null pointer as a last resort — the Deno FFI layer treats null as an
/// empty string, which is safer than unwinding across the FFI boundary.
pub fn ffi_error_literal(msg: &str) -> *mut std::ffi::c_char {
    CString::new(msg).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Response-shape fields that `get_calibration_summary`'s error responses must
/// carry in addition to `success` and `error` (Issue #2045).
///
/// A JSON object-body fragment, comma-terminated so it composes directly into
/// the error response built by [`validate_c_str_input_with_fields`].
pub const CALIBRATION_SUMMARY_FIELDS: &str = r#""calibrationSummary":[],"#;

/// Build the input-guard error response for an entry point's response shape.
///
/// `extra_fields` is inserted verbatim between `success` and `error`; `message`
/// is one of this module's own literals, so neither needs JSON escaping.
fn input_guard_error(extra_fields: &str, message: &str) -> *mut std::ffi::c_char {
    ffi_error_literal(&format!(
        r#"{{"success":false,{extra_fields}"error":"{message}"}}"#
    ))
}

/// Validate a `*const c_char` FFI input pointer and borrow it as `&str`
/// (Issue #2045).
///
/// Rejects a null pointer and invalid UTF-8, returning the ready-made FFI
/// error pointer for the caller to return directly. This is the single
/// implementation of the guard every FFI entry point applies to its input.
///
/// # Safety
///
/// - `ptr` must be null, or a valid pointer to a null-terminated C string.
/// - The string must stay alive and unmodified for the lifetime of the
///   returned borrow.
pub unsafe fn validate_c_str_input<'a>(
    ptr: *const std::ffi::c_char,
) -> Result<&'a str, *mut std::ffi::c_char> {
    // SAFETY: the caller upholds this function's own pointer contract.
    unsafe { validate_c_str_input_with_fields(ptr, "") }
}

/// [`validate_c_str_input`] for an entry point whose error response shape
/// carries extra fields (e.g. [`CALIBRATION_SUMMARY_FIELDS`]).
///
/// # Safety
///
/// Same contract as [`validate_c_str_input`].
pub unsafe fn validate_c_str_input_with_fields<'a>(
    ptr: *const std::ffi::c_char,
    extra_fields: &str,
) -> Result<&'a str, *mut std::ffi::c_char> {
    if ptr.is_null() {
        return Err(input_guard_error(extra_fields, "Null input pointer"));
    }
    // SAFETY: `ptr` is non-null here, and the caller guarantees it points to a
    // null-terminated C string that outlives the returned borrow.
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(s) => Ok(s),
        Err(_) => Err(input_guard_error(extra_fields, "Invalid UTF-8 in input")),
    }
}

/// Maximum number of bytes of a panic message embedded in an FFI error
/// response before truncation (4 KiB). Bounds the worst-case allocation so the
/// "Never panics itself" path stays cheap even when a panic carries a large
/// payload (Issue #1365).
const MAX_PANIC_MSG_BYTES: usize = 4096;

/// Marker appended to a panic message that has been truncated.
const TRUNCATION_MARKER: &str = "… (truncated)";

/// Truncate `msg` to at most `MAX_PANIC_MSG_BYTES` bytes on a UTF-8 char
/// boundary, appending [`TRUNCATION_MARKER`] when truncation occurred.
///
/// Returns the message unchanged when it already fits within the cap. The
/// returned string never exceeds `MAX_PANIC_MSG_BYTES + TRUNCATION_MARKER.len()`
/// bytes and always lands on a char boundary, so it cannot split a multi-byte
/// UTF-8 sequence.
fn truncate_panic_msg(msg: &str) -> std::borrow::Cow<'_, str> {
    if msg.len() <= MAX_PANIC_MSG_BYTES {
        return std::borrow::Cow::Borrowed(msg);
    }
    // Walk back to the nearest char boundary at or below the cap.
    let mut boundary = MAX_PANIC_MSG_BYTES;
    while boundary > 0 && !msg.is_char_boundary(boundary) {
        boundary -= 1;
    }
    std::borrow::Cow::Owned(format!("{}{}", &msg[..boundary], TRUNCATION_MARKER))
}

/// Build an FFI-safe error response from a caught panic.
///
/// Extracts the panic message, truncates it to a fixed cap (see
/// [`MAX_PANIC_MSG_BYTES`]), and returns a JSON error string as
/// `*mut c_char`. Never panics itself.
pub fn panic_to_ffi_json(panic_info: Box<dyn std::any::Any + Send>) -> *mut std::ffi::c_char {
    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic_info.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic".to_string()
    };
    let msg = truncate_panic_msg(&msg);
    let error_json = format!(
        "{{\"success\":false,\"error\":\"Internal panic caught: {}\"}}",
        msg.replace('\\', "\\\\").replace('"', "\\\"")
    );
    match CString::new(error_json) {
        Ok(c) => c.into_raw(),
        Err(_) => ffi_error_literal(
            r#"{"success":false,"error":"Internal panic caught (message not representable)"}"#,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct TestOutput {
        success: bool,
        value: i32,
    }

    #[test]
    fn test_to_ffi_json_success() {
        let output = TestOutput {
            success: true,
            value: 42,
        };
        let ptr = to_ffi_json(&output);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        let json: serde_json::Value = serde_json::from_str(result.to_str().unwrap()).unwrap();
        assert_eq!(json["success"], true);
        assert_eq!(json["value"], 42);
    }

    #[test]
    fn test_ffi_error_literal_returns_valid_string() {
        let msg = r#"{"success":false,"error":"test"}"#;
        let ptr = ffi_error_literal(msg);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        assert_eq!(result.to_str().unwrap(), msg);
    }

    #[test]
    fn test_panic_to_ffi_json_with_string_message() {
        let panic_info: Box<dyn std::any::Any + Send> =
            Box::new("something went wrong".to_string());
        let ptr = panic_to_ffi_json(panic_info);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        let json: serde_json::Value = serde_json::from_str(result.to_str().unwrap()).unwrap();
        assert_eq!(json["success"], false);
        assert!(
            json["error"]
                .as_str()
                .unwrap()
                .contains("something went wrong")
        );
    }

    #[test]
    fn test_panic_to_ffi_json_with_str_message() {
        let panic_info: Box<dyn std::any::Any + Send> = Box::new("str panic");
        let ptr = panic_to_ffi_json(panic_info);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        let json: serde_json::Value = serde_json::from_str(result.to_str().unwrap()).unwrap();
        assert!(json["error"].as_str().unwrap().contains("str panic"));
    }

    #[test]
    fn test_panic_to_ffi_json_with_unknown_type() {
        let panic_info: Box<dyn std::any::Any + Send> = Box::new(42i32);
        let ptr = panic_to_ffi_json(panic_info);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        let json: serde_json::Value = serde_json::from_str(result.to_str().unwrap()).unwrap();
        assert!(json["error"].as_str().unwrap().contains("Unknown panic"));
    }

    #[test]
    fn test_panic_to_ffi_json_truncates_oversized_message() {
        // Panic message far larger than the cap.
        let huge = "A".repeat(MAX_PANIC_MSG_BYTES * 4);
        let panic_info: Box<dyn std::any::Any + Send> = Box::new(huge);
        let ptr = panic_to_ffi_json(panic_info);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        let raw = result.to_str().unwrap();
        // Still valid JSON.
        let json: serde_json::Value = serde_json::from_str(raw).unwrap();
        let error = json["error"].as_str().unwrap();
        // The embedded message is bounded to the cap plus the marker.
        let embedded_cap = MAX_PANIC_MSG_BYTES + TRUNCATION_MARKER.len();
        assert!(
            error.len() <= "Internal panic caught: ".len() + embedded_cap,
            "error field length {} exceeds bound",
            error.len()
        );
        assert!(error.ends_with(TRUNCATION_MARKER));
    }

    #[test]
    fn test_truncate_panic_msg_respects_char_boundary() {
        // Multi-byte char (3 bytes each) straddling the cap must not be split.
        let multibyte = "字".repeat(MAX_PANIC_MSG_BYTES); // 3 bytes each
        let truncated = truncate_panic_msg(&multibyte);
        // Must remain valid UTF-8 (guaranteed by &str) and end with the marker.
        assert!(truncated.ends_with(TRUNCATION_MARKER));
        // Truncation point lands on a char boundary at or below the cap.
        let body = truncated.strip_suffix(TRUNCATION_MARKER).unwrap();
        assert!(body.len() <= MAX_PANIC_MSG_BYTES);
        assert!(multibyte.is_char_boundary(body.len()));
    }

    #[test]
    fn test_truncate_panic_msg_short_message_unchanged() {
        let short = "small panic";
        assert_eq!(truncate_panic_msg(short), short);
    }

    #[test]
    fn test_panic_to_ffi_json_escapes_special_chars() {
        let panic_info: Box<dyn std::any::Any + Send> =
            Box::new("error with \"quotes\" and \\backslash".to_string());
        let ptr = panic_to_ffi_json(panic_info);
        assert!(!ptr.is_null());
        // SAFETY: we just created this pointer via CString::into_raw
        let result = unsafe { CString::from_raw(ptr) };
        // Should be valid JSON despite special characters
        let json: serde_json::Value = serde_json::from_str(result.to_str().unwrap()).unwrap();
        assert_eq!(json["success"], false);
    }

    // ========================================================================
    // validate_c_str_input — the shared null / UTF-8 input guard (Issue #2045)
    // ========================================================================

    /// Read an FFI error pointer back as JSON, freeing the allocation.
    fn error_json(ptr: *mut std::ffi::c_char) -> serde_json::Value {
        assert!(!ptr.is_null());
        // SAFETY: `ptr` was produced by `ffi_error_literal` via
        // `CString::into_raw` and is reclaimed exactly once here.
        let owned = unsafe { CString::from_raw(ptr) };
        serde_json::from_str(owned.to_str().unwrap()).unwrap()
    }

    #[test]
    fn test_validate_c_str_input_accepts_valid_utf8() {
        let input = CString::new(r#"{"a":"ü"}"#).unwrap();
        // SAFETY: `input` is a valid C string that outlives the borrow.
        let result = unsafe { validate_c_str_input(input.as_ptr()) };
        assert_eq!(result.unwrap(), r#"{"a":"ü"}"#);
    }

    #[test]
    fn test_validate_c_str_input_accepts_empty_string() {
        let input = CString::new("").unwrap();
        // SAFETY: `input` is a valid C string that outlives the borrow.
        let result = unsafe { validate_c_str_input(input.as_ptr()) };
        assert_eq!(result.unwrap(), "");
    }

    #[test]
    fn test_validate_c_str_input_rejects_null_pointer() {
        // SAFETY: a null pointer is an explicitly permitted input.
        let err = unsafe { validate_c_str_input(std::ptr::null()) }.unwrap_err();
        let json = error_json(err);
        assert_eq!(json["success"], false);
        assert_eq!(json["error"], "Null input pointer");
        assert!(json.get("calibrationSummary").is_none());
    }

    #[test]
    fn test_validate_c_str_input_rejects_invalid_utf8() {
        // Lone continuation bytes — not valid UTF-8, no interior NUL.
        let input = CString::new(vec![0xffu8, 0xfe]).unwrap();
        // SAFETY: `input` is a valid C string that outlives the call.
        let err = unsafe { validate_c_str_input(input.as_ptr()) }.unwrap_err();
        let json = error_json(err);
        assert_eq!(json["success"], false);
        assert_eq!(json["error"], "Invalid UTF-8 in input");
    }

    #[test]
    fn test_validate_c_str_input_with_fields_carries_extra_shape() {
        // SAFETY: a null pointer is an explicitly permitted input.
        let err = unsafe {
            validate_c_str_input_with_fields(std::ptr::null(), CALIBRATION_SUMMARY_FIELDS)
        }
        .unwrap_err();
        let json = error_json(err);
        assert_eq!(json["success"], false);
        assert_eq!(json["error"], "Null input pointer");
        assert_eq!(json["calibrationSummary"], serde_json::json!([]));

        let input = CString::new(vec![0xffu8, 0xfe]).unwrap();
        // SAFETY: `input` is a valid C string that outlives the call.
        let err =
            unsafe { validate_c_str_input_with_fields(input.as_ptr(), CALIBRATION_SUMMARY_FIELDS) }
                .unwrap_err();
        let json = error_json(err);
        assert_eq!(json["error"], "Invalid UTF-8 in input");
        assert_eq!(json["calibrationSummary"], serde_json::json!([]));
    }

    #[test]
    fn test_validate_c_str_input_with_fields_passes_valid_input_through() {
        let input = CString::new(r#"{"discoveryHistory":[]}"#).unwrap();
        // SAFETY: `input` is a valid C string that outlives the borrow.
        let result =
            unsafe { validate_c_str_input_with_fields(input.as_ptr(), CALIBRATION_SUMMARY_FIELDS) };
        assert_eq!(result.unwrap(), r#"{"discoveryHistory":[]}"#);
    }
}
