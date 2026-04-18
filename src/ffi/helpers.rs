//! Shared FFI helpers for safe JSON-to-C-string conversion (Issue #772).
//!
//! These helpers eliminate `unwrap()` calls at the FFI boundary by handling
//! serialisation failures and null-byte edge cases gracefully, returning a
//! JSON error response instead of panicking.

use serde::Serialize;
use std::ffi::CString;

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

/// Build an FFI-safe error response from a caught panic.
///
/// Extracts the panic message and returns a JSON error string as
/// `*mut c_char`. Never panics itself.
pub fn panic_to_ffi_json(panic_info: Box<dyn std::any::Any + Send>) -> *mut std::ffi::c_char {
    let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = panic_info.downcast_ref::<String>() {
        s.clone()
    } else {
        "Unknown panic".to_string()
    };
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
}
