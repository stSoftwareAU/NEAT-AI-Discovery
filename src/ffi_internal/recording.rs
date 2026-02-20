//! Internal business-logic functions for recording FFI entry points.

use anyhow::Result;

use crate::ffi_types::*;
use crate::record;

/// Main entry point for recording discovery data
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS
/// Always returns a JSON string, even on error (with success=false)
///
/// This is the internal Rust function. For FFI, use `record_discovery`.
pub fn record_discovery_internal(input_json: &str) -> Result<String> {
    // Parse input JSON - if this fails, return JSON error
    let input: RecordDiscoveryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let err_msg = format!("Failed to parse input JSON: {e}");
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Process discovery data - if this fails, return JSON error
    let result = match record::record_discovery_data(&input) {
        Ok(result) => result,
        Err(e) => {
            let err_msg = e.to_string();
            let (error_kind, retryable) = error_fields(&err_msg);
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(err_msg),
                error_kind,
                retryable,
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Success case
    let (error_kind, retryable) = no_error_fields();
    let output = RecordDiscoveryOutput {
        success: true,
        temp_dir: Some(result.temp_dir),
        file: Some(result.file),
        error: None,
        error_kind,
        retryable,
    };

    Ok(serde_json::to_string(&output)?)
}
