//! Rust-native counterpart of the recording entry point in
//! `src/ffi/recording.rs`: `record_discovery` calls into
//! `record_discovery_internal` here rather than inlining the logic, so the
//! recording path is reachable from Rust integration tests without crossing
//! the C boundary.
//!
//! **Contract** — JSON in, JSON out. A caller-input failure (unparsable JSON,
//! a creature rejected by `validate_creature`, an unwritable temp directory)
//! is returned as a `success: false` payload carrying `error` and
//! `error_kind`, never as a Rust `Err` handed back to the caller; `Err` is
//! reserved for a failure to serialise the response itself. The C-boundary
//! concerns — null/invalid-UTF-8 pointer rejection, `catch_unwind` panic
//! containment and `CString` conversion — stay one layer up in `src/ffi/`.

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
            let typed = DiscoveryError::InvalidInput {
                detail: format!("Failed to parse input JSON: {e}"),
            };
            let kind = typed.error_kind();
            let output = RecordDiscoveryOutput {
                success: false,
                schema_version: SCHEMA_VERSION.to_string(),
                temp_dir: None,
                file: None,
                error: Some(typed.to_string()),
                error_kind: Some(kind),
                retryable: Some(kind.is_retryable()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Issue #1184: Reject corrupt creatures carrying recurrent or
    // unresolved synapses before they reach the recording pipeline.
    // Issue #1867: and reject an out-of-range input-neuron count, which
    // would otherwise drive an unbounded allocation inside the pipeline.
    if let Err(typed) = validate_creature(&input.creature) {
        let kind = typed.error_kind();
        let output = RecordDiscoveryOutput {
            success: false,
            schema_version: SCHEMA_VERSION.to_string(),
            temp_dir: None,
            file: None,
            error: Some(typed.to_string()),
            error_kind: Some(kind),
            retryable: Some(kind.is_retryable()),
        };
        return Ok(serde_json::to_string(&output)?);
    }

    // Process discovery data - if this fails, return JSON error
    let result = match record::record_discovery_data(&input) {
        Ok(result) => result,
        Err(e) => {
            let (err_msg, error_kind, retryable) = error_fields_from_anyhow(&e);
            let output = RecordDiscoveryOutput {
                success: false,
                schema_version: SCHEMA_VERSION.to_string(),
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
        schema_version: SCHEMA_VERSION.to_string(),
        temp_dir: Some(result.temp_dir),
        file: Some(result.file),
        error: None,
        error_kind,
        retryable,
    };

    Ok(serde_json::to_string(&output)?)
}
