//! NEAT-AI Discovery Library
//!
//! High-performance Rust library for recording neuron activations and errors
//! during the discovery training phase, then scanning recorded data to identify
//! beneficial new synapses/neurons that would reduce error.

pub mod parquet_format;
pub mod record;
pub mod types;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// JSON input for record_discovery function
#[derive(Debug, Deserialize)]
pub struct RecordDiscoveryInput {
    pub creature: CreatureJson,
    pub training_data: Vec<TrainingRecord>,
    pub temp_dir: String,
    #[serde(default)]
    pub binary_file_path: Option<String>,
    #[serde(default)]
    pub record_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub timeout_seconds: Option<u64>,
}

/// JSON representation of Creature
#[derive(Debug, Deserialize)]
pub struct CreatureJson {
    pub neurons: Vec<NeuronJson>,
    pub synapses: Vec<SynapseJson>,
    pub input: usize,
    pub output: usize,
}

#[derive(Debug, Deserialize)]
pub struct NeuronJson {
    pub uuid: String,
    #[serde(rename = "type")]
    pub neuron_type: String,
    pub squash: String,
    pub bias: f32,
}

#[derive(Debug, Deserialize)]
pub struct SynapseJson {
    pub from_uuid: String,
    pub to_uuid: String,
    pub weight: f32,
}

/// Pre-computed neuron data for a single neuron
#[derive(Debug, Deserialize)]
pub struct NeuronData {
    pub neuron_uuid: String,
    pub activation: f32,
    #[serde(default)]
    pub value: Option<f32>,
    pub errors: Vec<f32>,
}

/// Training data record
#[derive(Debug, Deserialize)]
pub struct TrainingRecord {
    pub input: Vec<f32>,
    pub output: Vec<f32>,
    #[serde(default)]
    pub neuron_data: Option<Vec<NeuronData>>,
}

/// JSON output from record_discovery function
#[derive(Debug, Serialize)]
pub struct RecordDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temp_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

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
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Process discovery data - if this fails, return JSON error
    let result = match record::record_discovery_data(&input) {
        Ok(result) => result,
        Err(e) => {
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(e.to_string()),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Success case
    let output = RecordDiscoveryOutput {
        success: true,
        temp_dir: Some(result.temp_dir),
        file: Some(result.file),
        error: None,
    };

    Ok(serde_json::to_string(&output)?)
}

/// FFI export for recording discovery data
///
/// # Safety
/// This function is unsafe because it deals with raw C strings.
/// The caller must ensure:
/// - input_json is a valid null-terminated C string
/// - The returned pointer is freed using free_discovery_result
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn record_discovery(input_json: *const std::ffi::c_char) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};

    // Read input C string
    let input_str = unsafe {
        if input_json.is_null() {
            let error = r#"{"success":false,"error":"Null input pointer"}"#;
            return CString::new(error).unwrap().into_raw();
        }
        match CStr::from_ptr(input_json).to_str() {
            Ok(s) => s,
            Err(_) => {
                let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                return CString::new(error).unwrap().into_raw();
            }
        }
    };

    // DEBUG: Log that we're calling the internal function
    eprintln!("[DEBUG Rust lib] record_discovery FFI called, calling internal function");

    // DEBUG: Log first 500 chars of input JSON to verify structure
    let input_preview = if input_str.len() > 500 {
        format!("{}...", &input_str[..500])
    } else {
        input_str.to_string()
    };
    eprintln!("[DEBUG Rust lib] Input JSON preview (first 500 chars): {input_preview}");

    // Parse JSON to verify structure (for debugging only)
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(input_str) {
        eprintln!("[DEBUG Rust lib] JSON parsed successfully");
        // Log training_data length and first record's neuron_data if available
        if let Some(training_data) = v.get("training_data").and_then(|td| td.as_array()) {
            eprintln!(
                "[DEBUG Rust lib] training_data.length={}",
                training_data.len()
            );
            if let Some(first_record) = training_data.first() {
                if let Some(neuron_data) = first_record
                    .get("neuron_data")
                    .and_then(|neuron_data_val| neuron_data_val.as_array())
                {
                    eprintln!(
                        "[DEBUG Rust lib] First record neuron_data.length={}",
                        neuron_data.len()
                    );
                    if let Some(hidden3) = neuron_data
                        .iter()
                        .find(|n| n.get("neuron_uuid").and_then(|u| u.as_str()) == Some("hidden-3"))
                    {
                        eprintln!(
                            "[DEBUG Rust lib] First record hidden-3: activation={:?}, errors={:?}",
                            hidden3.get("activation"),
                            hidden3.get("errors")
                        );
                    }
                }
            }
        }
    } else {
        eprintln!("[DEBUG Rust lib] JSON parse failed (but continuing anyway)");
    }

    // Call the Rust function with original string
    let result = match record_discovery_internal(input_str) {
        Ok(json) => {
            eprintln!("[DEBUG Rust lib] record_discovery_internal succeeded");
            json
        }
        Err(e) => {
            eprintln!("[DEBUG Rust lib] record_discovery_internal failed: {e}");
            // Properly serialize error message to avoid JSON injection issues
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(e.to_string()),
            };
            let error_json = serde_json::to_string(&output).unwrap_or_else(|_| {
                // Fallback if serialization fails (shouldn't happen)
                r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
            });
            return CString::new(error_json).unwrap().into_raw();
        }
    };

    // Return as C string
    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
}

/// JSON input for read_discovery_records function
#[derive(Debug, Deserialize)]
pub struct ReadDiscoveryInput {
    pub parquet_file: String,
    pub neuron_uuid: String,
}

/// JSON output from read_discovery_records function
#[derive(Debug, Serialize)]
pub struct ReadDiscoveryOutput {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub records: Option<Vec<DiscoverRecordJson>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// JSON representation of DiscoverRecord for serialization
#[derive(Debug, Serialize)]
pub struct DiscoverRecordJson {
    pub obs_index: u32,
    pub neuron_uuid: String,
    pub value: Option<f32>,
    pub activation: f32,
    pub errors: Vec<f32>,
}

/// Read discovery records from Parquet file for a specific neuron
///
/// Takes JSON input and returns JSON output for easy integration with TypeScript/DenoJS
pub fn read_discovery_records(input_json: &str) -> Result<String> {
    use crate::parquet_format::read_records_from_parquet;
    use crate::types::DiscoverRecord;

    // Parse input JSON
    let input: ReadDiscoveryInput = match serde_json::from_str(input_json) {
        Ok(input) => input,
        Err(e) => {
            let output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(format!("Failed to parse input JSON: {e}")),
            };
            return Ok(serde_json::to_string(&output)?);
        }
    };

    // Read records from Parquet
    let records: Vec<DiscoverRecord> =
        match read_records_from_parquet(&input.parquet_file, &input.neuron_uuid) {
            Ok(records) => records,
            Err(e) => {
                let output = ReadDiscoveryOutput {
                    success: false,
                    records: None,
                    error: Some(e.to_string()),
                };
                return Ok(serde_json::to_string(&output)?);
            }
        };

    // Convert to JSON format
    let json_records: Vec<DiscoverRecordJson> = records
        .into_iter()
        .map(|r| {
            // DEBUG: Log first few records
            if r.neuron_uuid == "hidden-3" && r.obs_index < 3 {
                eprintln!("[DEBUG Rust lib read] Converting record: obs_index={}, activation={}, errors.len()={}, errors={:?}", 
                    r.obs_index, r.activation, r.errors.len(), &r.errors[..r.errors.len().min(3)]);
            }
            DiscoverRecordJson {
                obs_index: r.obs_index,
                neuron_uuid: r.neuron_uuid,
                value: r.value,
                activation: r.activation,
                errors: r.errors,
            }
        })
        .collect();

    let output = ReadDiscoveryOutput {
        success: true,
        records: Some(json_records),
        error: None,
    };

    let json_string = serde_json::to_string(&output)?;

    // DEBUG: Log first 1000 chars of JSON to verify errors are included
    let json_preview = if json_string.len() > 1000 {
        format!("{}...", &json_string[..1000])
    } else {
        json_string.clone()
    };
    eprintln!("[DEBUG Rust lib read] JSON output preview (first 1000 chars): {json_preview}");

    Ok(json_string)
}

/// FFI export for reading discovery records
///
/// # Safety
/// This function is unsafe because it deals with raw C strings.
/// The caller must ensure:
/// - input_json is a valid null-terminated C string
/// - The returned pointer is freed using free_discovery_result
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn read_discovery_records_ffi(
    input_json: *const std::ffi::c_char,
) -> *mut std::ffi::c_char {
    use std::ffi::{CStr, CString};

    // Read input C string
    let input_str = unsafe {
        if input_json.is_null() {
            let error = r#"{"success":false,"error":"Null input pointer"}"#;
            return CString::new(error).unwrap().into_raw();
        }
        match CStr::from_ptr(input_json).to_str() {
            Ok(s) => s,
            Err(_) => {
                let error = r#"{"success":false,"error":"Invalid UTF-8 in input"}"#;
                return CString::new(error).unwrap().into_raw();
            }
        }
    };

    // DEBUG: Log that we're calling the internal function
    eprintln!("[DEBUG Rust lib] read_discovery_records_ffi called, calling internal function");

    // Call the Rust function
    let result = match read_discovery_records(input_str) {
        Ok(json) => {
            eprintln!("[DEBUG Rust lib] read_discovery_records succeeded");
            json
        }
        Err(e) => {
            eprintln!("[DEBUG Rust lib] read_discovery_records failed: {e}");
            // Properly serialize error message to avoid JSON injection issues
            let output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(e.to_string()),
            };
            let error_json = serde_json::to_string(&output).unwrap_or_else(|_| {
                // Fallback if serialization fails (shouldn't happen)
                r#"{"success":false,"error":"Failed to serialize error message"}"#.to_string()
            });
            return CString::new(error_json).unwrap().into_raw();
        }
    };

    // Return as C string
    match CString::new(result) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => {
            let error = r#"{"success":false,"error":"Failed to create output string"}"#;
            CString::new(error).unwrap().into_raw()
        }
    }
}

/// FFI export for freeing memory allocated by read_discovery_records_ffi
///
/// # Safety
/// This function is unsafe because it frees memory allocated by Rust.
/// The caller must ensure ptr was returned by read_discovery_records_ffi.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
#[no_mangle]
pub extern "C" fn free_discovery_result(ptr: *mut std::ffi::c_char) {
    use std::ffi::CString;
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_discovery_json_interface() {
        let input = r#"{
            "creature": {
                "neurons": [],
                "synapses": [],
                "input": 2,
                "output": 1
            },
            "training_data": [],
            "temp_dir": ".discovery/test"
        }"#;

        // Should parse without error
        let parsed: RecordDiscoveryInput = serde_json::from_str(input).unwrap();
        assert_eq!(parsed.creature.input, 2);
        assert_eq!(parsed.creature.output, 1);
    }

    #[test]
    fn test_error_message_json_escaping() {
        // Test that error messages with special characters are properly escaped
        let error_messages = vec![
            r#"Error with "quotes""#,
            r#"Error with \backslash"#,
            "Error with\nnewline",
            r#"Error with "quotes" and \backslash and\nnewline"#,
            r#"Error with "multiple" "quotes" and \multiple\backslashes"#,
        ];

        for error_msg in error_messages {
            // Test RecordDiscoveryOutput
            let output = RecordDiscoveryOutput {
                success: false,
                temp_dir: None,
                file: None,
                error: Some(error_msg.to_string()),
            };
            let json = serde_json::to_string(&output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed["success"], false);
            assert_eq!(parsed["error"].as_str(), Some(error_msg));

            // Test ReadDiscoveryOutput
            let read_output = ReadDiscoveryOutput {
                success: false,
                records: None,
                error: Some(error_msg.to_string()),
            };
            let read_json = serde_json::to_string(&read_output).unwrap();
            // Verify JSON is valid and can be parsed back
            let parsed_read: serde_json::Value = serde_json::from_str(&read_json).unwrap();
            assert_eq!(parsed_read["success"], false);
            assert_eq!(parsed_read["error"].as_str(), Some(error_msg));
        }
    }
}
