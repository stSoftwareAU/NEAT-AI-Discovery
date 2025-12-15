//! Streaming recording session management
//!
//! This module provides a streaming API for recording discovery data incrementally,
//! solving the JavaScript "Invalid string length" error that occurs when trying to
//! serialise large datasets in a single JSON string.
//!
//! ## Usage Pattern
//!
//! ```text
//! TypeScript                          Rust (this library)
//! ─────────────────────────────────────────────────────────────
//! 1. start_discovery_session()   →    Creates session + Parquet file
//!    ↓ returns session_id
//!
//! 2. Loop while collecting data:
//!    - Collect records (estimate size)
//!    - When batch reaches ~50MB:
//!      append_discovery_records()  →   Writes batch to Parquet
//!
//! 3. finish_discovery_session()  →    Finalises Parquet file
//!    ↓ returns { temp_dir, file }
//! ```
//!
//! ## Benefits
//!
//! - **No string length limits**: Each batch is small enough to serialise
//! - **Unlimited sample sizes**: Can record for hours without memory issues
//! - **Fail-safe**: Partial data is preserved if process crashes mid-recording

use anyhow::{Context, Result};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use crate::parquet_format::ParquetRecordWriter;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronData};

/// Maximum estimated capacity for streaming sessions.
/// This is a large value since we don't know the final size upfront.
const STREAMING_MAX_CAPACITY: usize = i32::MAX as usize;

/// Global session storage
static SESSIONS: once_cell::sync::Lazy<Arc<Mutex<HashMap<String, RecordingSession>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(Mutex::new(HashMap::new())));

/// A recording session that holds an open Parquet writer
pub struct RecordingSession {
    writer: ParquetRecordWriter,
    creature: CreatureJson,
    temp_dir: String,
    records_written: u64,
}

impl RecordingSession {
    fn new(creature: CreatureJson, temp_dir: String, parquet_path: &str) -> Result<Self> {
        let max_arrow_offset = i32::MAX as usize;
        let writer = ParquetRecordWriter::new(
            parquet_path,
            max_arrow_offset,
            max_arrow_offset,
            STREAMING_MAX_CAPACITY,
        )
        .context("Failed to initialise Parquet writer for streaming session")?;

        Ok(Self {
            writer,
            creature,
            temp_dir,
            records_written: 0,
        })
    }
}

/// Start a new recording session
///
/// Creates a Parquet file and returns a session ID for subsequent append/finish calls.
pub fn start_session(creature: CreatureJson, temp_dir: String) -> Result<String> {
    // Create temp directory
    let temp_path = Path::new(&temp_dir);
    fs::create_dir_all(temp_path)
        .with_context(|| format!("Failed to create temp directory: {temp_dir}"))?;

    // Validate creature has non-input neurons
    let non_input_neuron_count = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type != "input")
        .count();

    if non_input_neuron_count == 0 {
        anyhow::bail!(
            "Cannot start recording session: creature has no non-input neurons. \
             Discovery recording requires at least one hidden or output neuron."
        );
    }

    // Generate session ID
    let session_id = Uuid::new_v4().to_string();

    // Create Parquet file path
    let parquet_file = temp_path.join("discovery_data.parquet");
    let parquet_path = parquet_file
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid file path"))?;

    // Create session
    let session = RecordingSession::new(creature, temp_dir, parquet_path)?;

    // Store session
    let mut sessions = SESSIONS.lock();
    sessions.insert(session_id.clone(), session);

    Ok(session_id)
}

/// Append records to an existing session
///
/// Returns the number of records written in this batch.
pub fn append_records(
    session_id: &str,
    neuron_data_batches: Vec<(u32, Vec<NeuronData>, Vec<f32>)>, // (obs_index, neuron_data, inputs)
) -> Result<u64> {
    let mut sessions = SESSIONS.lock();
    let session = sessions
        .get_mut(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    let mut records_in_batch = 0u64;

    for (obs_index, neuron_data, inputs) in neuron_data_batches {
        let mut batch_records = Vec::new();

        // Process neuron data
        for neuron_info in neuron_data {
            // Skip non-existent neurons (match existing behaviour)
            let neuron = match session
                .creature
                .neurons
                .iter()
                .find(|n| n.uuid == neuron_info.neuron_uuid)
            {
                Some(n) => n,
                None => continue,
            };

            // Skip input neurons
            if neuron.neuron_type == "input" {
                continue;
            }

            let record = DiscoverRecord::new(
                obs_index,
                neuron_info.neuron_uuid.clone(),
                neuron_info.value,
                neuron_info.activation,
                neuron_info.errors.clone(),
            );

            batch_records.push(record);
        }

        // Record input neuron activations
        for (input_index, value) in inputs.iter().enumerate() {
            let input_uuid = format!("input-{input_index}");
            let record =
                DiscoverRecord::new(obs_index, input_uuid, Some(*value), *value, Vec::new());
            batch_records.push(record);
        }

        if !batch_records.is_empty() {
            session
                .writer
                .write_records(&batch_records)
                .context("Failed to write records to Parquet")?;
            records_in_batch += batch_records.len() as u64;
        }
    }

    session.records_written += records_in_batch;
    Ok(records_in_batch)
}

/// Finish a recording session
///
/// Finalises the Parquet file and returns the file location.
/// The session is removed from storage after this call.
pub fn finish_session(session_id: &str) -> Result<(String, String, u64)> {
    let mut sessions = SESSIONS.lock();
    let session = sessions
        .remove(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    if session.records_written == 0 {
        anyhow::bail!("No records were written to the session");
    }

    session
        .writer
        .finish()
        .context("Failed to finalise Parquet writer")?;

    Ok((
        session.temp_dir,
        "discovery_data.parquet".to_string(),
        session.records_written,
    ))
}

/// Cancel a recording session without finalising
///
/// Use this to clean up if recording fails or is cancelled.
pub fn cancel_session(session_id: &str) -> Result<()> {
    let mut sessions = SESSIONS.lock();
    sessions
        .remove(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;
    // Session and writer are dropped, file may be incomplete but that's OK
    Ok(())
}

/// Get the number of active sessions (for diagnostics)
pub fn active_session_count() -> usize {
    SESSIONS.lock().len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NeuronData, NeuronJson};
    use tempfile::TempDir;

    fn create_test_creature() -> CreatureJson {
        CreatureJson {
            neurons: vec![
                NeuronJson {
                    uuid: "hidden-1".to_string(),
                    neuron_type: "hidden".to_string(),
                    squash: "TANH".to_string(),
                    bias: 0.0,
                },
                NeuronJson {
                    uuid: "output-0".to_string(),
                    neuron_type: "output".to_string(),
                    squash: "IDENTITY".to_string(),
                    bias: 0.0,
                },
            ],
            synapses: vec![],
            input: 2,
            output: 1,
        }
    }

    #[test]
    fn test_streaming_session_lifecycle() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();

        // Capture initial count (tests run in parallel, so other tests may have sessions)
        let initial_count = active_session_count();

        // Start session
        let session_id =
            start_session(creature, temp_dir.path().to_str().unwrap().to_string()).unwrap();
        assert!(!session_id.is_empty());
        assert!(active_session_count() > initial_count);

        // Append records - batch 1
        let batch1 = vec![(
            0u32,
            vec![
                NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.5,
                    value: Some(0.4),
                    errors: vec![0.1],
                },
                NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.5,
                    value: Some(0.5),
                    errors: vec![0.0],
                },
            ],
            vec![0.1, 0.2], // inputs
        )];
        let written1 = append_records(&session_id, batch1).unwrap();
        assert!(written1 > 0);

        // Append records - batch 2
        let batch2 = vec![(
            1u32,
            vec![
                NeuronData {
                    neuron_uuid: "hidden-1".to_string(),
                    activation: 0.6,
                    value: Some(0.5),
                    errors: vec![0.15],
                },
                NeuronData {
                    neuron_uuid: "output-0".to_string(),
                    activation: 0.6,
                    value: Some(0.6),
                    errors: vec![0.0],
                },
            ],
            vec![0.3, 0.4], // inputs
        )];
        let written2 = append_records(&session_id, batch2).unwrap();
        assert!(written2 > 0);

        // Finish session
        let count_before_finish = active_session_count();
        let (result_dir, file, total_records) = finish_session(&session_id).unwrap();
        assert_eq!(file, "discovery_data.parquet");
        assert_eq!(total_records, written1 + written2);
        assert!(active_session_count() < count_before_finish);

        // Verify file exists
        let parquet_path = Path::new(&result_dir).join(&file);
        assert!(parquet_path.exists());
    }

    #[test]
    fn test_streaming_session_cancel() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();

        let count_before = active_session_count();
        let session_id =
            start_session(creature, temp_dir.path().to_str().unwrap().to_string()).unwrap();
        assert!(active_session_count() > count_before);

        // Cancel without writing
        let count_before_cancel = active_session_count();
        cancel_session(&session_id).unwrap();
        assert!(active_session_count() < count_before_cancel);
    }

    #[test]
    fn test_streaming_session_not_found() {
        let result = append_records("nonexistent-session", vec![]);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Session not found"));
    }

    #[test]
    fn test_streaming_rejects_input_only_creature() {
        let temp_dir = TempDir::new().unwrap();
        let creature = CreatureJson {
            neurons: vec![NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            }],
            synapses: vec![],
            input: 1,
            output: 0,
        };

        let result = start_session(creature, temp_dir.path().to_str().unwrap().to_string());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("no non-input neurons"));
    }

    #[test]
    fn test_streaming_empty_session_fails_to_finish() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();

        let session_id =
            start_session(creature, temp_dir.path().to_str().unwrap().to_string()).unwrap();

        // Try to finish without writing any records
        let result = finish_session(&session_id);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("No records were written"));
    }

    #[test]
    fn test_streaming_multiple_batches_large_indices() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();

        let session_id =
            start_session(creature, temp_dir.path().to_str().unwrap().to_string()).unwrap();

        // Write with large, non-sequential indices (simulating sampled data)
        let batches: Vec<_> = [0, 1000, 50000, 123456]
            .iter()
            .map(|&idx| {
                (
                    idx as u32,
                    vec![
                        NeuronData {
                            neuron_uuid: "hidden-1".to_string(),
                            activation: 0.5,
                            value: Some(0.4),
                            errors: vec![0.1],
                        },
                        NeuronData {
                            neuron_uuid: "output-0".to_string(),
                            activation: 0.5,
                            value: Some(0.5),
                            errors: vec![0.0],
                        },
                    ],
                    vec![0.1, 0.2],
                )
            })
            .collect();

        let written = append_records(&session_id, batches).unwrap();
        assert!(written > 0);

        let (_, _, total) = finish_session(&session_id).unwrap();
        assert_eq!(total, written);
    }
}
