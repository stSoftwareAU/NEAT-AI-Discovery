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

#![allow(clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::{Context, Result};
use parking_lot::Mutex;
use rand::RngExt;
use rand::distr::Alphanumeric;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock};

use crate::parquet_format::ParquetRecordWriter;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronData};

/// Maximum estimated capacity for streaming sessions.
/// This is a large value since we don't know the final size upfront.
const STREAMING_MAX_CAPACITY: usize = i32::MAX as usize;

/// Global session storage
static SESSIONS: LazyLock<Arc<Mutex<HashMap<String, RecordingSession>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

/// A recording session that holds an open Parquet writer.
///
/// If the session is dropped without calling `finish_session()`, the incomplete
/// parquet file is automatically cleaned up to prevent orphaned files.
pub struct RecordingSession {
    writer: Option<ParquetRecordWriter>,
    creature: CreatureJson,
    temp_dir: String,
    parquet_path: String,
    records_written: u64,
    finished: bool,
}

impl RecordingSession {
    fn new(creature: CreatureJson, temp_dir: String, parquet_path: &str) -> Result<Self> {
        // Write to a temporary file; rename to final path on successful finish
        let tmp_path = format!("{parquet_path}.tmp");
        let max_arrow_offset = i32::MAX as usize;
        let writer = ParquetRecordWriter::new(
            &tmp_path,
            max_arrow_offset,
            max_arrow_offset,
            STREAMING_MAX_CAPACITY,
        )
        .context("Failed to initialise Parquet writer for streaming session")?;

        Ok(Self {
            writer: Some(writer),
            creature,
            temp_dir,
            parquet_path: parquet_path.to_string(),
            records_written: 0,
            finished: false,
        })
    }
}

impl Drop for RecordingSession {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        // Session was not finished normally — clean up the incomplete temporary file
        let tmp_path = format!("{}.tmp", self.parquet_path);
        if let Err(err) = fs::remove_file(&tmp_path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            tracing::warn!(
                path = %tmp_path,
                error = %err,
                "Failed to clean up incomplete parquet file on session drop"
            );
        }
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

    // Generate session ID using rand (already a dependency)
    let session_id: String = rand::rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();

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
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("Session writer already consumed"))?
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
    let mut session = sessions
        .remove(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;

    if session.records_written == 0 {
        anyhow::bail!("No records were written to the session");
    }

    let writer = session
        .writer
        .take()
        .ok_or_else(|| anyhow::anyhow!("Session writer already consumed"))?;
    writer
        .finish()
        .context("Failed to finalise Parquet writer")?;

    // Atomically rename .parquet.tmp to .parquet so consumers never see partial files
    let tmp_path = format!("{}.tmp", session.parquet_path);
    fs::rename(&tmp_path, &session.parquet_path).with_context(|| {
        format!(
            "Failed to rename temporary file {tmp_path} to {}",
            session.parquet_path
        )
    })?;

    session.finished = true;

    let temp_dir = session.temp_dir.clone();
    let records_written = session.records_written;

    Ok((
        temp_dir,
        "discovery_data.parquet".to_string(),
        records_written,
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
    // Session Drop impl cleans up the incomplete temporary file
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

    fn session_exists(session_id: &str) -> bool {
        SESSIONS.lock().contains_key(session_id)
    }

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

        // Start session
        let session_id =
            start_session(creature, temp_dir.path().to_str().unwrap().to_string()).unwrap();
        assert!(!session_id.is_empty());
        assert!(
            session_exists(&session_id),
            "expected started session to exist"
        );

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
        assert!(
            session_exists(&session_id),
            "expected session to exist before finish"
        );
        let (result_dir, file, total_records) = finish_session(&session_id).unwrap();
        assert_eq!(file, "discovery_data.parquet");
        assert_eq!(total_records, written1 + written2);
        assert!(
            !session_exists(&session_id),
            "expected finished session to be removed"
        );

        // Verify file exists
        let parquet_path = Path::new(&result_dir).join(&file);
        assert!(parquet_path.exists());
    }

    #[test]
    fn test_streaming_session_cancel() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();

        let session_id =
            start_session(creature, temp_dir.path().to_str().unwrap().to_string()).unwrap();
        assert!(
            session_exists(&session_id),
            "expected started session to exist"
        );

        // Cancel without writing
        cancel_session(&session_id).unwrap();
        assert!(
            !session_exists(&session_id),
            "expected cancelled session to be removed"
        );
    }

    #[test]
    fn test_streaming_session_not_found() {
        let result = append_records("nonexistent-session", vec![]);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Session not found")
        );
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
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no non-input neurons")
        );
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
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("No records were written")
        );
    }

    #[test]
    fn test_cancel_session_cleans_up_incomplete_file() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();

        let session_id = start_session(creature, temp_path.clone()).unwrap();

        // Write some records so a partial file exists on disk
        let batch = vec![(
            0u32,
            vec![NeuronData {
                neuron_uuid: "hidden-1".to_string(),
                activation: 0.5,
                value: Some(0.4),
                errors: vec![0.1],
            }],
            vec![0.1, 0.2],
        )];
        append_records(&session_id, batch).unwrap();

        // The temporary file should exist before cancel
        let tmp_file = Path::new(&temp_path).join("discovery_data.parquet.tmp");
        assert!(
            tmp_file.exists(),
            "expected tmp file to exist before cancel"
        );

        // The final file should NOT exist (not yet finished)
        let final_file = Path::new(&temp_path).join("discovery_data.parquet");
        assert!(
            !final_file.exists(),
            "expected final file to not exist before finish"
        );

        // Cancel the session — Drop should clean up the tmp file
        cancel_session(&session_id).unwrap();

        assert!(
            !tmp_file.exists(),
            "expected tmp file to be cleaned up after cancel"
        );
        assert!(
            !final_file.exists(),
            "expected final file to not exist after cancel"
        );
    }

    #[test]
    fn test_drop_without_finish_cleans_up_file() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();

        let session_id = start_session(creature, temp_path.clone()).unwrap();

        // Write some records
        let batch = vec![(
            0u32,
            vec![NeuronData {
                neuron_uuid: "hidden-1".to_string(),
                activation: 0.5,
                value: Some(0.4),
                errors: vec![0.1],
            }],
            vec![0.1, 0.2],
        )];
        append_records(&session_id, batch).unwrap();

        let tmp_file = Path::new(&temp_path).join("discovery_data.parquet.tmp");
        assert!(tmp_file.exists(), "expected tmp file to exist");

        // Manually remove the session to trigger Drop (simulates panic/drop scenario)
        {
            let mut sessions = SESSIONS.lock();
            sessions.remove(&session_id);
            // Session is dropped here
        }

        assert!(
            !tmp_file.exists(),
            "expected tmp file to be cleaned up on drop"
        );
    }

    #[test]
    fn test_finished_session_preserves_file() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();

        let session_id = start_session(creature, temp_path).unwrap();

        let batch = vec![(
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
            vec![0.1, 0.2],
        )];
        append_records(&session_id, batch).unwrap();

        // Finish the session — file should be renamed and preserved
        let (result_dir, file, _) = finish_session(&session_id).unwrap();

        let final_file = Path::new(&result_dir).join(&file);
        assert!(
            final_file.exists(),
            "expected final parquet file to exist after finish"
        );

        // The tmp file should no longer exist (renamed to final)
        let tmp_file = Path::new(&result_dir).join("discovery_data.parquet.tmp");
        assert!(
            !tmp_file.exists(),
            "expected tmp file to be gone after finish (renamed)"
        );
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
