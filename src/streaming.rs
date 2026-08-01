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
//!
//! ## Locking (Issue #1751)
//!
//! Sessions are stored as `Arc<SessionEntry>` behind the process-global `SESSIONS`
//! map. The global lock is held only long enough to look up, insert, or remove a
//! handle; every Parquet disk write happens under the *per-session* lock with the
//! global lock already released. Operations on unrelated sessions therefore never
//! queue behind another session's disk I/O.
//!
//! ## Cancellation tombstone (Issue #1876)
//!
//! Removal from the map alone is invisible to an append that already cloned the
//! handle, so `cancel_session` and the TTL sweep also set a `cancelled` flag on the
//! entry. `append_records` checks it after taking the per-session lock and again
//! after its writes, so a batch destined for a `.tmp` file that is about to be
//! discarded fails loudly instead of returning `Ok`. The flag lives *outside* the
//! per-session mutex so cancelling never waits on an in-flight Parquet write.

#![allow(clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use anyhow::{Context, Result};
use parking_lot::Mutex;
use rand::RngExt;
use rand::distr::Alphanumeric;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use crate::parquet_format::ParquetRecordWriter;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronData};

/// Maximum estimated capacity for streaming sessions.
/// This is a large value since we don't know the final size upfront.
const STREAMING_MAX_CAPACITY: usize = i32::MAX as usize;

/// A shared handle to one recording session.
///
/// Each session carries its own lock so that a slow Parquet write on one session
/// never blocks operations on another (Issue #1751).
type SessionHandle = Arc<SessionEntry>;

/// A registered recording session plus its cancellation tombstone.
///
/// The tombstone sits outside the mutex so `cancel_session` and the TTL sweep can
/// mark a session dead without waiting for an in-flight Parquet write to finish
/// (Issue #1876).
struct SessionEntry {
    cancelled: AtomicBool,
    session: Mutex<RecordingSession>,
}

impl SessionEntry {
    fn new(session: RecordingSession) -> Self {
        Self {
            cancelled: AtomicBool::new(false),
            session: Mutex::new(session),
        }
    }

    /// Mark the session dead — its incomplete `.tmp` file is about to be discarded.
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

/// The error returned by an append to a session that has been cancelled or swept.
fn cancelled_error(session_id: &str) -> anyhow::Error {
    anyhow::anyhow!("Session cancelled: {session_id}")
}

/// Global session storage
static SESSIONS: LazyLock<Arc<Mutex<HashMap<String, SessionHandle>>>> =
    LazyLock::new(|| Arc::new(Mutex::new(HashMap::new())));

/// Look up a session handle, releasing the global lock before returning.
///
/// The caller locks the returned handle for the duration of its work, so the
/// global map stays available to other sessions throughout.
fn session_handle(session_id: &str) -> Option<SessionHandle> {
    SESSIONS.lock().get(session_id).map(Arc::clone)
}

/// Remove a session handle from the global map, releasing the global lock before
/// returning.
///
/// The handle is returned rather than dropped in place: `RecordingSession::drop`
/// deletes the incomplete Parquet file, and that disk I/O must not run under the
/// global lock.
fn remove_session_handle(session_id: &str) -> Option<SessionHandle> {
    SESSIONS.lock().remove(session_id)
}

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
    created_at: Instant,
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
            created_at: Instant::now(),
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

/// Remove streaming sessions that have exceeded the configured TTL.
///
/// This is called automatically at the start of each `start_session()` call
/// to prevent orphaned sessions from leaking memory when the host process
/// crashes or fails to call finish/cancel. Stale sessions are logged at
/// `warn` level before removal; their temp files are cleaned up via the
/// `Drop` implementation on `RecordingSession`.
pub fn cleanup_stale_sessions() {
    let ttl_secs = crate::config::session_ttl_secs();
    let ttl = std::time::Duration::from_secs(ttl_secs);

    let stale: Vec<(String, u64, SessionHandle)> = {
        let mut sessions = SESSIONS.lock();

        let stale_ids: Vec<(String, u64)> = sessions
            .iter()
            .filter_map(|(id, handle)| {
                // A session whose per-session lock is held is mid-write, so it is
                // active by definition — skip it rather than block the sweep behind
                // its disk I/O (Issue #1751). It is reconsidered on the next sweep.
                let session = handle.session.try_lock()?;
                let age = session.created_at.elapsed();
                (age > ttl).then(|| (id.clone(), age.as_secs()))
            })
            .collect();

        stale_ids
            .into_iter()
            .filter_map(|(id, age_secs)| {
                sessions.remove(&id).map(|handle| {
                    // Tombstone at the instant of eviction so an append holding a
                    // handle cloned earlier cannot report success (Issue #1876).
                    handle.cancel();
                    (id, age_secs, handle)
                })
            })
            .collect()
    };
    // Global lock released — `RecordingSession::drop` (which deletes the incomplete
    // temporary file) now runs outside it.

    for (id, age_secs, handle) in stale {
        tracing::warn!(
            session_id = %id,
            age_secs = age_secs,
            ttl_secs = ttl_secs,
            "Removing stale streaming session (exceeded TTL)"
        );
        // Session `Drop` cleans up the incomplete temporary file
        drop(handle);
    }
}

/// Start a new recording session
///
/// Creates a Parquet file and returns a session ID for subsequent append/finish calls.
pub fn start_session(creature: CreatureJson, temp_dir: String) -> Result<String> {
    // Clean up any orphaned sessions that have exceeded the TTL
    cleanup_stale_sessions();
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

    // Store session — the global lock is held only for the map insert
    SESSIONS
        .lock()
        .insert(session_id.clone(), Arc::new(SessionEntry::new(session)));

    Ok(session_id)
}

/// Append records to an existing session
///
/// Returns the number of records written in this batch. Returns an error if the
/// session was cancelled or TTL-swept — its records would be discarded, so they
/// must never be acknowledged as written (Issue #1876).
pub fn append_records(
    session_id: &str,
    neuron_data_batches: Vec<(u32, Vec<NeuronData>, Vec<f32>)>, // (obs_index, neuron_data, inputs)
) -> Result<u64> {
    let handle = session_handle(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;
    append_to_handle(&handle, session_id, neuron_data_batches)
}

/// Append a batch through an already-resolved session handle.
///
/// Split out from `append_records` so the cancellation window between resolving
/// the handle and taking the per-session lock is directly testable (Issue #1876).
fn append_to_handle(
    handle: &SessionHandle,
    session_id: &str,
    neuron_data_batches: Vec<(u32, Vec<NeuronData>, Vec<f32>)>,
) -> Result<u64> {
    // Global lock already released — the Parquet writes below run under the
    // per-session lock only (Issue #1751).
    let mut session = handle.session.lock();

    // The session may have been cancelled or swept between the handle lookup and
    // this lock; writing now would fill a `.tmp` file that is already doomed.
    if handle.is_cancelled() {
        return Err(cancelled_error(session_id));
    }

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

    // Cancellation can also land while this append holds the per-session lock, so
    // re-check before acknowledging the batch.
    if handle.is_cancelled() {
        return Err(cancelled_error(session_id));
    }

    session.records_written += records_in_batch;
    Ok(records_in_batch)
}

/// Finish a recording session
///
/// Finalises the Parquet file and returns the file location.
/// The session is removed from storage after this call.
pub fn finish_session(session_id: &str) -> Result<(String, String, u64)> {
    let handle = remove_session_handle(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;
    // Global lock already released — finalisation (writer flush + rename) runs
    // under the per-session lock only (Issue #1751).
    let mut session = handle.session.lock();

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
    let handle = remove_session_handle(session_id)
        .ok_or_else(|| anyhow::anyhow!("Session not found: {session_id}"))?;
    // Tombstone before dropping so a concurrent append that already holds a handle
    // fails instead of reporting doomed records as written (Issue #1876). The flag
    // is lock-free, so cancelling never waits on an in-flight Parquet write.
    handle.cancel();
    // Global lock already released — the Session Drop impl cleans up the incomplete
    // temporary file outside it (Issue #1751).
    drop(handle);
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
    use std::time::Duration;
    use tempfile::TempDir;

    fn session_exists(session_id: &str) -> bool {
        SESSIONS.lock().contains_key(session_id)
    }

    /// Backdate a session's creation time so the TTL sweep treats it as stale.
    fn backdate_session(session_id: &str, secs: u64) {
        let handle = session_handle(session_id).expect("session should exist");
        handle.session.lock().created_at = Instant::now() - Duration::from_secs(secs);
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
        drop(remove_session_handle(&session_id));

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

    #[test]
    fn test_cleanup_stale_sessions_removes_expired() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();

        // Start a session and manually backdate its creation time
        let session_id = start_session(creature, temp_path).unwrap();
        assert!(session_exists(&session_id));

        // Backdate created_at to 2 hours ago (exceeds default 1-hour TTL)
        backdate_session(&session_id, 7200);

        // Run cleanup — should remove the stale session
        cleanup_stale_sessions();

        assert!(
            !session_exists(&session_id),
            "expected stale session to be removed after cleanup"
        );
    }

    #[test]
    fn test_cleanup_stale_sessions_preserves_fresh() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();

        // Start a fresh session (created_at is now)
        let session_id = start_session(creature, temp_path).unwrap();
        assert!(session_exists(&session_id));

        // Run cleanup — fresh session should survive
        cleanup_stale_sessions();

        assert!(
            session_exists(&session_id),
            "expected fresh session to survive cleanup"
        );

        // Clean up
        cancel_session(&session_id).unwrap();
    }

    #[test]
    fn test_cleanup_stale_sessions_cleans_temp_files() {
        let temp_dir = TempDir::new().unwrap();
        let creature = create_test_creature();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();

        let session_id = start_session(creature, temp_path.clone()).unwrap();

        // Write some records so a temp file exists on disk
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
        assert!(
            tmp_file.exists(),
            "expected tmp file to exist before cleanup"
        );

        // Backdate the session
        backdate_session(&session_id, 7200);

        // Run cleanup — should remove the session and its temp file via Drop
        cleanup_stale_sessions();

        assert!(
            !session_exists(&session_id),
            "expected stale session to be removed"
        );
        assert!(
            !tmp_file.exists(),
            "expected tmp file to be cleaned up when stale session is removed"
        );
    }

    #[test]
    fn test_start_session_triggers_cleanup() {
        let temp_dir1 = TempDir::new().unwrap();
        let temp_dir2 = TempDir::new().unwrap();
        let creature1 = create_test_creature();
        let creature2 = create_test_creature();

        // Start first session and backdate it
        let session_id1 =
            start_session(creature1, temp_dir1.path().to_str().unwrap().to_string()).unwrap();
        backdate_session(&session_id1, 7200);

        // Starting a new session should trigger cleanup of the stale one
        let session_id2 =
            start_session(creature2, temp_dir2.path().to_str().unwrap().to_string()).unwrap();

        assert!(
            !session_exists(&session_id1),
            "expected stale session to be cleaned up when starting new session"
        );
        assert!(
            session_exists(&session_id2),
            "expected new session to exist"
        );

        // Clean up
        cancel_session(&session_id2).unwrap();
    }

    fn single_record_batch(obs_index: u32) -> Vec<(u32, Vec<NeuronData>, Vec<f32>)> {
        vec![(
            obs_index,
            vec![NeuronData {
                neuron_uuid: "hidden-1".to_string(),
                activation: 0.5,
                value: Some(0.4),
                errors: vec![0.1],
            }],
            vec![0.1, 0.2],
        )]
    }

    /// Issue #1751: a session busy writing to disk must not block operations on
    /// other sessions. Holding session A's per-session lock stands in for an
    /// in-progress Parquet write; all other API calls must still complete.
    #[test]
    fn test_other_sessions_proceed_while_one_session_is_locked() {
        let dir_a = TempDir::new().unwrap();
        let dir_b = TempDir::new().unwrap();
        let dir_c = TempDir::new().unwrap();

        let id_a = start_session(
            create_test_creature(),
            dir_a.path().to_str().unwrap().to_string(),
        )
        .unwrap();
        let id_b = start_session(
            create_test_creature(),
            dir_b.path().to_str().unwrap().to_string(),
        )
        .unwrap();

        // Simulate a long disk write in progress on session A
        let handle_a = session_handle(&id_a).expect("session A should exist");
        let guard_a = handle_a.session.lock();

        let (tx, rx) = std::sync::mpsc::channel();
        let id_b_thread = id_b.clone();
        let dir_c_path = dir_c.path().to_str().unwrap().to_string();
        std::thread::spawn(move || {
            let count = active_session_count();
            let written = append_records(&id_b_thread, single_record_batch(0));
            let id_c = start_session(create_test_creature(), dir_c_path);
            tx.send((count, written, id_c)).expect("receiver alive");
        });

        let (count, written, id_c) = rx
            .recv_timeout(Duration::from_secs(10))
            .expect("operations on other sessions must not block behind session A's write lock");

        drop(guard_a);

        assert!(count >= 2, "expected at least both sessions to be counted");
        assert_eq!(
            written.expect("append to session B should succeed"),
            3,
            "expected one hidden record plus two input records"
        );
        let id_c = id_c.expect("starting a new session should succeed");

        cancel_session(&id_a).unwrap();
        cancel_session(&id_b).unwrap();
        cancel_session(&id_c).unwrap();
    }

    /// Issue #1751: the TTL sweep must not block on a session that is mid-write;
    /// it skips it and reclaims it on a later sweep.
    #[test]
    fn test_cleanup_skips_locked_session_then_reclaims_it() {
        let temp_dir = TempDir::new().unwrap();
        let session_id = start_session(
            create_test_creature(),
            temp_dir.path().to_str().unwrap().to_string(),
        )
        .unwrap();
        backdate_session(&session_id, 7200);

        let handle = session_handle(&session_id).expect("session should exist");
        let guard = handle.session.lock();

        // Must return promptly rather than deadlocking on the held session lock
        cleanup_stale_sessions();
        assert!(
            session_exists(&session_id),
            "expected a session that is mid-write to be skipped by the TTL sweep"
        );

        drop(guard);

        cleanup_stale_sessions();
        assert!(
            !session_exists(&session_id),
            "expected the stale session to be reclaimed once its write completed"
        );
    }

    /// Issue #1751: concurrent appends to the *same* session stay serialised —
    /// every record still lands in the Parquet file.
    #[test]
    fn test_concurrent_appends_to_same_session_are_serialised() {
        let temp_dir = TempDir::new().unwrap();
        let session_id = start_session(
            create_test_creature(),
            temp_dir.path().to_str().unwrap().to_string(),
        )
        .unwrap();

        let threads: Vec<_> = (0..4u32)
            .map(|i| {
                let id = session_id.clone();
                std::thread::spawn(move || append_records(&id, single_record_batch(i)).unwrap())
            })
            .collect();

        let written: u64 = threads.into_iter().map(|t| t.join().unwrap()).sum();
        assert_eq!(written, 12, "expected 3 records from each of 4 threads");

        let (_, _, total) = finish_session(&session_id).unwrap();
        assert_eq!(total, written, "every concurrent append must be counted");
    }

    /// Issue #1876: an append that resolved its handle *before* the session was
    /// cancelled must not report the doomed batch as written.
    #[test]
    fn test_append_via_handle_cancelled_after_lookup_errors() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();
        let session_id = start_session(create_test_creature(), temp_path).unwrap();

        // Stand in for an append that has resolved its handle but not yet locked it
        let handle = session_handle(&session_id).expect("session should exist");

        cancel_session(&session_id).unwrap();

        let err = append_to_handle(&handle, &session_id, single_record_batch(0))
            .expect_err("appending to a cancelled session must return an error");
        assert!(
            err.to_string().contains("Session cancelled"),
            "unexpected error: {err}"
        );
    }

    /// Issue #1876: the TTL sweep tombstones the sessions it evicts, so an append
    /// racing the sweep fails rather than writing into a discarded file.
    #[test]
    fn test_append_via_handle_swept_after_lookup_errors() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();
        let session_id = start_session(create_test_creature(), temp_path).unwrap();

        let handle = session_handle(&session_id).expect("session should exist");
        backdate_session(&session_id, 7200);
        cleanup_stale_sessions();

        let err = append_to_handle(&handle, &session_id, single_record_batch(0))
            .expect_err("appending to a swept session must return an error");
        assert!(
            err.to_string().contains("Session cancelled"),
            "unexpected error: {err}"
        );
    }

    /// Issue #1876: the tombstone is lock-free, so cancelling a session whose
    /// per-session lock is held by an in-flight write returns promptly — and the
    /// write that follows still fails.
    #[test]
    fn test_cancel_does_not_wait_for_an_in_flight_write() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();
        let session_id = start_session(create_test_creature(), temp_path).unwrap();

        let handle = session_handle(&session_id).expect("session should exist");
        // Simulate a long Parquet write holding the per-session lock
        let guard = handle.session.lock();

        let (tx, rx) = std::sync::mpsc::channel();
        let id_thread = session_id.clone();
        std::thread::spawn(move || {
            tx.send(cancel_session(&id_thread)).expect("receiver alive");
        });

        rx.recv_timeout(Duration::from_secs(10))
            .expect("cancel must not block behind an in-flight write")
            .expect("cancelling an open session should succeed");

        drop(guard);

        let err = append_to_handle(&handle, &session_id, single_record_batch(0))
            .expect_err("appending after cancellation must return an error");
        assert!(
            err.to_string().contains("Session cancelled"),
            "unexpected error: {err}"
        );
    }

    /// Issue #1876: a cancelled session's batch must not be counted, so a
    /// subsequent handle-level append still sees zero recorded records.
    #[test]
    fn test_cancelled_append_does_not_count_records() {
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path().to_str().unwrap().to_string();
        let session_id = start_session(create_test_creature(), temp_path).unwrap();

        let handle = session_handle(&session_id).expect("session should exist");
        cancel_session(&session_id).unwrap();

        assert!(append_to_handle(&handle, &session_id, single_record_batch(0)).is_err());
        assert_eq!(
            handle.session.lock().records_written,
            0,
            "a rejected append must not be counted as written"
        );
    }
}
