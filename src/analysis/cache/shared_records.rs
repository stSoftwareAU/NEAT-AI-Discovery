//! Process-side shared cache of grouped discovery records (Issue #1406).
//!
//! Focus selection (`rank_focus_neurons`) and the analysis phase
//! (`analyze_all`) are two separate FFI calls that historically each read the
//! same parquet file from scratch. On large files the second full scan — the
//! "parquet reload" — consumed a meaningful slice of the analysis deadline
//! before any synapse/neuron analysis began.
//!
//! This module bridges the two calls: the first phase to load a parquet file
//! stores the grouped records (one `Arc` per neuron) in a single-entry global
//! keyed by `(path, mtime, size)`. The second phase reuses them when the file
//! is unchanged, so the parquet is fully read **at most once per cycle** when
//! focus selection and analysis run back-to-back on the same file.
//!
//! ## Invalidation
//!
//! The key includes the file's modification time and size, so any change to
//! the parquet — including a new cycle writing a fresh temp file at the same
//! path — misses the cache and triggers a fresh load. Only the most recent
//! entry is retained; loading a different file drops the previous one.
//!
//! ## Memory lifetime
//!
//! The global holds only `Arc` handles. Once a consumer has built its own
//! cache from the shared records it should call [`clear`] so the bridge no
//! longer pins the data; the consumer's own `Arc`s keep the records alive for
//! exactly as long as that phase needs them, matching the pre-#1406 lifetime.

use crate::parquet_format::read_all_records_grouped_by_neuron_with_deadline;
use crate::types::DiscoverRecord;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;

/// Grouped discovery records keyed by neuron UUID, each neuron's records sorted
/// by `obs_index`. Shared via `Arc` between focus selection and `analyze_all`.
pub type SharedGroupedRecords = Arc<HashMap<String, Arc<Vec<DiscoverRecord>>>>;

/// Identity of a cached parquet load. Two loads share records only when the
/// path, modification time, and size all match.
#[derive(Clone, PartialEq, Eq)]
struct CacheKey {
    path: String,
    mtime_ns: u128,
    size: u64,
}

struct Entry {
    key: CacheKey,
    records: SharedGroupedRecords,
}

static SHARED: Mutex<Option<Entry>> = Mutex::new(None);

/// Lock the global, recovering from a poisoned mutex. The critical sections are
/// tiny and panic-free, so poisoning should not occur in practice; recovering
/// rather than propagating keeps the bridge usable instead of cascading
/// failures into the analysis pipeline.
fn lock_shared() -> std::sync::MutexGuard<'static, Option<Entry>> {
    SHARED
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// Build the `(path, mtime, size)` key for `path`. Returns `None` when the file
/// metadata cannot be read, in which case callers fall back to a fresh load
/// without caching.
fn cache_key(path: &str) -> Option<CacheKey> {
    let meta = std::fs::metadata(path).ok()?;
    let size = meta.len();
    let mtime_ns = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())?;
    Some(CacheKey {
        path: path.to_string(),
        mtime_ns,
        size,
    })
}

/// Return the shared records for `path` when the bridge holds an entry whose
/// key matches the file's current metadata, otherwise `None`.
#[must_use]
pub fn get(path: &str) -> Option<SharedGroupedRecords> {
    let key = cache_key(path)?;
    let guard = lock_shared();
    guard
        .as_ref()
        .filter(|entry| entry.key == key)
        .map(|entry| Arc::clone(&entry.records))
}

/// Store `records` under `path`'s current metadata key, replacing any previous
/// entry (the bridge retains only the most recent file).
fn store(path: &str, records: &SharedGroupedRecords) {
    if let Some(key) = cache_key(path) {
        *lock_shared() = Some(Entry {
            key,
            records: Arc::clone(records),
        });
    }
}

/// Drop the bridged records so the global no longer pins them. Consumers call
/// this once they hold their own `Arc`s to the records.
pub fn clear() {
    *lock_shared() = None;
}

/// Load the grouped records for `path`, reusing the shared bridge when the file
/// is unchanged since the previous phase loaded it.
///
/// On a cache miss the parquet is read once (honouring `deadline`), each
/// neuron's records are sorted by `obs_index` (matching the per-neuron load
/// path so downstream ordering is identical), and the result is stored for the
/// next phase before being returned.
///
/// # Errors
///
/// Propagates any error from reading the parquet file (including deadline or
/// cancellation aborts).
pub fn get_or_load_with_deadline(
    path: &str,
    deadline: Option<SystemTime>,
) -> Result<SharedGroupedRecords> {
    if let Some(hit) = get(path) {
        return Ok(hit);
    }

    let grouped = read_all_records_grouped_by_neuron_with_deadline(path, deadline)?;

    let shared: SharedGroupedRecords = Arc::new(
        grouped
            .into_iter()
            .map(|(uuid, mut recs)| {
                recs.sort_by_key(|r| r.obs_index);
                (uuid, Arc::new(recs))
            })
            .collect(),
    );

    store(path, &shared);
    Ok(shared)
}
