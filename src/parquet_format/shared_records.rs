//! Process-wide single-decode cache for grouped discovery records (Issue #1406).
//!
//! Focus selection and the analysis phase are two separate FFI calls that each
//! used to read the same parquet file from scratch. On large files that second
//! full decode — the "parquet reload" — consumed a meaningful slice of the
//! analysis deadline before any synapse / neuron work began.
//!
//! This module decodes the grouped discovery records **once per discovery
//! cycle** and shares them across both phases. The cache holds a single entry
//! keyed by the parquet file's path, byte length, and modification time. When
//! the second phase requests the same file, the already-decoded records are
//! returned without touching disk. A changed file (a new recording produces a
//! new mtime / size) invalidates the entry automatically, and back-to-back
//! cycles on different files reuse the single slot.
//!
//! Records are stored behind nested `Arc`s so the analysis phase can reuse each
//! neuron's records without copying. Focus ranking sorts its own copy by
//! `obs_index`; the analysis cache consumes the records in decode order. Both
//! orderings match the pre-cache behaviour, so no analysis or focus output
//! changes.

use crate::parquet_format::read_all_records_grouped_by_neuron_bounded;
use crate::types::DiscoverRecord;
use anyhow::Result;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Grouped discovery records keyed by neuron UUID. Each neuron's records are
/// shared behind an `Arc` so consumers reuse them without copying.
pub type SharedGroupedRecords = HashMap<String, Arc<Vec<DiscoverRecord>>>;

/// Identity of a cached parquet decode. Two requests share the cached records
/// only when the path, byte length, and modification time all match.
#[derive(Clone, PartialEq, Eq)]
struct CacheKey {
    path: String,
    len: u64,
    mtime_nanos: Option<u128>,
}

impl CacheKey {
    /// Derive the cache key from the file's current metadata. Missing metadata
    /// (e.g. the path no longer exists) falls back to a zero length and absent
    /// mtime, so a later successful stat naturally counts as a different key.
    fn for_path(path: &str) -> Self {
        let meta = std::fs::metadata(path).ok();
        let len = meta.as_ref().map_or(0, std::fs::Metadata::len);
        let mtime_nanos = meta
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos());
        Self {
            path: path.to_string(),
            len,
            mtime_nanos,
        }
    }
}

/// The single cached decode plus a count of how many times the currently
/// resident file has been decoded. The count resets whenever a different file
/// takes the slot, keeping it bounded to the active cycle.
struct CacheSlot {
    key: CacheKey,
    records: Arc<SharedGroupedRecords>,
    decodes: u64,
}

static CACHE: LazyLock<Mutex<Option<CacheSlot>>> = LazyLock::new(|| Mutex::new(None));

/// Load the grouped discovery records for `path`, reusing a prior decode when
/// the same file (path + length + mtime) was already loaded this cycle.
///
/// On a cache hit the records are returned without any disk access, so a
/// discovery cycle that runs focus selection then analysis on the same parquet
/// decodes the file exactly once. On a miss the file is decoded via
/// [`read_all_records_grouped_by_neuron_bounded`], stored, and returned.
///
/// `deadline` applies only to the decode on a miss; a hit ignores it because no
/// loading occurs.
pub fn load_grouped_records_shared(
    path: &str,
    deadline: Option<SystemTime>,
) -> Result<Arc<SharedGroupedRecords>> {
    load_grouped_records_shared_with_budget(path, deadline, None)
}

/// Load the grouped discovery records for `path`, bounding a decode on a cache
/// miss with the caller's memory budget in megabytes (Issue #1869).
///
/// The budget applies only to the decode: a cache hit returns the already
/// materialised records, which were themselves bounded when first decoded.
pub fn load_grouped_records_shared_with_budget(
    path: &str,
    deadline: Option<SystemTime>,
    budget_mb: Option<u64>,
) -> Result<Arc<SharedGroupedRecords>> {
    let key = CacheKey::for_path(path);

    // Fast path: return the cached decode when the file identity is unchanged.
    {
        let slot = CACHE.lock();
        if let Some(existing) = slot.as_ref()
            && existing.key == key
        {
            tracing::debug!(
                target: "neat_ai_discovery::parquet_format::shared_records",
                path,
                "reusing shared grouped records (cache hit) — skipping parquet reload",
            );
            return Ok(Arc::clone(&existing.records));
        }
    }

    // Miss: decode the file outside the lock so callers for other files are not
    // blocked on this (potentially slow) read.
    let grouped = read_all_records_grouped_by_neuron_bounded(
        path,
        deadline,
        crate::parquet_format::ColumnProfile::Full,
        budget_mb,
    )?;
    let shared: Arc<SharedGroupedRecords> = Arc::new(
        grouped
            .into_iter()
            .map(|(uuid, recs)| (uuid, Arc::new(recs)))
            .collect(),
    );

    let mut slot = CACHE.lock();
    // Re-check under the lock: a concurrent caller may have decoded the same
    // file while we read. Reuse theirs and count both decodes for honesty.
    if let Some(existing) = slot.as_mut()
        && existing.key == key
    {
        existing.decodes = existing.decodes.saturating_add(1);
        return Ok(Arc::clone(&existing.records));
    }
    *slot = Some(CacheSlot {
        key,
        records: Arc::clone(&shared),
        decodes: 1,
    });
    Ok(shared)
}

/// Number of times the file currently cached for `path` has been decoded since
/// it took the cache slot. Returns `0` when a different file (or nothing) is
/// resident. Used to verify the parquet is decoded once across the focus and
/// analysis phases (Issue #1406).
#[must_use]
pub fn decodes_for_path(path: &str) -> u64 {
    let slot = CACHE.lock();
    slot.as_ref()
        .filter(|s| s.key.path == path)
        .map_or(0, |s| s.decodes)
}

/// Drop any cached records. Exposed for explicit invalidation and tests.
pub fn invalidate() {
    *CACHE.lock() = None;
}
