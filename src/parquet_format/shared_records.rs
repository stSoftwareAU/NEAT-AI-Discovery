//! Process-wide single-decode cache for grouped discovery records (Issue #1406).
//!
//! Focus selection and the analysis phase are two separate FFI calls that each
//! used to read the same parquet file from scratch. On large files that second
//! full decode — the "parquet reload" — consumed a meaningful slice of the
//! analysis deadline before any synapse / neuron work began.
//!
//! This module decodes the grouped discovery records **once per discovery
//! cycle** and shares them across both phases. The cache holds a single entry
//! keyed by the parquet file's path, device/inode, byte length, and
//! modification time. When the second phase requests the same file, the
//! already-decoded records are returned without touching disk. A changed file (a
//! new recording produces a new mtime / size, a replacement produces a new
//! inode) invalidates the entry automatically, and back-to-back cycles on
//! different files reuse the single slot.
//!
//! The stored key describes the bytes actually in the cache (Issue #1907): the
//! file is `stat`ed again **after** the decode and the records are cached only
//! when that identity is unchanged across the whole stat → decode → stat window.
//! A file that cannot be `stat`ed has no identity at all, so it is never
//! cacheable — a placeholder key would let two consecutive failures match.
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
/// only when the path, device/inode, byte length, and modification time all
/// match.
#[derive(Clone, PartialEq, Eq)]
struct CacheKey {
    path: String,
    /// `(dev, ino)` on unix, distinguishing a replacement file that reproduces
    /// the original's length and mtime. `None` on platforms without the unix
    /// metadata extension.
    dev_ino: Option<(u64, u64)>,
    len: u64,
    mtime_nanos: u128,
}

impl CacheKey {
    /// Derive the cache key from the file's current metadata, or `None` when no
    /// identity can be established — a failed `stat`, or an mtime the platform
    /// cannot express as a duration since the unix epoch. An unknown identity is
    /// deliberately not cacheable: collapsing failures onto a placeholder key
    /// would let one transient failure seed an entry that a second transient
    /// failure then hits (Issue #1907).
    fn for_path(path: &str) -> Option<Self> {
        let meta = std::fs::metadata(path).ok()?;
        let mtime_nanos = meta
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        Some(Self {
            path: path.to_string(),
            dev_ino: dev_ino(&meta),
            len: meta.len(),
            mtime_nanos,
        })
    }
}

#[cfg(unix)]
fn dev_ino(meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    use std::os::unix::fs::MetadataExt;
    Some((meta.dev(), meta.ino()))
}

#[cfg(not(unix))]
fn dev_ino(_meta: &std::fs::Metadata) -> Option<(u64, u64)> {
    None
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
/// the same file (path + device/inode + length + mtime) was already loaded this
/// cycle.
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
    load_shared_with_decoder(path, || {
        read_all_records_grouped_by_neuron_bounded(
            path,
            deadline,
            crate::parquet_format::ColumnProfile::Full,
            budget_mb,
        )
    })
}

/// Shared cache logic, parameterised by the decode so tests can act inside the
/// stat → decode window (Issue #1907).
fn load_shared_with_decoder<F>(path: &str, decode: F) -> Result<Arc<SharedGroupedRecords>>
where
    F: FnOnce() -> Result<HashMap<String, Vec<DiscoverRecord>>>,
{
    let key_before = CacheKey::for_path(path);

    // Fast path: return the cached decode when the file identity is unchanged.
    if let Some(key) = key_before.as_ref() {
        let slot = CACHE.lock();
        if let Some(existing) = slot.as_ref()
            && existing.key == *key
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
    let grouped = decode()?;
    let shared: Arc<SharedGroupedRecords> = Arc::new(
        grouped
            .into_iter()
            .map(|(uuid, recs)| (uuid, Arc::new(recs)))
            .collect(),
    );

    // The cached records must be described by the key they are stored under, so
    // re-stat after the decode: cache only when the identity held for the whole
    // window, and never when it is unknown at either end.
    let Some(key) = CacheKey::for_path(path).filter(|after| key_before.as_ref() == Some(after))
    else {
        tracing::debug!(
            target: "neat_ai_discovery::parquet_format::shared_records",
            path,
            "parquet identity unknown or changed during the decode — returning records uncached",
        );
        return Ok(shared);
    };

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

#[cfg(test)]
mod tests {
    use super::{
        SharedGroupedRecords, decodes_for_path, invalidate, load_shared_with_decoder,
        read_all_records_grouped_by_neuron_bounded,
    };
    use crate::parquet_format::{ColumnProfile, write_records_to_parquet};
    use crate::types::DiscoverRecord;
    use serial_test::serial;
    use std::sync::Arc;
    use tempfile::NamedTempFile;

    fn records(uuid: &str, count: u32) -> Vec<DiscoverRecord> {
        (0..count)
            .map(|i| DiscoverRecord::new(i, uuid.to_string(), Some(0.5), 0.7, vec![0.1]))
            .collect()
    }

    /// The file is rewritten inside the stat → decode window, so the identity
    /// sampled before the decode no longer describes the decoded bytes. The
    /// records must be returned but never cached, so no later caller can hit an
    /// entry keyed to the superseded identity.
    #[test]
    #[serial]
    fn rewrite_between_stat_and_decode_does_not_serve_stale_identity() {
        let tmp = NamedTempFile::new().expect("create temp file");
        let path = tmp.path().to_str().expect("utf-8 path").to_string();
        write_records_to_parquet(&path, &records("neuron-old", 4)).expect("write original");
        invalidate();

        let decoded: Arc<SharedGroupedRecords> = load_shared_with_decoder(&path, || {
            // The stat has happened; rewrite the file before the decode reads it.
            write_records_to_parquet(&path, &records("neuron-new", 32)).expect("rewrite");
            read_all_records_grouped_by_neuron_bounded(&path, None, ColumnProfile::Full, None)
        })
        .expect("load succeeds");

        assert!(
            decoded.contains_key("neuron-new"),
            "the caller must receive the bytes that were actually decoded",
        );
        assert_eq!(
            decodes_for_path(&path),
            0,
            "a decode whose file identity changed mid-window must not be cached",
        );

        // A later load re-reads the file rather than serving the racy decode.
        let reloaded = super::load_grouped_records_shared(&path, None).expect("reload succeeds");
        assert!(
            !Arc::ptr_eq(&decoded, &reloaded),
            "the racy decode must not be handed back from the cache",
        );
        assert_eq!(decodes_for_path(&path), 1, "the reload decodes fresh");

        invalidate();
    }
}
