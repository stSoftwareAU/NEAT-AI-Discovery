//! Record provider trait and implementations for discovery data access.
//!
//! Provides both eager (pre-loaded) and lazy (on-demand with bounded cache)
//! strategies for accessing recorded discovery data during neuron ranking.

use crate::analysis::utils::lock_or_bail;
use crate::parquet_format::read_records_from_parquet;
use crate::types::DiscoverRecord;
use anyhow::{Context, Result, anyhow};
use parking_lot::Mutex;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Provides access to recorded discovery data without assuming an in-memory `HashMap`.
/// Implementations may pre-load all records or stream them on demand with bounded caching.
pub trait RecordProvider: Send + Sync {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub(crate) type RecordLoader =
    dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;

pub(super) struct EagerRecordProvider {
    records: HashMap<String, Arc<Vec<DiscoverRecord>>>,
}

impl EagerRecordProvider {
    pub(super) fn new(records: HashMap<String, Vec<DiscoverRecord>>) -> Self {
        let records = records
            .into_iter()
            .map(|(uuid, mut recs)| {
                recs.sort_by_key(|r| r.obs_index);
                (uuid, Arc::new(recs))
            })
            .collect();
        Self { records }
    }
}

impl RecordProvider for EagerRecordProvider {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        Ok(self.records.get(neuron_uuid).map(Arc::clone))
    }

    fn len(&self) -> usize {
        self.records.len()
    }
}

pub(in crate::focus) struct LazyRecordProvider {
    parquet_file: String,
    cache: Mutex<LazyCache>,
    loader: Arc<RecordLoader>,
}

struct LazyCache {
    entries: HashMap<String, Arc<Vec<DiscoverRecord>>>,
    order: VecDeque<String>,
    capacity: usize,
}

impl LazyCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    fn insert(&mut self, key: String, value: Arc<Vec<DiscoverRecord>>) {
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        }

        self.entries.insert(key.clone(), value);
        self.order.push_back(key);
        self.evict();
    }

    fn evict(&mut self) {
        while self.entries.len() > self.capacity {
            if let Some(oldest) = self.order.pop_front() {
                self.entries.remove(&oldest);
            }
        }
    }
}

impl LazyRecordProvider {
    const DEFAULT_CACHE_CAPACITY: usize = 8;

    /// Create a lazy provider whose cache can hold `capacity` distinct neurons
    /// without eviction (Issue #1374).
    ///
    /// Sizing the cache to the ranking working set guarantees each neuron is
    /// materialised **at most once** per run, even though the ranking pipeline
    /// sweeps over every selectable neuron several times (margins, max output
    /// error, impacts, ranking). With the previous fixed 8-entry cache, a
    /// larger working set thrashed and every `get()` triggered a full-file
    /// parquet rescan — turning a 7s preload into over an hour in lazy mode.
    ///
    /// The capacity is floored at [`Self::DEFAULT_CACHE_CAPACITY`] so tiny
    /// working sets keep a small amount of headroom.
    pub(super) fn with_capacity(parquet_file: &str, capacity: usize) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(LazyCache::new(capacity.max(Self::DEFAULT_CACHE_CAPACITY))),
            loader: Arc::new(read_records_from_parquet),
        }
    }

    /// Pre-populate the cache from a single grouped parquet pass (Issue #1374).
    ///
    /// Warming the cache with one decode of the whole file replaces the
    /// `O(passes × neurons)` per-neuron full-file rescans the lazy loader would
    /// otherwise perform. Records are sorted by `obs_index` to match the
    /// per-neuron load path so downstream ordering is identical.
    ///
    /// The cache must be sized (via [`Self::with_capacity`]) to hold the seeded
    /// set, otherwise seeding would immediately evict its own entries.
    pub(in crate::focus) fn seed(
        &self,
        grouped: HashMap<String, Vec<DiscoverRecord>>,
    ) -> Result<()> {
        let mut cache = lock_or_bail(&self.cache, "lazy record cache")?;
        for (uuid, mut records) in grouped {
            if records.is_empty() {
                continue;
            }
            records.sort_by_key(|r| r.obs_index);
            cache.insert(uuid, Arc::new(records));
        }
        Ok(())
    }

    #[cfg(test)]
    pub(in crate::focus) fn with_loader_for_tests(
        parquet_file: &str,
        capacity: usize,
        loader: Arc<RecordLoader>,
    ) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(LazyCache::new(capacity)),
            loader,
        }
    }
}

impl RecordProvider for LazyRecordProvider {
    fn get(&self, neuron_uuid: &str) -> Result<Option<Arc<Vec<DiscoverRecord>>>> {
        {
            let cache = lock_or_bail(&self.cache, "lazy record cache")?;
            if let Some(records) = cache.entries.get(neuron_uuid) {
                return Ok(Some(Arc::clone(records)));
            }
        }

        let mut records = (self.loader)(&self.parquet_file, neuron_uuid).with_context(|| {
            format!(
                "Failed to load discovery records for {neuron_uuid} from {}",
                self.parquet_file
            )
        })?;
        if records.is_empty() {
            return Ok(None);
        }
        records.sort_by_key(|r| r.obs_index);
        let arc_records = Arc::new(records);

        let mut cache = lock_or_bail(&self.cache, "lazy record cache")?;
        cache.insert(neuron_uuid.to_string(), Arc::clone(&arc_records));
        Ok(Some(arc_records))
    }

    fn len(&self) -> usize {
        self.cache.lock().entries.len()
    }
}

pub(super) fn get_records_or_error(
    provider: &dyn RecordProvider,
    neuron_uuid: &str,
) -> Result<Arc<Vec<DiscoverRecord>>> {
    provider
        .get(neuron_uuid)?
        .ok_or_else(|| anyhow!("Missing discovery records for selectable neuron: {neuron_uuid}"))
}
