//! LRU record cache for neuron discovery records (Issue #215, #1088).
//!
//! Per-neuron caching with LRU eviction, bounded memory, and thread-safe access.
//! Enforces a hard entry-count limit (`max_entries`) as well as a byte-based
//! capacity limit.  When either limit is reached the least-recently-used entry
//! is evicted before the new one is inserted.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::analysis::utils::verbose_enabled;
use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

/// Statistics for LRU record cache operations.
#[derive(Debug, Clone, Default)]
pub struct LruCacheStats {
    /// Number of neurons currently cached.
    pub cached_neurons: usize,
    /// Current memory usage in bytes.
    pub current_bytes: usize,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Number of neurons evicted.
    pub eviction_count: u64,
}

/// A cached neuron entry with LRU tracking.
struct LruCacheEntry {
    /// The cached records for this neuron.
    records: Arc<Vec<DiscoverRecord>>,
    /// Last access time for LRU ordering.
    last_access: Instant,
    /// When this entry was first inserted.
    inserted_at: Instant,
    /// Number of times this entry has been accessed (hits).
    access_count: u64,
    /// Estimated size in bytes.
    size_bytes: usize,
}

impl LruCacheEntry {
    fn new(records: Vec<DiscoverRecord>) -> Self {
        let size_bytes = estimate_records_size(&records);
        Self {
            records: Arc::new(records),
            last_access: Instant::now(),
            inserted_at: Instant::now(),
            access_count: 0,
            size_bytes,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
        self.access_count += 1;
    }
}

/// Estimate the memory size of a vector of records.
pub(crate) fn estimate_records_size(records: &[DiscoverRecord]) -> usize {
    records
        .iter()
        .map(|r| {
            std::mem::size_of::<DiscoverRecord>()
                + r.neuron_uuid.len()
                + r.errors.len() * std::mem::size_of::<f32>()
        })
        .sum()
}

type RecordLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;

/// An LRU cache for neuron discovery records.
///
/// Unlike the block-based `StreamingRecordCache`, this cache operates at the
/// neuron level, which is more appropriate for workloads that access specific
/// neurons repeatedly (e.g., during analysis of focus neurons).
///
/// ## Features
///
/// - **Per-neuron caching**: Each neuron's records are cached as a unit
/// - **LRU eviction**: Least-recently-used neurons are evicted when capacity exceeded
/// - **Hard entry limit** (Issue #1088): No more than `max_entries` entries at any time
/// - **Bounded memory**: Total cache size respects the configured byte capacity
/// - **Thread-safe**: Uses `RwLock` for concurrent access from Rayon threads
pub struct LruRecordCache {
    /// Path to the parquet file.
    parquet_file: String,
    /// Maximum capacity in bytes.
    capacity_bytes: usize,
    /// Hard limit on the number of cached entries (Issue #1088).
    /// `None` means no entry-count limit (only byte-based eviction).
    max_entries: Option<usize>,
    /// Cached neurons, keyed by neuron UUID.
    cache: RwLock<HashMap<String, LruCacheEntry>>,
    /// Current memory usage in bytes.
    current_bytes: AtomicUsize,
    /// Cache hit counter.
    cache_hits: AtomicU64,
    /// Cache miss counter.
    cache_misses: AtomicU64,
    /// Eviction counter.
    eviction_count: AtomicU64,
    /// Custom loader function (default: parquet reader).
    loader: Arc<RecordLoader>,
    /// Held open to prevent the OS from reclaiming the file data while the
    /// cache is alive (Issue #1048).
    _file_guard: Option<File>,
}

impl LruRecordCache {
    /// Create a new LRU record cache.
    ///
    /// # Arguments
    ///
    /// * `parquet_file` - Path to the parquet file
    /// * `capacity_bytes` - Maximum memory to use for caching (in bytes)
    pub fn new(parquet_file: &str, capacity_bytes: usize) -> Result<Self> {
        // Issue #1048: Open the file as a guard to keep the inode alive.
        let file_guard = File::open(parquet_file)
            .with_context(|| format!("Parquet file not found: {parquet_file}"))?;

        let max_entries = crate::config::max_cached_blocks();

        if verbose_enabled() {
            let capacity_mb = capacity_bytes as f64 / (1024.0 * 1024.0);
            tracing::debug!(capacity_mb, ?max_entries, "creating LRU cache");
        }

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            capacity_bytes,
            max_entries,
            cache: RwLock::new(HashMap::new()),
            current_bytes: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
            loader: Arc::new(|file, neuron_uuid| {
                crate::parquet_format::read_records_from_parquet(file, neuron_uuid)
            }),
            _file_guard: Some(file_guard),
        })
    }

    /// Create an LRU cache with an explicit entry-count limit (Issue #1088).
    ///
    /// This is useful for tests and for callers that want to enforce a hard
    /// limit on the number of cached entries regardless of byte capacity.
    ///
    /// # Arguments
    ///
    /// * `parquet_file` - Path to the parquet file
    /// * `capacity_bytes` - Maximum memory to use for caching (in bytes)
    /// * `max_entries` - Hard limit on the number of cached entries
    pub fn new_with_max_entries(
        parquet_file: &str,
        capacity_bytes: usize,
        max_entries: usize,
    ) -> Result<Self> {
        let file_guard = File::open(parquet_file)
            .with_context(|| format!("Parquet file not found: {parquet_file}"))?;

        if verbose_enabled() {
            let capacity_mb = capacity_bytes as f64 / (1024.0 * 1024.0);
            tracing::debug!(
                capacity_mb,
                max_entries,
                "creating LRU cache with entry limit"
            );
        }

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            capacity_bytes,
            max_entries: Some(max_entries),
            cache: RwLock::new(HashMap::new()),
            current_bytes: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
            loader: Arc::new(|file, neuron_uuid| {
                crate::parquet_format::read_records_from_parquet(file, neuron_uuid)
            }),
            _file_guard: Some(file_guard),
        })
    }

    /// Create a cache with a custom loader function (Issue #1088).
    ///
    /// Used primarily for testing.  The loader is called on cache miss with the
    /// parquet file path and the neuron UUID, and should return the records.
    pub fn with_loader(
        parquet_file: &str,
        capacity_bytes: usize,
        max_entries: Option<usize>,
        loader: Arc<RecordLoader>,
    ) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            capacity_bytes,
            max_entries,
            cache: RwLock::new(HashMap::new()),
            current_bytes: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
            loader,
            _file_guard: None, // No file guard needed for test loaders
        }
    }

    /// Get records for a specific neuron UUID.
    ///
    /// Returns cached records if available, otherwise loads from parquet.
    /// Evicts least-recently-used neurons if capacity is exceeded.
    pub fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        // Fast path: check cache with write lock (need to update last_access)
        {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get_mut(neuron_uuid) {
                entry.touch();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.log_stats_debug();
                return Ok(Arc::clone(&entry.records));
            }
        }

        // Cache miss - load from parquet
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        let records = (self.loader)(&self.parquet_file, neuron_uuid)?;
        let entry = LruCacheEntry::new(records);
        let size = entry.size_bytes;
        let records_arc = Arc::clone(&entry.records);

        // Evict if necessary and insert new entry
        {
            let mut cache = self.cache.write();
            self.evict_if_needed(&mut cache, size);

            // Insert new entry
            self.current_bytes.fetch_add(size, Ordering::Relaxed);
            cache.insert(neuron_uuid.to_string(), entry);
        }

        self.log_stats_debug();
        Ok(records_arc)
    }

    /// Evict LRU entries until both the byte-capacity and entry-count limits
    /// have room for a new insertion.
    fn evict_if_needed(&self, cache: &mut HashMap<String, LruCacheEntry>, incoming_bytes: usize) {
        // Evict for byte capacity
        while self.current_bytes.load(Ordering::Relaxed) + incoming_bytes > self.capacity_bytes {
            if !self.evict_one(cache) {
                break;
            }
        }

        // Evict for entry-count hard limit (Issue #1088)
        if let Some(max) = self.max_entries {
            while cache.len() >= max {
                if !self.evict_one(cache) {
                    break;
                }
            }
        }
    }

    /// Evict the single least-recently-used entry.  Returns `true` if an entry
    /// was evicted, `false` if the cache was empty.
    fn evict_one(&self, cache: &mut HashMap<String, LruCacheEntry>) -> bool {
        if let Some(lru_key) = Self::find_lru_key(cache)
            && let Some(evicted) = cache.remove(&lru_key)
        {
            let age = evicted.inserted_at.elapsed();
            let access_count = evicted.access_count;

            self.current_bytes
                .fetch_sub(evicted.size_bytes, Ordering::Relaxed);
            self.eviction_count.fetch_add(1, Ordering::Relaxed);

            tracing::info!(
                neuron = %lru_key,
                age_secs = age.as_secs(),
                access_count,
                size_bytes = evicted.size_bytes,
                "LRU cache eviction under memory pressure"
            );
            return true;
        }
        false
    }

    /// Find the least-recently-used cache key.
    fn find_lru_key(cache: &HashMap<String, LruCacheEntry>) -> Option<String> {
        cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
    }

    /// Log cache statistics at debug level (Issue #1088).
    fn log_stats_debug(&self) {
        if verbose_enabled() {
            let hits = self.cache_hits.load(Ordering::Relaxed);
            let misses = self.cache_misses.load(Ordering::Relaxed);
            let evictions = self.eviction_count.load(Ordering::Relaxed);
            let current_bytes = self.current_bytes.load(Ordering::Relaxed);
            tracing::debug!(
                hits,
                misses,
                evictions,
                current_bytes,
                "LRU cache statistics"
            );
        }
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> LruCacheStats {
        let cache = self.cache.read();
        LruCacheStats {
            cached_neurons: cache.len(),
            current_bytes: self.current_bytes.load(Ordering::Relaxed),
            cache_hits: self.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.cache_misses.load(Ordering::Relaxed),
            eviction_count: self.eviction_count.load(Ordering::Relaxed),
        }
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Get the number of cached neurons.
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_records_size_basic() {
        use crate::types::DiscoverRecord;

        let records = vec![
            DiscoverRecord::new(0, "test-neuron".to_string(), Some(0.5), 0.5, vec![0.1, 0.2]),
            DiscoverRecord::new(1, "test-neuron".to_string(), Some(0.6), 0.6, vec![0.3]),
        ];

        let size = estimate_records_size(&records);
        // Size should be non-zero and reasonable
        assert!(size > 0);
        assert!(size < 1024 * 1024); // Less than 1MB for 2 records
    }

    /// Helper: create a test cache with a custom loader and a hard entry limit.
    fn make_test_cache(max_entries: usize) -> LruRecordCache {
        LruRecordCache::with_loader(
            "test.parquet",
            usize::MAX, // No byte limit — only entry limit matters
            Some(max_entries),
            Arc::new(|_file, neuron_uuid| {
                Ok(vec![DiscoverRecord::new(
                    0,
                    neuron_uuid.to_string(),
                    Some(1.0),
                    1.0,
                    vec![0.0],
                )])
            }),
        )
    }

    #[test]
    fn eviction_occurs_at_configured_limit() {
        let cache = make_test_cache(3);

        // Insert 3 entries — should all fit
        cache.get("neuron-a").unwrap();
        cache.get("neuron-b").unwrap();
        cache.get("neuron-c").unwrap();
        assert_eq!(cache.len(), 3);

        // Insert a 4th — should trigger eviction, keeping count at 3
        cache.get("neuron-d").unwrap();
        assert_eq!(cache.len(), 3);

        let stats = cache.stats();
        assert_eq!(stats.eviction_count, 1);
        assert_eq!(stats.cache_misses, 4);
    }

    #[test]
    fn lru_ordering_evicts_oldest_accessed() {
        let cache = make_test_cache(3);

        // Insert a, b, c in order
        cache.get("neuron-a").unwrap();
        cache.get("neuron-b").unwrap();
        cache.get("neuron-c").unwrap();

        // Touch "neuron-a" so it is most-recently-used
        cache.get("neuron-a").unwrap();

        // Insert "neuron-d" — should evict "neuron-b" (the LRU entry)
        cache.get("neuron-d").unwrap();
        assert_eq!(cache.len(), 3);

        // neuron-a should still be cached (was recently accessed)
        cache.get("neuron-a").unwrap();
        let stats = cache.stats();
        // neuron-a: miss(1) + hit(1) + hit(1) = 2 hits for neuron-a
        // neuron-b: miss(1), then evicted
        // neuron-c: miss(1)
        // neuron-d: miss(1)
        // Total misses = 4 (a, b, c, d), hits = 2 (a touch, a final get)
        assert_eq!(stats.cache_hits, 2);
        assert_eq!(stats.cache_misses, 4);

        // neuron-b should have been evicted (cache miss if accessed again)
        let hits_before = cache.stats().cache_hits;
        cache.get("neuron-b").unwrap();
        let hits_after = cache.stats().cache_hits;
        // Accessing neuron-b should be a miss (not a hit)
        assert_eq!(hits_after, hits_before);
    }

    #[test]
    fn cache_statistics_are_tracked() {
        let cache = make_test_cache(10);

        cache.get("neuron-x").unwrap(); // miss
        cache.get("neuron-x").unwrap(); // hit
        cache.get("neuron-y").unwrap(); // miss
        cache.get("neuron-y").unwrap(); // hit
        cache.get("neuron-y").unwrap(); // hit

        let stats = cache.stats();
        assert_eq!(stats.cache_misses, 2);
        assert_eq!(stats.cache_hits, 3);
        assert_eq!(stats.cached_neurons, 2);
        assert_eq!(stats.eviction_count, 0);
    }

    #[test]
    fn thread_safe_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let cache = Arc::new(make_test_cache(50));
        let mut handles = Vec::new();

        for i in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for j in 0..20 {
                    let uuid = format!("neuron-{}-{}", i, j % 5);
                    cache.get(&uuid).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let stats = cache.stats();
        // Every thread did 20 gets; at least some should be hits
        assert!(stats.cache_hits + stats.cache_misses > 0);
        // No more than 50 entries
        assert!(cache.len() <= 50);
    }

    #[test]
    fn max_entries_of_one_always_evicts() {
        let cache = make_test_cache(1);

        cache.get("a").unwrap();
        assert_eq!(cache.len(), 1);

        cache.get("b").unwrap();
        assert_eq!(cache.len(), 1);

        cache.get("c").unwrap();
        assert_eq!(cache.len(), 1);

        let stats = cache.stats();
        assert_eq!(stats.eviction_count, 2);
    }
}
