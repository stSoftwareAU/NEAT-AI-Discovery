//! LRU record cache for neuron discovery records (Issue #215).
//!
//! Per-neuron caching with LRU eviction, bounded memory, and thread-safe access.

use crate::analysis::utils::verbose_enabled;
use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
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
    /// Estimated size in bytes.
    size_bytes: usize,
}

impl LruCacheEntry {
    fn new(records: Vec<DiscoverRecord>) -> Self {
        let size_bytes = estimate_records_size(&records);
        Self {
            records: Arc::new(records),
            last_access: Instant::now(),
            size_bytes,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
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
/// - **Bounded memory**: Total cache size respects the configured capacity
/// - **Thread-safe**: Uses RwLock for concurrent access
pub struct LruRecordCache {
    /// Path to the parquet file.
    parquet_file: String,
    /// Maximum capacity in bytes.
    capacity_bytes: usize,
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
}

impl LruRecordCache {
    /// Create a new LRU record cache.
    ///
    /// # Arguments
    ///
    /// * `parquet_file` - Path to the parquet file
    /// * `capacity_bytes` - Maximum memory to use for caching (in bytes)
    pub fn new(parquet_file: &str, capacity_bytes: usize) -> Result<Self> {
        // Verify the file exists
        std::fs::metadata(parquet_file)
            .with_context(|| format!("Parquet file not found: {parquet_file}"))?;

        if verbose_enabled() {
            let capacity_mb = capacity_bytes as f64 / (1024.0 * 1024.0);
            tracing::debug!(capacity_mb, "creating LRU cache");
        }

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            capacity_bytes,
            cache: RwLock::new(HashMap::new()),
            current_bytes: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
        })
    }

    /// Get records for a specific neuron UUID.
    ///
    /// Returns cached records if available, otherwise loads from parquet.
    /// Evicts least-recently-used neurons if capacity is exceeded.
    pub fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        // Fast path: check cache with read lock
        {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get_mut(neuron_uuid) {
                entry.touch();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::clone(&entry.records));
            }
        }

        // Cache miss - load from parquet
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        let records = self.load_neuron_records(neuron_uuid)?;
        let entry = LruCacheEntry::new(records);
        let size = entry.size_bytes;
        let records_arc = Arc::clone(&entry.records);

        // Evict if necessary and insert new entry
        {
            let mut cache = self.cache.write();

            // Evict LRU entries until we have space
            while self.current_bytes.load(Ordering::Relaxed) + size > self.capacity_bytes {
                if let Some(lru_key) = self.find_lru_key(&cache) {
                    if let Some(evicted) = cache.remove(&lru_key) {
                        self.current_bytes
                            .fetch_sub(evicted.size_bytes, Ordering::Relaxed);
                        self.eviction_count.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    break;
                }
            }

            // Insert new entry
            self.current_bytes.fetch_add(size, Ordering::Relaxed);
            cache.insert(neuron_uuid.to_string(), entry);
        }

        Ok(records_arc)
    }

    /// Find the least-recently-used cache key.
    fn find_lru_key(&self, cache: &HashMap<String, LruCacheEntry>) -> Option<String> {
        cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
    }

    /// Load records for a specific neuron from parquet.
    fn load_neuron_records(&self, neuron_uuid: &str) -> Result<Vec<DiscoverRecord>> {
        use crate::parquet_format::read_records_from_parquet;
        read_records_from_parquet(&self.parquet_file, neuron_uuid)
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
}
