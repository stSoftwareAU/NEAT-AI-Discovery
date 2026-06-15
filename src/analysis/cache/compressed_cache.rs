//! LZ4-compressed LRU record cache (Issue #420).
//!
//! Compresses records using LZ4 before storing them, trading CPU time for
//! reduced memory usage. Typical compression ratios of 2-4x allow the cache
//! to hold significantly more neurons in the same memory footprint.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::analysis::utils::verbose_enabled;
use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use super::lru_cache::LruCacheStats;
use super::serialisation::CompressedCacheEntry;

/// An LZ4-compressed LRU cache for neuron discovery records (Issue #420).
///
/// This cache compresses records using LZ4 before storing them, trading CPU
/// time for reduced memory usage. This is particularly beneficial for
/// memory-constrained systems where the uncompressed cache would exhaust
/// available memory and trigger excessive evictions.
///
/// ## Memory Savings
///
/// Discovery records contain repetitive floating-point data (activations,
/// errors) that compresses well with LZ4. Typical compression ratios are
/// 2-4x, meaning the cache can hold 2-4x more neurons in the same memory.
///
/// ## Performance Trade-off
///
/// LZ4 decompression is fast (~4 GB/s on modern hardware), so the CPU
/// overhead is minimal compared to the I/O savings from fewer cache misses.
pub struct CompressedLruRecordCache {
    parquet_file: String,
    capacity_bytes: usize,
    cache: RwLock<HashMap<String, CompressedCacheEntry>>,
    current_bytes: AtomicUsize,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    eviction_count: AtomicU64,
    /// Held open to prevent the OS from reclaiming the file data while the
    /// cache is alive (Issue #1048).
    _file_guard: File,
}

impl CompressedLruRecordCache {
    /// Create a new compressed LRU record cache.
    ///
    /// # Arguments
    ///
    /// * `parquet_file` - Path to the parquet file
    /// * `capacity_bytes` - Maximum compressed memory to use for caching
    pub fn new(parquet_file: &str, capacity_bytes: usize) -> Result<Self> {
        // Issue #1048: Open the file as a guard to keep the inode alive.
        let file_guard = File::open(parquet_file)
            .with_context(|| format!("Parquet file not found: {parquet_file}"))?;

        if verbose_enabled() {
            let capacity_mb = capacity_bytes as f64 / (1024.0 * 1024.0);
            tracing::debug!(capacity_mb, "creating compressed LRU cache");
        }

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            capacity_bytes,
            cache: RwLock::new(HashMap::new()),
            current_bytes: AtomicUsize::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
            _file_guard: file_guard,
        })
    }

    /// Get records for a specific neuron UUID.
    ///
    /// Returns decompressed records. On cache hit, the compressed data is
    /// decompressed before returning. On cache miss, records are loaded from
    /// parquet, compressed, and stored in the cache.
    pub fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        // Check cache with write lock (need to update last_access)
        {
            let mut cache = self.cache.write();
            if let Some(entry) = cache.get_mut(neuron_uuid) {
                entry.touch();
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Ok(Arc::new(entry.decompress()?));
            }
        }

        // Cache miss — load from parquet
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let records = self.load_neuron_records(neuron_uuid)?;
        let entry = CompressedCacheEntry::new(&records)?;
        let size = entry.compressed_size;
        let result = Arc::new(records);

        // Evict if necessary and insert
        {
            let mut cache = self.cache.write();
            while self.current_bytes.load(Ordering::Relaxed) + size > self.capacity_bytes {
                if let Some(lru_key) = self.find_lru_key(&cache) {
                    if let Some(evicted) = cache.remove(&lru_key) {
                        self.current_bytes
                            .fetch_sub(evicted.compressed_size, Ordering::Relaxed);
                        self.eviction_count.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    break;
                }
            }
            self.current_bytes.fetch_add(size, Ordering::Relaxed);
            cache.insert(neuron_uuid.to_string(), entry);
        }

        Ok(result)
    }

    fn find_lru_key(&self, cache: &HashMap<String, CompressedCacheEntry>) -> Option<String> {
        cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(key, _)| key.clone())
    }

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
