//! Tiered record cache with automatic strategy selection (Issue #215).
//!
//! Wraps either a preloaded, LRU, or streaming cache based on file size
//! and available system memory.

use crate::analysis::utils::verbose_enabled;
use crate::types::DiscoverRecord;
use anyhow::Result;
use std::sync::Arc;

use super::RecordCache;
use super::loading_strategy::{LoadingStrategy, select_loading_strategy};
use super::lru_cache::LruRecordCache;
use crate::analysis::streaming::StreamingRecordCache;
use crate::analysis::utils::get_memory_info;

/// A tiered cache that automatically selects the best loading strategy.
///
/// This is a higher-level abstraction that wraps either a preloaded cache,
/// an LRU cache, or a streaming cache based on file size and available memory.
///
/// ## Usage
///
/// ```ignore
/// let cache = TieredRecordCache::new("data.parquet")?;
/// let records = cache.get("neuron-42")?;
/// ```
pub struct TieredRecordCache {
    /// The underlying cache implementation.
    inner: TieredCacheInner,
}

enum TieredCacheInner {
    Preloaded(RecordCache),
    Lru(Arc<LruRecordCache>),
    Streaming(StreamingRecordCache),
}

impl TieredRecordCache {
    /// Create a tiered cache with automatic strategy selection.
    pub fn new(parquet_file: &str) -> Result<Self> {
        let file_size = std::fs::metadata(parquet_file)?.len();
        let (available, _total) = get_memory_info();

        let strategy = select_loading_strategy(file_size, available);

        if verbose_enabled() {
            tracing::debug!(?strategy, "TieredRecordCache selected loading strategy");
        }

        let inner = match strategy {
            LoadingStrategy::PreloadAll => {
                TieredCacheInner::Preloaded(RecordCache::new_adaptive(parquet_file)?)
            }
            LoadingStrategy::LruCache { capacity_bytes } => {
                TieredCacheInner::Lru(Arc::new(LruRecordCache::new(parquet_file, capacity_bytes)?))
            }
            LoadingStrategy::Streaming => {
                TieredCacheInner::Streaming(StreamingRecordCache::new(parquet_file, None, None)?)
            }
        };

        Ok(Self { inner })
    }

    /// Create a tiered cache with a specific memory limit.
    /// This forces LRU mode with the specified capacity.
    pub fn new_with_memory_limit(parquet_file: &str, max_memory_bytes: usize) -> Result<Self> {
        let lru = LruRecordCache::new(parquet_file, max_memory_bytes)?;
        Ok(Self {
            inner: TieredCacheInner::Lru(Arc::new(lru)),
        })
    }

    /// Get records for a specific neuron UUID.
    pub fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        match &self.inner {
            TieredCacheInner::Preloaded(cache) => cache.get(neuron_uuid),
            TieredCacheInner::Lru(cache) => cache.get(neuron_uuid),
            TieredCacheInner::Streaming(cache) => cache.get(neuron_uuid),
        }
    }

    /// Get current cache statistics.
    ///
    /// Returns a unified stats structure regardless of the underlying cache type.
    pub fn stats(&self) -> TieredCacheStats {
        match &self.inner {
            TieredCacheInner::Preloaded(cache) => TieredCacheStats {
                strategy: "preloaded".to_string(),
                cached_neurons: cache.len(),
                current_bytes: 0, // Not tracked for preloaded
                cache_hits: 0,
                cache_misses: 0,
                eviction_count: 0,
            },
            TieredCacheInner::Lru(cache) => {
                let s = cache.stats();
                TieredCacheStats {
                    strategy: "lru".to_string(),
                    cached_neurons: s.cached_neurons,
                    current_bytes: s.current_bytes,
                    cache_hits: s.cache_hits,
                    cache_misses: s.cache_misses,
                    eviction_count: s.eviction_count,
                }
            }
            TieredCacheInner::Streaming(cache) => {
                let s = cache.stats();
                TieredCacheStats {
                    strategy: "streaming".to_string(),
                    cached_neurons: 0, // Streaming uses block-based tracking
                    current_bytes: s.cached_bytes,
                    cache_hits: s.cache_hits,
                    cache_misses: s.cache_misses,
                    eviction_count: s.eviction_count,
                }
            }
        }
    }
}

/// Unified statistics for tiered cache.
#[derive(Debug, Clone)]
pub struct TieredCacheStats {
    /// The loading strategy being used.
    pub strategy: String,
    /// Number of neurons currently cached (for LRU/preloaded).
    pub cached_neurons: usize,
    /// Current memory usage in bytes.
    pub current_bytes: usize,
    /// Number of cache hits.
    pub cache_hits: u64,
    /// Number of cache misses.
    pub cache_misses: u64,
    /// Number of evictions.
    pub eviction_count: u64,
}
