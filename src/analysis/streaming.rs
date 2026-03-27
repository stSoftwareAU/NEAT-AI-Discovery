//! Streaming Parquet loading with block-based caching and prefetch.
//!
//! This module implements Issue #193: Memory-efficient streaming loading for large
//! Parquet datasets. Instead of loading the entire file into memory, records are
//! loaded in blocks with LRU eviction and optional prefetching.
//!
//! ## Key Features
//!
//! - **Block-based loading**: Parquet row groups are loaded as logical blocks
//! - **LRU eviction**: Least-recently-used blocks are evicted when memory limit reached
//! - **Prefetch**: Background thread loads adjacent blocks predictively
//! - **Configurable**: Memory limits and prefetch depth via environment variables
//!
//! ## Configuration
//!
//! - `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS`: Maximum blocks in memory (default: adaptive)
//! - `NEAT_AI_DISCOVERY_PREFETCH_DEPTH`: How many blocks ahead to prefetch (default: 2)
//! - `NEAT_AI_DISCOVERY_PRELOAD_ALL`: Set to 1 to disable streaming (use full preload)
//! - `NEAT_AI_DISCOVERY_BLOCK_SIZE`: Records per block (default: 10000, min: 10)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

// =============================================================================
// Constants
// =============================================================================

/// Minimum records per block (for testing with small datasets).
const MIN_BLOCK_SIZE: usize = 10;

/// Maximum records per block (prevents excessively large blocks).
const MAX_BLOCK_SIZE: usize = 100_000;

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for streaming record cache.
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// Maximum number of blocks to keep in cache. None = adaptive based on RAM.
    pub max_cached_blocks: Option<usize>,
    /// How many blocks ahead to prefetch. 0 = disabled.
    pub prefetch_depth: Option<usize>,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            max_cached_blocks: None,
            prefetch_depth: Some(2),
        }
    }
}

/// Get streaming configuration from environment variables.
///
/// Delegates to [`crate::config`] accessors.
pub fn get_streaming_config_from_env() -> StreamingConfig {
    StreamingConfig {
        max_cached_blocks: crate::config::max_cached_blocks(),
        prefetch_depth: Some(crate::config::prefetch_depth()),
    }
}

/// Check if streaming mode is enabled (vs full preload).
///
/// Delegates to [`crate::config::streaming_enabled()`].
pub fn is_streaming_enabled() -> bool {
    crate::config::streaming_enabled()
}

/// Get the block size from environment or use default.
///
/// Delegates to [`crate::config::block_size()`].
fn get_block_size() -> usize {
    crate::config::block_size()
}

/// Calculate an adaptive block size based on available memory (Issue #420).
///
/// Smaller blocks use less memory per cached block, allowing more blocks to fit
/// in memory and reducing peak memory usage. Larger blocks are more efficient
/// for sequential access.
///
/// # Arguments
/// * `available_memory_bytes` - Available system memory in bytes
///
/// # Returns
/// Block size in records, clamped between `MIN_BLOCK_SIZE` and `MAX_BLOCK_SIZE`.
pub fn adaptive_block_size(available_memory_bytes: u64) -> usize {
    let available_gb = available_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

    // Scale block size with available memory:
    // < 2GB:   1,000 records (small blocks for tight memory)
    // 2-4GB:   2,500 records
    // 4-8GB:   5,000 records
    // 8-16GB:  10,000 records (default)
    // 16-32GB: 25,000 records
    // > 32GB:  50,000 records
    let block_size = if available_gb < 2.0 {
        1_000
    } else if available_gb < 4.0 {
        2_500
    } else if available_gb < 8.0 {
        5_000
    } else if available_gb < 16.0 {
        10_000
    } else if available_gb < 32.0 {
        25_000
    } else {
        50_000
    };

    block_size.clamp(MIN_BLOCK_SIZE, MAX_BLOCK_SIZE)
}

// =============================================================================
// Statistics
// =============================================================================

/// Statistics for streaming cache operations.
#[derive(Debug, Clone, Default)]
pub struct StreamingCacheStats {
    /// Number of blocks currently cached.
    pub cached_blocks: usize,
    /// Number of cache hits (block already loaded).
    pub cache_hits: u64,
    /// Number of cache misses (block needed loading).
    pub cache_misses: u64,
    /// Number of blocks evicted due to memory pressure.
    pub eviction_count: u64,
    /// Number of blocks prefetched.
    pub prefetch_count: u64,
    /// Total bytes currently cached.
    pub cached_bytes: usize,
}

// =============================================================================
// Block Management
// =============================================================================

/// A block of records loaded from a Parquet row group.
///
/// Each block contains records for one or more neurons, grouped by row group
/// in the Parquet file.
#[derive(Debug)]
struct CacheBlock {
    /// Records in this block, keyed by neuron UUID.
    records: HashMap<String, Arc<Vec<DiscoverRecord>>>,
    /// When this block was last accessed.
    last_access: Instant,
    /// Approximate memory size of this block in bytes.
    size_bytes: usize,
}

impl CacheBlock {
    fn new(records: HashMap<String, Vec<DiscoverRecord>>) -> Self {
        // Estimate memory size
        let size_bytes: usize = records
            .iter()
            .map(|(k, v)| {
                k.len()
                    + v.iter()
                        .map(|r| {
                            std::mem::size_of::<DiscoverRecord>()
                                + r.neuron_uuid.len()
                                + r.errors.len() * std::mem::size_of::<f32>()
                        })
                        .sum::<usize>()
            })
            .sum();

        Self {
            records: records.into_iter().map(|(k, v)| (k, Arc::new(v))).collect(),
            last_access: Instant::now(),
            size_bytes,
        }
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
    }
}

// =============================================================================
// Prefetch Request
// =============================================================================

/// Request to prefetch a block.
#[derive(Debug)]
struct PrefetchRequest {
    block_id: usize,
}

// =============================================================================
// StreamingRecordCache
// =============================================================================

/// A streaming cache for neuron discovery records loaded from Parquet files.
///
/// Unlike the standard `RecordCache` which loads all data upfront, this cache
/// loads data in blocks on-demand, with LRU eviction and optional prefetching.
///
/// ## Usage
///
/// ```ignore
/// let cache = StreamingRecordCache::new("data.parquet", Some(100), Some(2))?;
/// let records = cache.get("neuron-42")?;
/// ```
pub struct StreamingRecordCache {
    /// Inner state wrapped in Arc for sharing with prefetch thread.
    inner: Arc<StreamingCacheInner>,
}

/// Inner state of the streaming cache, shareable with prefetch thread.
struct StreamingCacheInner {
    /// Path to the Parquet file.
    parquet_file: String,
    /// Maximum number of blocks to cache (None = unlimited).
    max_cached_blocks: Option<usize>,
    /// How many blocks ahead to prefetch.
    prefetch_depth: usize,
    /// Block size (records per block).
    block_size: usize,
    /// Mapping from neuron UUID to block IDs (a neuron may span multiple blocks).
    neuron_to_block: RwLock<HashMap<String, Vec<usize>>>,
    /// Cached blocks, keyed by block ID.
    blocks: RwLock<HashMap<usize, CacheBlock>>,
    /// Statistics counters.
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    eviction_count: AtomicU64,
    prefetch_count: AtomicU64,
    /// Prefetch thread sender.
    prefetch_tx: RwLock<Option<crossbeam_channel::Sender<PrefetchRequest>>>,
    /// Total number of blocks in the file.
    total_blocks: AtomicUsize,
}

/// Index of blocks in a Parquet file.
#[derive(Debug)]
struct BlockIndex {
    /// Maps neuron UUID to the block(s) containing its records.
    neuron_blocks: HashMap<String, Vec<usize>>,
    /// Total number of blocks.
    num_blocks: usize,
}

impl StreamingRecordCache {
    /// Create a new streaming record cache.
    ///
    /// # Arguments
    ///
    /// * `parquet_file` - Path to the Parquet file
    /// * `max_cached_blocks` - Maximum blocks to keep in memory (None = adaptive)
    /// * `prefetch_depth` - How many blocks ahead to prefetch (None = 2)
    pub fn new(
        parquet_file: &str,
        max_cached_blocks: Option<usize>,
        prefetch_depth: Option<usize>,
    ) -> Result<Self> {
        let prefetch_depth = prefetch_depth.unwrap_or(2);
        let block_size = get_block_size();

        // Build the block index by scanning the Parquet file metadata
        let index = Self::build_block_index(parquet_file, block_size)?;
        let num_blocks = index.num_blocks;

        // Determine max blocks based on memory tier if not specified
        let max_cached_blocks = max_cached_blocks.or_else(|| {
            // Adaptive: use 50% of estimated safe block count
            let tier = crate::analysis::utils::detect_memory_tier();
            match tier {
                crate::analysis::utils::MemoryTier::Low => Some(10),
                crate::analysis::utils::MemoryTier::Standard => Some(50),
                crate::analysis::utils::MemoryTier::High => Some(100),
            }
        });

        let inner = Arc::new(StreamingCacheInner {
            parquet_file: parquet_file.to_string(),
            max_cached_blocks,
            prefetch_depth,
            block_size,
            neuron_to_block: RwLock::new(index.neuron_blocks),
            blocks: RwLock::new(HashMap::new()),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            eviction_count: AtomicU64::new(0),
            prefetch_count: AtomicU64::new(0),
            prefetch_tx: RwLock::new(None),
            total_blocks: AtomicUsize::new(num_blocks),
        });

        // Set up prefetch thread if enabled
        if prefetch_depth > 0 {
            let (tx, rx) = crossbeam_channel::bounded::<PrefetchRequest>(32);
            *inner.prefetch_tx.write() = Some(tx);

            let inner_clone = Arc::clone(&inner);
            std::thread::Builder::new()
                .name("streaming-prefetch".to_string())
                .spawn(move || {
                    Self::prefetch_thread_loop(inner_clone, rx);
                })
                .ok();
        }

        Ok(Self { inner })
    }

    /// Background thread loop for prefetching blocks.
    fn prefetch_thread_loop(
        inner: Arc<StreamingCacheInner>,
        rx: crossbeam_channel::Receiver<PrefetchRequest>,
    ) {
        for req in rx {
            // Check if block is already cached
            {
                let blocks = inner.blocks.read();
                if blocks.contains_key(&req.block_id) {
                    continue;
                }
            }

            // Load the block
            if let Ok(block_records) =
                Self::load_block_records(&inner.parquet_file, req.block_id, inner.block_size)
            {
                // Evict and store atomically under the same write lock
                let mut blocks = inner.blocks.write();
                Self::evict_blocks(&mut blocks, inner.max_cached_blocks, &inner.eviction_count);
                blocks
                    .entry(req.block_id)
                    .or_insert_with(|| CacheBlock::new(block_records));
            }
        }
    }

    /// Build an index mapping neurons to blocks.
    fn build_block_index(parquet_file: &str, block_size: usize) -> Result<BlockIndex> {
        use arrow::array::StringArray;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let file = File::open(parquet_file)
            .with_context(|| format!("Failed to open Parquet file: {parquet_file}"))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader builder")?;

        let reader = builder.build().context("Failed to build Parquet reader")?;

        let mut neuron_blocks: HashMap<String, Vec<usize>> = HashMap::new();
        let mut current_block = 0;
        let mut rows_in_current_block = 0;

        for batch_result in reader {
            let batch = batch_result.context("Failed to read record batch")?;

            // Get neuron_uuid column
            let neuron_uuid_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to cast neuron_uuid column")?;

            // Track which neurons are in this batch
            for i in 0..batch.num_rows() {
                let uuid = neuron_uuid_col.value(i);
                neuron_blocks
                    .entry(uuid.to_string())
                    .or_default()
                    .push(current_block);

                rows_in_current_block += 1;

                // Check block boundary
                if rows_in_current_block >= block_size {
                    current_block += 1;
                    rows_in_current_block = 0;
                }
            }
        }

        // Deduplicate block lists
        for blocks in neuron_blocks.values_mut() {
            blocks.sort();
            blocks.dedup();
        }

        let num_blocks = if rows_in_current_block > 0 {
            current_block + 1
        } else {
            current_block.max(1)
        };

        Ok(BlockIndex {
            neuron_blocks,
            num_blocks,
        })
    }

    /// Load records for a specific block.
    fn load_block_records(
        parquet_file: &str,
        block_id: usize,
        block_size: usize,
    ) -> Result<HashMap<String, Vec<DiscoverRecord>>> {
        use arrow::array::{Array, Float32Array, ListArray, StringArray, UInt32Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let file = File::open(parquet_file)
            .with_context(|| format!("Failed to open Parquet file: {parquet_file}"))?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(file)
            .context("Failed to create Parquet reader builder")?;

        let reader = builder.build().context("Failed to build Parquet reader")?;

        let mut records_by_neuron: HashMap<String, Vec<DiscoverRecord>> = HashMap::new();
        let mut current_block = 0;
        let mut rows_in_block = 0;
        let mut found_target_block = false;

        for batch_result in reader {
            let batch = batch_result.context("Failed to read record batch")?;

            let obs_index_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .context("Failed to cast obs_index column")?;
            let neuron_uuid_col = batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .context("Failed to cast neuron_uuid column")?;
            let value_col = batch
                .column(2)
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to cast value column")?;
            let activation_col = batch
                .column(3)
                .as_any()
                .downcast_ref::<Float32Array>()
                .context("Failed to cast activation column")?;
            let errors_col = batch
                .column(4)
                .as_any()
                .downcast_ref::<ListArray>()
                .context("Failed to cast errors column")?;

            for i in 0..batch.num_rows() {
                // Only process records from the target block
                if current_block == block_id {
                    found_target_block = true;

                    let uuid = neuron_uuid_col.value(i).to_string();
                    let obs_index = obs_index_col.value(i);
                    let value = if value_col.is_null(i) {
                        None
                    } else {
                        Some(value_col.value(i))
                    };
                    let activation = activation_col.value(i);

                    let errors_list = errors_col.value(i);
                    let errors_array = errors_list
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .context("Failed to cast errors array")?;
                    let errors: Vec<f32> = (0..errors_array.len())
                        .map(|j| errors_array.value(j))
                        .collect();

                    let record =
                        DiscoverRecord::new(obs_index, uuid.clone(), value, activation, errors);
                    records_by_neuron.entry(uuid).or_default().push(record);
                }

                rows_in_block += 1;

                // Check block boundary
                if rows_in_block >= block_size {
                    if found_target_block {
                        // We've processed the target block, can stop
                        return Ok(records_by_neuron);
                    }
                    current_block += 1;
                    rows_in_block = 0;
                }
            }
        }

        Ok(records_by_neuron)
    }

    /// Get records for a specific neuron UUID.
    ///
    /// If a neuron's records span multiple blocks, all blocks are loaded and
    /// the records are combined.
    pub fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        // Find which blocks contain this neuron
        let block_ids: Vec<usize> = {
            let neuron_map = self.inner.neuron_to_block.read();
            neuron_map
                .get(neuron_uuid)
                .cloned()
                .unwrap_or_else(|| vec![0])
        };

        // Check if all blocks are cached
        let mut all_records: Vec<DiscoverRecord> = Vec::new();
        let mut all_cached = true;

        {
            let mut blocks = self.inner.blocks.write();
            for &block_id in &block_ids {
                if let Some(block) = blocks.get_mut(&block_id) {
                    if let Some(records) = block.records.get(neuron_uuid) {
                        all_records.extend(records.iter().cloned());
                        block.touch();
                    }
                } else {
                    all_cached = false;
                    break;
                }
            }
        }

        if all_cached && !all_records.is_empty() {
            self.inner.cache_hits.fetch_add(1, Ordering::Relaxed);
            return Ok(Arc::new(all_records));
        }

        // Cache miss - need to load blocks
        self.inner.cache_misses.fetch_add(1, Ordering::Relaxed);
        all_records.clear();

        // Load each block that contains this neuron
        for &block_id in &block_ids {
            // Check if this block is already cached
            {
                let blocks = self.inner.blocks.read();
                if let Some(block) = blocks.get(&block_id)
                    && let Some(records) = block.records.get(neuron_uuid)
                {
                    all_records.extend(records.iter().cloned());
                    continue;
                }
            }

            // Load the block
            let block_records = Self::load_block_records(
                &self.inner.parquet_file,
                block_id,
                self.inner.block_size,
            )?;

            // Get the records for this neuron from the loaded block
            if let Some(records) = block_records.get(neuron_uuid) {
                all_records.extend(records.iter().cloned());
            }

            // Evict and cache atomically under the same write lock
            {
                let mut blocks = self.inner.blocks.write();
                Self::evict_blocks(
                    &mut blocks,
                    self.inner.max_cached_blocks,
                    &self.inner.eviction_count,
                );
                blocks
                    .entry(block_id)
                    .or_insert_with(|| CacheBlock::new(block_records));
            }

            // Trigger prefetch for adjacent blocks
            self.trigger_prefetch(block_id);
        }

        Ok(Arc::new(all_records))
    }

    /// Evict least-recently-used blocks from an already-locked block map.
    fn evict_blocks(
        blocks: &mut HashMap<usize, CacheBlock>,
        max_blocks: Option<usize>,
        eviction_count: &AtomicU64,
    ) {
        if let Some(max_blocks) = max_blocks {
            while blocks.len() >= max_blocks {
                // Find the LRU block
                let lru_block_id = blocks
                    .iter()
                    .min_by_key(|(_, block)| block.last_access)
                    .map(|(id, _)| *id);

                if let Some(block_id) = lru_block_id {
                    blocks.remove(&block_id);
                    eviction_count.fetch_add(1, Ordering::Relaxed);
                } else {
                    break;
                }
            }
        }
    }

    /// Trigger prefetch for adjacent blocks.
    fn trigger_prefetch(&self, current_block: usize) {
        let tx_guard = self.inner.prefetch_tx.read();
        if let Some(ref tx) = *tx_guard {
            let total = self.inner.total_blocks.load(Ordering::Relaxed);

            for offset in 1..=self.inner.prefetch_depth {
                let next_block = current_block + offset;
                if next_block < total {
                    // Check if already cached and try to prefetch if not
                    let already_cached = self.inner.blocks.read().contains_key(&next_block);
                    if !already_cached
                        && tx
                            .try_send(PrefetchRequest {
                                block_id: next_block,
                            })
                            .is_ok()
                    {
                        self.inner.prefetch_count.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
    }

    /// Get current cache statistics.
    pub fn stats(&self) -> StreamingCacheStats {
        let blocks = self.inner.blocks.read();
        let cached_bytes: usize = blocks.values().map(|b| b.size_bytes).sum();

        StreamingCacheStats {
            cached_blocks: blocks.len(),
            cache_hits: self.inner.cache_hits.load(Ordering::Relaxed),
            cache_misses: self.inner.cache_misses.load(Ordering::Relaxed),
            eviction_count: self.inner.eviction_count.load(Ordering::Relaxed),
            prefetch_count: self.inner.prefetch_count.load(Ordering::Relaxed),
            cached_bytes,
        }
    }

    /// Check if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.blocks.read().is_empty()
    }

    /// Get the number of cached blocks.
    pub fn len(&self) -> usize {
        self.inner.blocks.read().len()
    }
}

impl Drop for StreamingRecordCache {
    fn drop(&mut self) {
        // Close the prefetch channel to signal thread shutdown
        *self.inner.prefetch_tx.write() = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_values() {
        let config = StreamingConfig::default();
        assert!(config.max_cached_blocks.is_none());
        assert_eq!(config.prefetch_depth, Some(2));
    }

    #[test]
    #[serial_test::serial]
    fn is_streaming_enabled_default() {
        // When env var is not set, streaming should be enabled
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_PRELOAD_ALL");
        }
        assert!(is_streaming_enabled());
    }

    #[test]
    fn streaming_cache_stats_initial() {
        let stats = StreamingCacheStats::default();
        assert_eq!(stats.cached_blocks, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    #[serial_test::serial]
    fn block_size_respects_minimum() {
        // Even with small values, block size should be at least MIN_BLOCK_SIZE
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::set_var("NEAT_AI_DISCOVERY_BLOCK_SIZE", "1");
        }
        assert!(get_block_size() >= MIN_BLOCK_SIZE);
        // SAFETY: Serialised via #[serial] — no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_BLOCK_SIZE");
        }
    }
}
