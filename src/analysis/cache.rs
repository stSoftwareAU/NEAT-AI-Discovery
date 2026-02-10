//! Record cache module for parquet data loading.
//!
//! This module provides the `RecordCache` struct which handles efficient loading
//! and caching of discovery records from parquet files. It supports multiple modes:
//!
//! - **Pre-loaded mode**: Fast, loads entire file into memory upfront
//! - **Lazy-loaded mode**: Memory-efficient, loads records on-demand per neuron
//! - **Streaming mode** (Issue #193): Block-based loading with LRU eviction and prefetch
//! - **Tiered mode** (Issue #215): Automatic strategy selection based on file size and memory
//!
//! The cache automatically chooses the best strategy based on available system memory.
//!
//! ## Issue #186: RwLock for Read-Heavy Workloads
//!
//! This module uses `parking_lot::RwLock` instead of `std::sync::Mutex` to allow
//! concurrent reads during analysis. During the analysis phase, the cache is
//! predominantly read (getting neuron records) with writes only happening on cache
//! misses. Using `RwLock` allows multiple focus neurons to be analysed in parallel
//! without serialising on lock acquisition.
//!
//! ## Issue #193: Streaming Mode with Prefetch
//!
//! For large datasets, streaming mode provides:
//! - Block-based loading from Parquet row groups
//! - LRU eviction for bounded memory usage
//! - Prefetch mechanism for improved performance
//! - Configuration via environment variables
//!
//! ## Issue #215: Tiered Loading Strategy
//!
//! The tiered loading strategy automatically selects between:
//! - **PreloadAll**: For small files (estimated expanded < available_memory / 4)
//! - **LruCache**: For medium files (keeps frequently-accessed neurons in memory)
//! - **Streaming**: For very large files (block-based loading with LRU eviction)

use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

use crate::analysis::utils::{check_memory_for_parquet, get_memory_info, verbose_enabled};

// Re-export streaming module types (Issue #193)
pub use super::streaming::{
    StreamingCacheStats, StreamingConfig, StreamingRecordCache, get_streaming_config_from_env,
    is_streaming_enabled,
};

type RecordCacheLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;
type CachedNeuronRecords = OnceCell<Arc<Vec<DiscoverRecord>>>;

/// A thread-safe cache for neuron discovery records loaded from parquet files.
///
/// Uses `RwLock` to allow concurrent reads during the analysis phase, which is
/// read-heavy after the initial cache population. This significantly improves
/// throughput when multiple focus neurons are analysed in parallel using Rayon.
pub struct RecordCache {
    parquet_file: String,
    /// The cache uses `RwLock` instead of `Mutex` to allow concurrent reads.
    /// During analysis, most accesses are reads (cache hits), with writes only
    /// occurring on cache misses. This allows Rayon parallel iteration over
    /// focus neurons without serialising on lock acquisition.
    pub(crate) cache: RwLock<HashMap<String, Arc<CachedNeuronRecords>>>,
    loader: Arc<RecordCacheLoader>,
}

impl RecordCache {
    /// Create a cache that automatically chooses the best loading strategy based on
    /// available system memory:
    ///
    /// - **Pre-loaded mode** (fast): Loads entire parquet file into memory upfront.
    ///   Used when there's sufficient RAM (3× file size + 1GB headroom).
    ///
    /// - **Lazy-loaded mode** (memory-efficient): Loads records on-demand per neuron.
    ///   Slower (O(N) parquet scans for N neurons) but works on memory-constrained systems.
    ///
    /// This ensures discovery works on any modern Mac/PC, adapting to available resources.
    pub fn new_adaptive(parquet_file: &str) -> Result<Self> {
        // Check if we have enough memory for pre-loading
        match check_memory_for_parquet(parquet_file) {
            Ok(()) => {
                // Sufficient memory - use fast pre-loaded mode
                Self::new_preloaded_internal(parquet_file)
            }
            Err(memory_error) => {
                // Insufficient memory - fall back to lazy loading
                eprintln!(
                    "[NEAT-AI-Discovery] Insufficient memory for pre-loading. \
                     Falling back to lazy-loading mode (slower but memory-efficient)."
                );
                if verbose_enabled() {
                    eprintln!("[NEAT-AI-Discovery][verbose] Memory check failed: {memory_error}");
                }
                Self::new_lazy(parquet_file)
            }
        }
    }

    /// Create a lazy-loading cache that loads records on-demand.
    /// Slower than pre-loaded mode but uses minimal memory.
    fn new_lazy(parquet_file: &str) -> Result<Self> {
        use crate::parquet_format::read_records_from_parquet;

        eprintln!(
            "[NEAT-AI-Discovery] Using lazy-loading mode for parquet file. \
             This is slower but uses less memory."
        );

        Ok(Self {
            parquet_file: parquet_file.to_string(),
            cache: RwLock::new(HashMap::new()),
            loader: Arc::new(move |file: &str, neuron_uuid: &str| {
                read_records_from_parquet(file, neuron_uuid)
            }),
        })
    }

    /// Internal pre-loaded implementation (called when memory check passes).
    fn new_preloaded_internal(parquet_file: &str) -> Result<Self> {
        use crate::parquet_format::read_all_records_grouped_by_neuron;
        use std::time::Instant;

        let start = Instant::now();
        let grouped = read_all_records_grouped_by_neuron(parquet_file)?;
        let elapsed = start.elapsed();

        // Pre-populate the cache with OnceCell-wrapped records
        let cache: RwLock<HashMap<String, Arc<CachedNeuronRecords>>> = RwLock::new(
            grouped
                .into_iter()
                .map(|(k, v)| {
                    let cell = OnceCell::new();
                    // OnceCell::set can't fail here - cell was just created
                    cell.set(Arc::new(v)).ok();
                    (k, Arc::new(cell))
                })
                .collect(),
        );

        if verbose_enabled() {
            let count = cache.read().len();
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Pre-loaded {count} neurons from parquet in {elapsed:?}"
            );
        }

        // In pre-loaded mode, if a neuron UUID wasn't in the parquet file,
        // return an empty vector. This matches the behaviour when the data simply
        // doesn't exist for that UUID.
        Ok(Self {
            parquet_file: parquet_file.to_string(),
            cache,
            loader: Arc::new(|_file: &str, _neuron_uuid: &str| {
                // This neuron UUID wasn't in the preloaded data - return empty
                Ok(Vec::new())
            }),
        })
    }

    /// Get records for a specific neuron UUID.
    /// Returns an Arc to avoid cloning the data.
    ///
    /// This method is optimised for read-heavy workloads:
    /// - First attempts a read lock to check if the cell already exists (fast path)
    /// - Only acquires a write lock if the cell needs to be created (slow path)
    ///
    /// After obtaining the cell, uses `OnceCell::get_or_try_init()` for thread-safe
    /// lazy initialisation without holding the cache lock.
    pub fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        // Fast path: try to get with read lock first (allows concurrent reads)
        let cell = {
            let cache = self.cache.read();
            cache.get(neuron_uuid).cloned()
        };

        let cell = match cell {
            Some(c) => c,
            None => {
                // Slow path: need to insert a new entry (requires write lock)
                let mut cache = self.cache.write();
                // Double-check: another thread may have inserted while we were waiting
                cache
                    .entry(neuron_uuid.to_string())
                    .or_insert_with(|| Arc::new(OnceCell::new()))
                    .clone()
            }
        };

        // Use get_or_try_init to lazily initialise the cell.
        // If the value has already been initialised (e.g., from pre-loading or
        // a previous call), this returns instantly.
        // Note: This happens OUTSIDE the cache lock, so other threads can access
        // other neurons while this one is being loaded.
        let records = cell.get_or_try_init(|| -> Result<Arc<Vec<DiscoverRecord>>> {
            let loader = Arc::clone(&self.loader);
            let result = loader(&self.parquet_file, neuron_uuid)
                .with_context(|| format!("Failed to load records for neuron '{neuron_uuid}'"))?;
            Ok(Arc::new(result))
        })?;

        Ok(Arc::clone(records))
    }

    /// Get the number of entries in the cache.
    ///
    /// Uses a read lock since this is a read-only operation.
    pub fn len(&self) -> usize {
        self.cache.read().len()
    }

    /// Check if the cache is empty.
    ///
    /// Uses a read lock since this is a read-only operation.
    pub fn is_empty(&self) -> bool {
        self.cache.read().is_empty()
    }

    /// Load records for a slice of neuron UUIDs in one call (Issue #493).
    ///
    /// Returns `(uuid, records)` pairs for each UUID.
    /// UUIDs where the cache lookup fails are silently skipped.
    ///
    /// This eliminates the repeated `filter_map(|uuid| cache.get(uuid)...)` boilerplate
    /// that previously appeared 25 times in `analyze_all()`.
    pub fn load_records_for_uuids(&self, uuids: &[String]) -> Vec<(String, Vec<DiscoverRecord>)> {
        uuids
            .iter()
            .filter_map(|uuid| {
                self.get(uuid)
                    .ok()
                    .map(|r| (uuid.clone(), r.as_ref().to_vec()))
            })
            .collect()
    }

    /// Load records for hidden neurons from the standard `(uuid, squash, bias)` tuple
    /// format used throughout the discovery dispatch (Issue #493).
    pub fn load_records_for_hidden(
        &self,
        hidden_neurons: &[(String, String, f32)],
    ) -> Vec<(String, Vec<DiscoverRecord>)> {
        hidden_neurons
            .iter()
            .filter_map(|(uuid, _, _)| {
                self.get(uuid)
                    .ok()
                    .map(|r| (uuid.clone(), r.as_ref().to_vec()))
            })
            .collect()
    }

    /// Create a cache with a custom loader function.
    /// Used primarily for testing. Available in both unit tests and integration tests.
    pub fn with_loader(parquet_file: &str, loader: Arc<RecordCacheLoader>) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: RwLock::new(HashMap::new()),
            loader,
        }
    }

    /// Create a cache using the tiered loading strategy (Issue #215).
    ///
    /// This method automatically selects the best loading strategy based on
    /// file size and available system memory:
    ///
    /// - **PreloadAll**: For small files where estimated expanded size < available_memory / 4
    /// - **LruCache**: For medium files, keeps frequently-accessed neurons in memory
    /// - **Streaming**: For very large files that exceed available memory
    ///
    /// This provides optimal performance across different file sizes while
    /// respecting memory constraints.
    pub fn new_tiered(parquet_file: &str) -> Result<Self> {
        let file_size = std::fs::metadata(parquet_file)
            .with_context(|| format!("Failed to get file size for {parquet_file}"))?
            .len();
        let (available, _total) = get_memory_info();

        let strategy = select_loading_strategy(file_size, available);

        if verbose_enabled() {
            let file_size_mb = file_size as f64 / (1024.0 * 1024.0);
            let available_gb = available as f64 / (1024.0 * 1024.0 * 1024.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Tiered loading: file={file_size_mb:.1}MB, \
                 available={available_gb:.1}GB, strategy={strategy:?}"
            );
        }

        match strategy {
            LoadingStrategy::PreloadAll => Self::new_preloaded_internal(parquet_file),
            LoadingStrategy::LruCache { capacity_bytes } => {
                // Wrap LruRecordCache in a RecordCache-compatible interface
                let lru_cache = Arc::new(
                    LruRecordCache::new(parquet_file, capacity_bytes)
                        .with_context(|| "Failed to create LRU cache")?,
                );
                let lru_clone = Arc::clone(&lru_cache);
                Ok(Self {
                    parquet_file: parquet_file.to_string(),
                    cache: RwLock::new(HashMap::new()),
                    loader: Arc::new(move |_file: &str, neuron_uuid: &str| {
                        // The LRU cache handles loading internally
                        let records = lru_clone.get(neuron_uuid)?;
                        Ok((*records).clone())
                    }),
                })
            }
            LoadingStrategy::Streaming => {
                // Use the existing streaming cache
                eprintln!("[NEAT-AI-Discovery] Using streaming mode for very large parquet file.");
                Self::new_lazy(parquet_file)
            }
        }
    }
}

// =============================================================================
// Tiered Loading Strategy (Issue #215)
// =============================================================================

/// Loading strategy for parquet files based on file size and available memory.
///
/// Issue #215: Automatic selection between different caching modes to support
/// larger files while maintaining good performance for smaller ones.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadingStrategy {
    /// Load everything into memory upfront.
    /// Best for small files where estimated_expanded < available_memory / 4.
    PreloadAll,

    /// Keep frequently-accessed neurons in memory with LRU eviction.
    /// Best for medium-sized files that fit in memory but benefit from bounded usage.
    LruCache {
        /// Maximum bytes to keep in cache.
        capacity_bytes: usize,
    },

    /// Load records on-demand without caching.
    /// Best for very large files that exceed available memory.
    Streaming,
}

/// Decompression ratio estimate (parquet typically expands 2-4x).
const DECOMPRESSION_RATIO: u64 = 3;

/// Select the optimal loading strategy based on file size and available memory.
///
/// # Strategy Selection Logic
///
/// - **PreloadAll**: When estimated expanded size < available_memory / 4
///   (small files that easily fit in memory with plenty of headroom)
///
/// - **LruCache**: When estimated expanded size < available_memory
///   (medium files that fit but benefit from bounded memory usage)
///
/// - **Streaming**: When estimated expanded size >= available_memory
///   (large files that would exceed available memory)
///
/// # Arguments
///
/// * `file_size_bytes` - Size of the parquet file in bytes
/// * `available_memory_bytes` - Available system memory in bytes
pub fn select_loading_strategy(
    file_size_bytes: u64,
    available_memory_bytes: u64,
) -> LoadingStrategy {
    let estimated_expanded = file_size_bytes.saturating_mul(DECOMPRESSION_RATIO);

    // Small file threshold: fits comfortably in 1/4 of available memory
    let small_threshold = available_memory_bytes / 4;

    if estimated_expanded < small_threshold {
        LoadingStrategy::PreloadAll
    } else if estimated_expanded < available_memory_bytes {
        // Medium file: use LRU cache with half of available memory
        let capacity_bytes = (available_memory_bytes / 2) as usize;
        LoadingStrategy::LruCache { capacity_bytes }
    } else {
        // Large file: use streaming
        LoadingStrategy::Streaming
    }
}

// =============================================================================
// LRU Record Cache (Issue #215)
// =============================================================================

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
fn estimate_records_size(records: &[DiscoverRecord]) -> usize {
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
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Creating LRU cache with capacity {capacity_mb:.1}MB"
            );
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

// =============================================================================
// Compressed LRU Record Cache (Issue #420)
// =============================================================================

/// A compressed entry in the LRU cache.
///
/// Records are serialised and LZ4-compressed when stored, then decompressed
/// on access. This trades CPU time for memory, allowing more neurons to fit
/// in the cache before eviction is needed.
struct CompressedCacheEntry {
    /// LZ4-compressed serialised record data.
    compressed_data: Vec<u8>,
    /// Number of records (for quick stats without decompression).
    #[allow(dead_code)]
    record_count: usize,
    /// Size of the compressed data in bytes (what we actually store).
    compressed_size: usize,
    /// Last access time for LRU ordering.
    last_access: Instant,
}

impl CompressedCacheEntry {
    fn new(records: &[DiscoverRecord]) -> Self {
        let serialised = serialise_records(records);
        let compressed = lz4_flex::compress_prepend_size(&serialised);
        let compressed_size = compressed.len();
        Self {
            compressed_data: compressed,
            record_count: records.len(),
            compressed_size,
            last_access: Instant::now(),
        }
    }

    fn decompress(&self) -> Vec<DiscoverRecord> {
        let decompressed = lz4_flex::decompress_size_prepended(&self.compressed_data)
            .expect("LZ4 decompression failed — data corruption");
        deserialise_records(&decompressed)
    }

    fn touch(&mut self) {
        self.last_access = Instant::now();
    }
}

/// Serialise records to a compact binary format for LZ4 compression.
///
/// Format per record:
/// - obs_index: u32 (4 bytes)
/// - uuid_len: u16 (2 bytes)
/// - uuid: [u8; uuid_len]
/// - has_value: u8 (1 byte)
/// - value: f32 (4 bytes, only if has_value)
/// - activation: f32 (4 bytes)
/// - errors_len: u16 (2 bytes)
/// - errors: [f32; errors_len]
fn serialise_records(records: &[DiscoverRecord]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(records.len() * 32);
    for r in records {
        buf.extend_from_slice(&r.obs_index.to_le_bytes());
        let uuid_bytes = r.neuron_uuid.as_bytes();
        buf.extend_from_slice(&(uuid_bytes.len() as u16).to_le_bytes());
        buf.extend_from_slice(uuid_bytes);
        match r.value {
            Some(v) => {
                buf.push(1);
                buf.extend_from_slice(&v.to_le_bytes());
            }
            None => {
                buf.push(0);
            }
        }
        buf.extend_from_slice(&r.activation.to_le_bytes());
        buf.extend_from_slice(&(r.errors.len() as u16).to_le_bytes());
        for &e in &r.errors {
            buf.extend_from_slice(&e.to_le_bytes());
        }
    }
    buf
}

/// Deserialise records from the compact binary format.
fn deserialise_records(data: &[u8]) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    let mut pos = 0;
    while pos + 6 <= data.len() {
        let obs_index = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let uuid_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + uuid_len > data.len() {
            break;
        }
        let neuron_uuid = String::from_utf8_lossy(&data[pos..pos + uuid_len]).to_string();
        pos += uuid_len;
        if pos >= data.len() {
            break;
        }
        let has_value = data[pos];
        pos += 1;
        let value = if has_value == 1 {
            if pos + 4 > data.len() {
                break;
            }
            let v = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            Some(v)
        } else {
            None
        };
        if pos + 4 > data.len() {
            break;
        }
        let activation = f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        if pos + 2 > data.len() {
            break;
        }
        let errors_len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + errors_len * 4 > data.len() {
            break;
        }
        let mut errors = Vec::with_capacity(errors_len);
        for _ in 0..errors_len {
            errors.push(f32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()));
            pos += 4;
        }
        records.push(DiscoverRecord::new(
            obs_index,
            neuron_uuid,
            value,
            activation,
            errors,
        ));
    }
    records
}

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
}

impl CompressedLruRecordCache {
    /// Create a new compressed LRU record cache.
    ///
    /// # Arguments
    ///
    /// * `parquet_file` - Path to the parquet file
    /// * `capacity_bytes` - Maximum compressed memory to use for caching
    pub fn new(parquet_file: &str, capacity_bytes: usize) -> Result<Self> {
        std::fs::metadata(parquet_file)
            .with_context(|| format!("Parquet file not found: {parquet_file}"))?;

        if verbose_enabled() {
            let capacity_mb = capacity_bytes as f64 / (1024.0 * 1024.0);
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Creating compressed LRU cache with capacity {capacity_mb:.1}MB"
            );
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
                return Ok(Arc::new(entry.decompress()));
            }
        }

        // Cache miss — load from parquet
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
        let records = self.load_neuron_records(neuron_uuid)?;
        let entry = CompressedCacheEntry::new(&records);
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

// =============================================================================
// Tiered Record Cache (Issue #215)
// =============================================================================

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
            eprintln!(
                "[NEAT-AI-Discovery][verbose] TieredRecordCache selected strategy: {strategy:?}"
            );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_strategy_preload_for_small_files() {
        // 10MB file, 8GB available
        let file_size = 10 * 1024 * 1024;
        let available = 8 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        assert!(matches!(strategy, LoadingStrategy::PreloadAll));
    }

    #[test]
    fn loading_strategy_lru_for_medium_files() {
        // 1GB file, 8GB available
        let file_size = 1024 * 1024 * 1024_u64;
        let available = 8 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        assert!(matches!(strategy, LoadingStrategy::LruCache { .. }));
    }

    #[test]
    fn loading_strategy_streaming_for_large_files() {
        // 4GB file, 8GB available
        let file_size = 4 * 1024 * 1024 * 1024_u64;
        let available = 8 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        assert!(matches!(strategy, LoadingStrategy::Streaming));
    }

    #[test]
    fn lru_capacity_scales_with_memory() {
        // 2GB file, 16GB available
        // Estimated expanded = 2GB * 3 = 6GB
        // Small threshold = 16GB / 4 = 4GB
        // 6GB > 4GB but < 16GB, so LruCache is selected
        let file_size = 2 * 1024 * 1024 * 1024_u64;
        let available = 16 * 1024 * 1024 * 1024_u64;

        let strategy = select_loading_strategy(file_size, available);
        if let LoadingStrategy::LruCache { capacity_bytes } = strategy {
            // Should be approximately 8GB (half of available)
            let expected = 8 * 1024 * 1024 * 1024_usize;
            let tolerance = expected / 10;
            assert!(
                (capacity_bytes as i64 - expected as i64).unsigned_abs() < tolerance as u64,
                "Capacity {capacity_bytes} should be ~{expected}"
            );
        } else {
            panic!("Expected LruCache strategy, got {strategy:?}");
        }
    }

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
