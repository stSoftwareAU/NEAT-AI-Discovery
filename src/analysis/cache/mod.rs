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
//! ## Sub-modules
//!
//! - `loading_strategy` — Strategy selection based on file size and memory
//! - `lru_cache` — Per-neuron LRU cache with eviction
//! - `compressed_cache` — LZ4-compressed LRU cache (Issue #420)
//! - `tiered_cache` — Automatic strategy wrapper (Issue #215)
//! - `serialisation` — Binary serialisation and LZ4 compression (Issue #420, #484)
//!
//! ## Issue #186: `RwLock` for Read-Heavy Workloads
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
//! - **`PreloadAll`**: For small files (estimated expanded < `available_memory` / 4)
//! - **`LruCache`**: For medium files (keeps frequently-accessed neurons in memory)
//! - **Streaming**: For very large files (block-based loading with LRU eviction)

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
mod compressed_cache;
mod loading_strategy;
mod lru_cache;
pub(crate) mod serialisation;
mod tiered_cache;

// Re-export all public types for backward compatibility
pub use compressed_cache::CompressedLruRecordCache;
pub use loading_strategy::{LoadingStrategy, select_loading_strategy};
pub use lru_cache::{LruCacheStats, LruRecordCache};
pub use tiered_cache::{TieredCacheStats, TieredRecordCache};

// Re-export streaming module types (Issue #193)
pub use super::streaming::{
    StreamingCacheStats, StreamingConfig, StreamingRecordCache, get_streaming_config_from_env,
    is_streaming_enabled,
};

use crate::CreatureJson;
use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use crate::analysis::utils::{check_memory_for_parquet, get_memory_info, verbose_enabled};

type RecordCacheLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;
type CachedNeuronRecords = OnceLock<Result<Arc<Vec<DiscoverRecord>>, String>>;

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
    #[tracing::instrument(skip_all, fields(parquet_file))]
    pub fn new_adaptive(parquet_file: &str) -> Result<Self> {
        Self::new_adaptive_with_deadline(parquet_file, None)
    }

    /// Create an adaptive cache with optional deadline checking (Issue #648).
    ///
    /// When a deadline is provided, the parquet loading phase checks the deadline
    /// periodically at batch boundaries and aborts early with a clear error if
    /// the deadline is reached. Watchdog beats are emitted during loading to
    /// prevent the watchdog from triggering for legitimately slow (but progressing)
    /// loads.
    ///
    /// If no deadline is provided, behaves identically to `new_adaptive()`.
    #[tracing::instrument(skip_all, fields(parquet_file))]
    pub fn new_adaptive_with_deadline(
        parquet_file: &str,
        deadline: Option<std::time::SystemTime>,
    ) -> Result<Self> {
        // Check if we have enough memory for pre-loading
        match check_memory_for_parquet(parquet_file) {
            Ok(()) => {
                // Sufficient memory - use fast pre-loaded mode
                Self::new_preloaded_with_deadline(parquet_file, deadline)
            }
            Err(memory_error) => {
                // Insufficient memory - fall back to lazy loading
                tracing::warn!(
                    "insufficient memory for pre-loading — falling back to lazy-loading mode"
                );
                if verbose_enabled() {
                    tracing::debug!(%memory_error, "memory check failed");
                }
                Self::new_lazy(parquet_file)
            }
        }
    }

    /// Create a lazy-loading cache that loads records on-demand.
    /// Slower than pre-loaded mode but uses minimal memory.
    fn new_lazy(parquet_file: &str) -> Result<Self> {
        use crate::parquet_format::read_records_from_parquet;

        tracing::info!("using lazy-loading mode for parquet file");

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
        Self::new_preloaded_with_deadline(parquet_file, None)
    }

    /// Internal pre-loaded implementation with optional deadline (Issue #648).
    fn new_preloaded_with_deadline(
        parquet_file: &str,
        deadline: Option<std::time::SystemTime>,
    ) -> Result<Self> {
        use crate::parquet_format::read_all_records_grouped_by_neuron_with_deadline;
        use std::time::Instant;

        let start = Instant::now();
        let grouped = read_all_records_grouped_by_neuron_with_deadline(parquet_file, deadline)?;
        let elapsed = start.elapsed();

        // Pre-populate the cache with OnceLock-wrapped records
        let cache: RwLock<HashMap<String, Arc<CachedNeuronRecords>>> = RwLock::new(
            grouped
                .into_iter()
                .map(|(k, v)| {
                    let cell = OnceLock::new();
                    // OnceLock::set can't fail here - cell was just created
                    cell.set(Ok(Arc::new(v))).ok();
                    (k, Arc::new(cell))
                })
                .collect(),
        );

        if verbose_enabled() {
            let count = cache.read().len();
            tracing::debug!(count, ?elapsed, "pre-loaded neurons from parquet");
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
    /// After obtaining the cell, uses `OnceLock::get_or_try_init()` for thread-safe
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
                    .or_insert_with(|| Arc::new(OnceLock::new()))
                    .clone()
            }
        };

        // Lazily initialise the cell outside the cache lock so other threads can
        // access other neurons while this one is being loaded. get_or_init
        // guarantees only one thread executes the loader for a given neuron.
        let parquet = self.parquet_file.clone();
        let loader = Arc::clone(&self.loader);
        let result = cell.get_or_init(|| {
            loader(&parquet, neuron_uuid)
                .map(Arc::new)
                .map_err(|e| format!("{e:#}"))
        });

        match result {
            Ok(records) => Ok(Arc::clone(records)),
            Err(msg) => Err(anyhow::anyhow!(
                "Failed to load records for neuron '{neuron_uuid}': {msg}"
            )),
        }
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
                    .map(|r| (uuid.clone(), r.as_ref().clone()))
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
                    .map(|r| (uuid.clone(), r.as_ref().clone()))
            })
            .collect()
    }

    /// Load records for every neuron in the creature (Issue #493).
    ///
    /// Extracts all neuron UUIDs from the creature and loads their records.
    /// Issue #1036: Inlined to avoid double-cloning (was: clone into `Vec<String>`,
    /// then clone again in `load_records_for_uuids`).
    pub fn load_records_for_all_neurons(
        &self,
        creature: &CreatureJson,
    ) -> Vec<(String, Vec<DiscoverRecord>)> {
        creature
            .neurons
            .iter()
            .filter_map(|n| {
                self.get(&n.uuid)
                    .ok()
                    .map(|r| (n.uuid.clone(), r.as_ref().clone()))
            })
            .collect()
    }

    /// Load records for neurons matching any of the given type names (Issue #493).
    ///
    /// Filters the creature's neurons by `neuron_type` then loads their records.
    /// Common usage: `&["output"]`, `&["input"]`, `&["input", "output"]`,
    /// `&["input", "hidden"]`.
    /// Issue #1036: Inlined to avoid double-cloning (was: clone into `Vec<String>`,
    /// then clone again in `load_records_for_uuids`).
    pub fn load_records_for_neuron_types(
        &self,
        creature: &CreatureJson,
        types: &[&str],
    ) -> Vec<(String, Vec<DiscoverRecord>)> {
        creature
            .neurons
            .iter()
            .filter(|n| types.contains(&n.neuron_type.as_str()))
            .filter_map(|n| {
                self.get(&n.uuid)
                    .ok()
                    .map(|r| (n.uuid.clone(), r.as_ref().clone()))
            })
            .collect()
    }

    /// Load records for the unique set of synapse source neuron UUIDs (Issue #493).
    ///
    /// Collects the deduplicated `from_uuid` values from all creature synapses,
    /// then loads their records.
    /// Issue #1036: Deduplicate via `HashSet<&str>` to avoid cloning UUIDs into a
    /// temporary `HashSet<String>`, then clone only once for the output tuple.
    pub fn load_records_for_synapse_sources(
        &self,
        creature: &CreatureJson,
    ) -> Vec<(String, Vec<DiscoverRecord>)> {
        let mut seen = std::collections::HashSet::new();
        creature
            .synapses
            .iter()
            .filter(|s| seen.insert(s.from_uuid.as_str()))
            .filter_map(|s| {
                self.get(&s.from_uuid)
                    .ok()
                    .map(|r| (s.from_uuid.clone(), r.as_ref().clone()))
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
    /// - **`PreloadAll`**: For small files where estimated expanded size < `available_memory` / 4
    /// - **`LruCache`**: For medium files, keeps frequently-accessed neurons in memory
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
            tracing::debug!(
                file_size_mb,
                available_gb,
                ?strategy,
                "tiered loading strategy selected"
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
                tracing::info!("using streaming mode for very large parquet file");
                Self::new_lazy(parquet_file)
            }
        }
    }
}
