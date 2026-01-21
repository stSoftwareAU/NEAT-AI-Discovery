//! Record cache module for parquet data loading.
//!
//! This module provides the `RecordCache` struct which handles efficient loading
//! and caching of discovery records from parquet files. It supports both:
//! - **Pre-loaded mode**: Fast, loads entire file into memory upfront
//! - **Lazy-loaded mode**: Memory-efficient, loads records on-demand per neuron
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

use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::analysis::utils::{check_memory_for_parquet, verbose_enabled};

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
    pub(crate) fn new_adaptive(parquet_file: &str) -> Result<Self> {
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

    /// Create a cache with a custom loader function.
    /// Used primarily for testing. Available in both unit tests and integration tests.
    pub fn with_loader(parquet_file: &str, loader: Arc<RecordCacheLoader>) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: RwLock::new(HashMap::new()),
            loader,
        }
    }
}
