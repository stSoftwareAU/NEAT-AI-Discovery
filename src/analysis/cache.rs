//! Record cache module for parquet data loading.
//!
//! This module provides the `RecordCache` struct which handles efficient loading
//! and caching of discovery records from parquet files. It supports both:
//! - **Pre-loaded mode**: Fast, loads entire file into memory upfront
//! - **Lazy-loaded mode**: Memory-efficient, loads records on-demand per neuron
//!
//! The cache automatically chooses the best strategy based on available system memory.

use crate::types::DiscoverRecord;
use anyhow::{Context, Result};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::analysis::utils::{check_memory_for_parquet, verbose_enabled};

type RecordCacheLoader = dyn Fn(&str, &str) -> Result<Vec<DiscoverRecord>> + Send + Sync + 'static;
type CachedNeuronRecords = OnceCell<Arc<Vec<DiscoverRecord>>>;

pub(crate) struct RecordCache {
    parquet_file: String,
    pub(crate) cache: Mutex<HashMap<String, Arc<CachedNeuronRecords>>>,
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
            cache: Mutex::new(HashMap::new()),
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
        let cache: Mutex<HashMap<String, Arc<CachedNeuronRecords>>> = Mutex::new(
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
            let count = cache.lock().expect("Mutex poisoned").len();
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
    pub(crate) fn get(&self, neuron_uuid: &str) -> Result<Arc<Vec<DiscoverRecord>>> {
        let cell = {
            let mut cache = self.cache.lock().expect("Mutex poisoned");
            cache
                .entry(neuron_uuid.to_string())
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone()
        };

        // Use get_or_try_init to lazily initialise the cell.
        // If the value has already been initialised (e.g., from pre-loading or
        // a previous call), this returns instantly.
        let records = cell.get_or_try_init(|| -> Result<Arc<Vec<DiscoverRecord>>> {
            let loader = Arc::clone(&self.loader);
            let result = loader(&self.parquet_file, neuron_uuid)
                .with_context(|| format!("Failed to load records for neuron '{neuron_uuid}'"))?;
            Ok(Arc::new(result))
        })?;

        Ok(Arc::clone(records))
    }

    /// Create a cache with a custom loader function.
    /// Used primarily for testing.
    #[cfg(test)]
    pub(crate) fn with_loader(parquet_file: &str, loader: Arc<RecordCacheLoader>) -> Self {
        Self {
            parquet_file: parquet_file.to_string(),
            cache: Mutex::new(HashMap::new()),
            loader,
        }
    }
}
