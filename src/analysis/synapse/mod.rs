//! Synapse analysis module
//!
//! This module contains the complete synapse analysis pipeline — identifying
//! beneficial new synapses, harmful existing synapses, and coordinated
//! structural changes that would reduce creature error.
//!
//! ## Key Functions
//!
//! - `analyze_synapses` — Public entry point for synapse analysis
//! - `analyze_synapses_with_cache` — Internal implementation with shared cache
//! - `analyze_synapses_with_cache_impl` — Core analysis engine (GPU-accelerated)
//! - `compute_synapse_improvement_and_count` — Core improvement calculation
//!
//! ## Sample Locality Optimisation (Issue #221)
//!
//! When analysing multiple source neurons for the same target, sources that share
//! the same `obs_indices` can benefit from batched sample building. This reduces
//! sample building overhead by up to 100x for creatures where input neurons share
//! the same observation indices.
//!
//! ## Module Structure (Issue #482, #802)
//!
//! - `scoring` — Improvement calculation, saturation-aware simulation, boosting
//! - `candidate_generation` — Sample building, locality grouping, ordered neurons
//! - `gpu_evaluation` — Batched GPU evaluation orchestration
//!   - `relu_evaluation` — `ReLU` candidate evaluation (split by error sign)
//!   - `activation_evaluation` — Non-ReLU activation candidate evaluation
//! - `filtering` — Candidate filtering, deduplication, truncation
//! - `target_analysis` — Per-target analysis loop (helpful, harmful, coordinated)
//! - `structural_patterns` — Coordinated structural discovery (noisy vs trusted, collapse hidden)
//! - `post_processing` — Impact discounting, sorting, diversification, metadata
//! - `orchestration` — Core analysis pipeline coordination
//! - `metadata` — Lock-free atomic metadata and result merging
//! - `results` — Result finalisation and assembly

mod activation_evaluation;
mod activation_subset_evaluation;
mod candidate_generation;
mod filtering;
mod gpu_evaluation;
pub(crate) mod holdout_validation;
mod metadata;
mod orchestration;
pub mod post_processing;
mod preparation;
mod relu_evaluation;
mod results;
mod scoring;
mod structural_patterns;
mod target_analysis;

// Re-export items used by code outside this module (analysis/mod.rs, neuron.rs, implementation_tests).
pub use scoring::{
    apply_activation_neuron_boost, apply_neuron_pessimism_discount, apply_pessimism_discount,
    apply_prediction_calibration, apply_source_type_boost, apply_synapse_pessimism_discount,
    apply_target_type_boost,
};

pub(crate) use candidate_generation::{
    MIN_GROUP_SIZE_FOR_LOCALITY, build_ordered_neurons, build_samples_for_locality_group,
    group_sources_by_locality,
};

pub use filtering::deterministic_coordinated_neuron_uuid;
pub(crate) use filtering::{
    ReplaceSynapseParams, expected_gain_replace_synapse_with_hidden_neuron,
    truncate_combined_synapse_candidate_sets,
};

pub(crate) use gpu_evaluation::{
    evaluate_all_activation_specs_batched, evaluate_relu_candidates_split,
};

pub(crate) use scoring::upsert_candidate;

#[cfg(test)]
pub(crate) use scoring::compute_candidate_dedup_key;

#[cfg(test)]
pub(crate) use scoring::compute_synapse_improvement_and_count;

// Test-only re-exports used by implementation_tests
#[cfg(test)]
pub(crate) use candidate_generation::build_samples;
#[cfg(test)]
pub(crate) use scoring::{
    compute_net_improvement_with_squash, compute_relu_improvement_and_count,
    compute_synapse_improvement_with_target_squash, count_improved_samples,
};

// =============================================================================
// Imports for the public API
// =============================================================================

use crate::AnalyzeSynapsesInput;
use anyhow::Result;

use crate::analysis::diagnostics::require_unique_focus;
use crate::analysis::gpu::GpuWorkQueue;
use crate::analysis::shared::AnalyzeSynapsesResult;

use super::cache::RecordCache;

use std::sync::Arc;

// =============================================================================
// Public API
// =============================================================================

/// Analyze synapses for a given input.
/// This is the public entry point for synapse analysis.
pub fn analyze_synapses(input: &AnalyzeSynapsesInput) -> Result<AnalyzeSynapsesResult> {
    // Validate focus_neurons before expensive pre-loading
    require_unique_focus(&input.focus_neurons, "Synapse analysis")?;

    // Pre-load all records for faster analysis (1 scan vs ~2000 scans)
    let cache = Arc::new(RecordCache::new_adaptive(&input.parquet_file)?);
    analyze_synapses_with_cache(input, cache)
}

/// Internal synapse analysis with shared cache.
/// This is called by `analyze_all` to share the cache between synapse and neuron analysis.
pub(crate) fn analyze_synapses_with_cache(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
) -> Result<AnalyzeSynapsesResult> {
    // Create GPU queue for this analysis
    let gpu_queue = Arc::new(GpuWorkQueue::new()?);
    orchestration::analyze_synapses_with_cache_impl(input, cache, gpu_queue)
}

/// Test-only helper for benchmarks that need to reuse GPU queue.
/// This allows benchmarks to avoid GPU initialisation overhead across iterations.
///
/// Note: This is public for use in external test files (tests/ directory).
pub fn analyze_synapses_with_cache_and_gpu_queue(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<GpuWorkQueue>,
) -> Result<AnalyzeSynapsesResult> {
    orchestration::analyze_synapses_with_cache_impl(input, cache, gpu_queue)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests;

// Issue #425: Implementation tests moved from implementation.rs as part of the refactoring.
// These test the core synapse analysis pipeline (GPU batch evaluation, diagnostics, etc.).
#[cfg(test)]
#[path = "../implementation_tests/mod.rs"]
mod implementation_tests;
