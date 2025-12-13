//! Analysis module for NEAT-AI Discovery
//!
//! This module provides functions for analysing recorded discovery data to identify
//! beneficial new synapses and neurons that would reduce error.
//!
//! **Note**: This module is currently being refactored from a single large file (~14k lines)
//! into focused submodules. The target structure is:
//! - `shared.rs` - Common types, result structures, diagnostics
//! - `synapse.rs` - Synapse analysis functions
//! - `neuron.rs` - Neuron analysis functions  
//! - `gpu.rs` - GPU infrastructure (GpuAnalyzer, GpuWorkQueue)
//! - `utils.rs` - Utility functions (memory checks, deadlines, activation functions)
//!
//! For now, everything is still in `impl.rs` and will be gradually moved.

pub mod gpu;
pub mod neuron;
pub mod shared;
pub mod synapse;
pub mod utils;

// Implementation module - contains all the analysis code
// TODO: Gradually extract pieces into focused modules (synapse.rs, neuron.rs, gpu.rs, utils.rs)
mod implementation;

// Re-export shared types
pub use shared::{
    AnalyzeAllResult, AnalyzeNeuronsResult, AnalyzeSynapsesResult, NeuronNoCandidateReason,
    NeuronNoCandidateSummary, SynapseNoCandidateReason, SynapseNoCandidateSummary,
};

// Re-export from utils
pub use utils::verbose_enabled;

// Re-export Detail types from shared
pub use shared::{NeuronNoCandidateDetail, SynapseNoCandidateDetail};

// Re-export check_memory_for_parquet and ACTIVATION_SPECS from implementation (hasn't been moved yet)
pub use implementation::{check_memory_for_parquet, ACTIVATION_SPECS};

// Implement analyze_all using the module functions
use crate::{AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput};
use anyhow::Result;
use std::sync::Arc;

/// Combined analysis function that runs both synapse and neuron analysis.
pub fn analyze_all(input: &AnalyzeAllInput) -> Result<AnalyzeAllResult> {
    let include_synapse = input.include_synapse_analysis.unwrap_or(true);
    let include_neuron = input.include_neuron_analysis.unwrap_or(true);

    if !include_synapse && !include_neuron {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
        });
    }

    // Pre-load ALL records from parquet in one pass. This is MUCH faster than
    // lazy-loading each neuron separately (1 scan vs ~2000 scans for large creatures).
    let shared_cache = Arc::new(implementation::RecordCache::new_adaptive(
        &input.parquet_file,
    )?);

    let synapse_input = if include_synapse {
        Some(AnalyzeSynapsesInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            improvement_threshold: input.improvement_threshold,
            max_candidates: input.max_synapse_candidates,
            analysis_deadline_ms: input.analysis_deadline_ms,
        })
    } else {
        None
    };

    let neuron_input = if include_neuron {
        Some(AnalyzeNeuronsInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            improvement_threshold: input.improvement_threshold,
            max_candidates: input.max_neuron_candidates,
            analysis_deadline_ms: input.analysis_deadline_ms,
        })
    } else {
        None
    };

    // Run neuron analysis FIRST (priority), then synapse analysis.
    // Neuron discovery is more valuable as it can create new network structure.
    // With pre-loaded cache, both run fast, but neurons get priority if timeout approaches.
    let neuron_result = if let Some(inner) = neuron_input.clone() {
        Some(implementation::analyze_neurons_with_cache(
            &inner,
            Arc::clone(&shared_cache),
        )?)
    } else {
        None
    };

    let synapse_result = if let Some(inner) = synapse_input.clone() {
        Some(implementation::analyze_synapses_with_cache(
            &inner,
            Arc::clone(&shared_cache),
        )?)
    } else {
        None
    };

    Ok(AnalyzeAllResult {
        synapse: synapse_result,
        neuron: neuron_result,
    })
}

// Re-export from modules
pub use gpu::GpuAnalyzer;
pub use gpu::GpuAvailabilityResult;
pub use neuron::analyze_neurons;
pub use synapse::analyze_synapses;
