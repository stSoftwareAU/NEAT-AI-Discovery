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
    AnalyzeAllResult, AnalyzeNeuronsResult, AnalyzeSynapsesResult, NeuronAnalysisMetadata,
    NeuronNoCandidateReason, NeuronNoCandidateSummary, SynapseAnalysisMetadata,
    SynapseNoCandidateReason, SynapseNoCandidateSummary,
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
use std::time::SystemTime;

/// Choose analysis ordering when deadline-constrained.
///
/// Rationale (2 Jan 2026):
/// - Production discovery runs are deadline-constrained and repeated over time.
/// - We randomise (time-vary) the order so that, across repeated runs, both analyses get a turn
///   running first under the same global deadline.
///
/// Notes:
/// - `random_seed` is included to allow reproducibility in tests and debugging.
/// - We deliberately mix in the current time so repeated calls with the same seed can still vary.
fn choose_deadline_order_synapse_first(random_seed: Option<u64>, now_ms: u64) -> bool {
    // A tiny, deterministic "coin flip": parity of (seed XOR time).
    //
    // This is good enough for long-run fairness (50/50 over time) and is easy to test.
    ((random_seed.unwrap_or(0) ^ now_ms) & 1) == 0
}

fn run_optional_analysis<T>(
    enabled: bool,
    starting: &'static str,
    finished: &'static str,
    skipped: &'static str,
    f: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    if enabled {
        crate::watchdog::beat(starting);
        let result = f()?;
        crate::watchdog::beat(finished);
        Ok(Some(result))
    } else {
        crate::watchdog::beat(skipped);
        Ok(None)
    }
}

/// Combined analysis function that runs both synapse and neuron analysis.
///
/// # Analysis ordering
///
/// When `analysis_deadline_ms` is set and both analyses are enabled, the library **randomises
/// the run order** on each invocation. This means one run may return only synapse candidates
/// (neuron starved) and the next may return only neuron candidates (synapse starved).
///
/// When no deadline is set, the original "neuron-first" ordering is preserved for
/// backwards compatibility (neuron discovery creates new network structure and may be
/// considered higher value when time is not constrained).
pub fn analyze_all(input: &AnalyzeAllInput) -> Result<AnalyzeAllResult> {
    // Optional hang watchdog for unattended workers.
    // If enabled, this will emit a thread dump then abort the process if analysis stalls.
    let _watchdog = crate::watchdog::start_from_env("analysis::analyze_all");

    let include_synapse = input.include_synapse_analysis.unwrap_or(true);
    let include_neuron = input.include_neuron_analysis.unwrap_or(true);

    if !include_synapse && !include_neuron {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
        });
    }

    crate::watchdog::beat("analysis::analyze_all → loading parquet cache");

    // Pre-load ALL records from parquet in one pass. This is MUCH faster than
    // lazy-loading each neuron separately (1 scan vs ~2000 scans for large creatures).
    let shared_cache = Arc::new(implementation::RecordCache::new_adaptive(
        &input.parquet_file,
    )?);
    crate::watchdog::beat("analysis::analyze_all → parquet cache loaded");

    let synapse_input = if include_synapse {
        Some(AnalyzeSynapsesInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            max_candidates: input.max_synapse_candidates,
            analysis_deadline_ms: input.analysis_deadline_ms,
            random_seed: input.random_seed,
        })
    } else {
        None
    };

    let neuron_input = if include_neuron {
        Some(AnalyzeNeuronsInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: input.focus_neurons.clone(),
            max_candidates: input.max_neuron_candidates,
            analysis_deadline_ms: input.analysis_deadline_ms,
            random_seed: input.random_seed,
        })
    } else {
        None
    };

    // Determine analysis order based on whether a deadline is set.
    // When deadline-constrained, we randomise ordering so that repeated runs provide
    // long-run coverage even though an individual run can return partial results.
    // Without a deadline, neuron analysis runs first (original behaviour).
    let has_deadline = input.analysis_deadline_ms.is_some();

    let (synapse_result, neuron_result) = if has_deadline {
        // Deadline-constrained: randomised ordering.
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let synapse_first = include_synapse
            && include_neuron
            && choose_deadline_order_synapse_first(input.random_seed, now_ms);

        if utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Deadline set ({}ms) - randomised ordering: {} first",
                input.analysis_deadline_ms.unwrap_or(0),
                if synapse_first { "synapse" } else { "neuron" }
            );
        }

        if synapse_first {
            let synapse_result = run_optional_analysis(
                synapse_input.is_some(),
                "analysis::analyze_all → synapse analysis starting",
                "analysis::analyze_all → synapse analysis finished",
                "analysis::analyze_all → synapse analysis skipped",
                || {
                    let inner = synapse_input.expect("checked is_some");
                    implementation::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            let neuron_result = run_optional_analysis(
                neuron_input.is_some(),
                "analysis::analyze_all → neuron analysis starting",
                "analysis::analyze_all → neuron analysis finished",
                "analysis::analyze_all → neuron analysis skipped",
                || {
                    let inner = neuron_input.expect("checked is_some");
                    implementation::analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            (synapse_result, neuron_result)
        } else {
            let neuron_result = run_optional_analysis(
                neuron_input.is_some(),
                "analysis::analyze_all → neuron analysis starting",
                "analysis::analyze_all → neuron analysis finished",
                "analysis::analyze_all → neuron analysis skipped",
                || {
                    let inner = neuron_input.expect("checked is_some");
                    implementation::analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            let synapse_result = run_optional_analysis(
                synapse_input.is_some(),
                "analysis::analyze_all → synapse analysis starting",
                "analysis::analyze_all → synapse analysis finished",
                "analysis::analyze_all → synapse analysis skipped",
                || {
                    let inner = synapse_input.expect("checked is_some");
                    implementation::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            (synapse_result, neuron_result)
        }
    } else {
        // NEURON-FIRST ordering (no deadline - original behaviour):
        // Neuron discovery is more valuable as it can create new network structure.
        // With pre-loaded cache, both run fast, but neurons get priority.
        let neuron_result = run_optional_analysis(
            neuron_input.is_some(),
            "analysis::analyze_all → neuron analysis starting",
            "analysis::analyze_all → neuron analysis finished",
            "analysis::analyze_all → neuron analysis skipped",
            || {
                let inner = neuron_input.expect("checked is_some");
                implementation::analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache))
            },
        )?;

        let synapse_result = run_optional_analysis(
            synapse_input.is_some(),
            "analysis::analyze_all → synapse analysis starting",
            "analysis::analyze_all → synapse analysis finished",
            "analysis::analyze_all → synapse analysis skipped",
            || {
                let inner = synapse_input.expect("checked is_some");
                implementation::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
            },
        )?;

        (synapse_result, neuron_result)
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

#[cfg(test)]
mod mod_tests;
