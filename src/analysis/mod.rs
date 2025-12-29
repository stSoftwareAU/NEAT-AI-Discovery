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
/// When `analysis_deadline_ms` is set and both analyses are enabled, **synapse analysis
/// runs first** to prevent starvation. Historically, neuron analysis ran first and could
/// consume the entire budget, leaving zero time for synapse analysis. This caused "no
/// add-synapses candidates" even when many potential synapses existed.
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
    // When deadline-constrained, synapse analysis runs FIRST to prevent starvation.
    // Without a deadline, neuron analysis runs first (original behaviour).
    let has_deadline = input.analysis_deadline_ms.is_some();

    let (synapse_result, neuron_result) = if has_deadline {
        // SYNAPSE-FIRST ordering (deadline-constrained):
        // Synapse analysis is often starved because neuron analysis consumes the budget.
        // Running synapses first ensures they get a fair share of the deadline.
        if utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Deadline set ({}ms) - running synapse analysis first to prevent starvation",
                input.analysis_deadline_ms.unwrap_or(0)
            );
        }

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
mod tests {
    use super::*;

    #[test]
    fn watchdog_beats_do_not_claim_finished_when_analysis_is_skipped() {
        let _lock = crate::watchdog::lock_for_test_serialisation();
        // Ensure a watchdog is active so `beat()` is observable in tests.
        let wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
            stall_timeout: std::time::Duration::from_secs(60),
            abort_delay: std::time::Duration::from_secs(1),
        });

        let skipped = "analysis::analyze_all → neuron analysis skipped";
        let finished = "analysis::analyze_all → neuron analysis finished";

        // When disabled, we should record "skipped" and never execute the closure.
        let result: Option<()> =
            run_optional_analysis(false, "starting", finished, skipped, || -> Result<()> {
                unreachable!("disabled analysis closure must not run")
            })
            .expect("should not error");
        assert!(result.is_none());
        assert_eq!(
            crate::watchdog::active_stage_for_test().as_deref(),
            Some(skipped)
        );

        // When enabled, we should end on "finished".
        let result: Option<()> =
            run_optional_analysis(true, "starting", finished, "skipped", || Ok(()))
                .expect("should not error");
        assert!(result.is_some());
        assert_eq!(
            crate::watchdog::active_stage_for_test().as_deref(),
            Some(finished)
        );

        drop(wd);
    }
}
