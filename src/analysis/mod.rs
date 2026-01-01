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

/// Split a global analysis deadline into two absolute (epoch-ms) deadlines.
///
/// This is designed for timeout-constrained production loops that call discovery repeatedly.
/// We want predictable coverage over time: both synapse and neuron analysis should get a share
/// of the budget on each invocation.
///
/// Notes:
/// - `raw_deadline_ms` can be either a relative duration (small values) or an absolute timestamp
///   (epoch milliseconds). This mirrors the parsing rules used by the analysis implementation.
/// - Returns `None` when the deadline is an absolute timestamp in the past (treated as “no deadline”
///   by the downstream implementation). In that case, callers should preserve the original value.
fn split_global_deadline_ms(
    raw_deadline_ms: u64,
    now_ms: u64,
    synapse_share: f64,
) -> Option<(u64, u64)> {
    // Mirror implementation bounds.
    const YEAR_2000_MS: u64 = 946_684_800_000;
    const MIN_DURATION_MS: u64 = 3_000; // 3 seconds
    const MAX_DURATION_MS: u64 = 3_600_000; // 1 hour

    let relative_ms_opt: Option<u64> = if raw_deadline_ms < YEAR_2000_MS {
        Some(raw_deadline_ms)
    } else if raw_deadline_ms <= now_ms {
        // Absolute deadline is in the past: downstream treats this as no deadline.
        None
    } else {
        Some(raw_deadline_ms - now_ms)
    };

    let total_ms = match relative_ms_opt {
        None => return None,
        Some(relative_ms) => {
            if !(MIN_DURATION_MS..=MAX_DURATION_MS).contains(&relative_ms) {
                // IMPORTANT (2 Jan 2026):
                // Do NOT clamp/normalise invalid durations here. The downstream implementation
                // (`calculate_effective_timeout_ms`) is responsible for clamping AND emitting the
                // documented warning messages. If we intercept here, the warnings never appear,
                // breaking observability for invalid configs.
                //
                // Returning `None` means "do not split" and the caller will pass the original
                // raw deadline through untouched.
                return None;
            } else {
                relative_ms
            }
        }
    };

    let total_end = now_ms.saturating_add(total_ms);

    // If the budget is too small to safely split (we require >= 3s per analysis window),
    // do not split. This avoids triggering the downstream “<3s => default 10 minutes” guard.
    if total_ms < MIN_DURATION_MS.saturating_mul(2) {
        return Some((total_end, total_end));
    }

    let share = synapse_share.clamp(0.0, 1.0);
    let synapse_ms = ((total_ms as f64) * share).round() as u64;
    let synapse_ms_max = total_ms.saturating_sub(MIN_DURATION_MS);
    let synapse_ms = synapse_ms.clamp(MIN_DURATION_MS, synapse_ms_max);
    let synapse_end = now_ms.saturating_add(synapse_ms);

    Some((synapse_end, total_end))
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
    //
    // Coverage under timeouts:
    // In production, the controller often supplies `analysis_deadline_ms` and repeats discovery
    // over time (e.g., GRQ’s `worker/Discovery/run.sh`). If we pass the same deadline to both
    // analyses, the first can consume the entire budget and starve the second.
    //
    // To ensure coverage over time, we split the *global* deadline into two time windows and
    // pass per-analysis absolute deadlines. This guarantees both synapse and neuron analysis
    // get a share of the budget each invocation.
    let has_deadline = input.analysis_deadline_ms.is_some();

    // Build per-analysis deadlines (absolute millisecond timestamps) when both analyses are enabled.
    // This keeps the total wall-clock budget bounded while ensuring both analyses run.
    //
    // Note: We intentionally prefer synapse analysis under deadlines, but we also reserve time
    // for neuron analysis so it is not starved.
    let (synapse_deadline_ms, neuron_deadline_ms) =
        if has_deadline && include_synapse && include_neuron {
            // Share of the *global* budget reserved for synapse analysis.
            // This is a tuning knob; 0.7 biases toward synapse discovery (historically starved).
            const SYNAPSE_SHARE: f64 = 0.7;

            // has_deadline is true here, so unwrap is safe.
            let raw = input
                .analysis_deadline_ms
                .expect("has_deadline implies analysis_deadline_ms is set");
            let now_ms = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .ok()
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            match split_global_deadline_ms(raw, now_ms, SYNAPSE_SHARE) {
                Some((syn_end, total_end)) => (Some(syn_end), Some(total_end)),
                None => (input.analysis_deadline_ms, input.analysis_deadline_ms),
            }
        } else {
            (input.analysis_deadline_ms, input.analysis_deadline_ms)
        };

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
                let mut inner = synapse_input.expect("checked is_some");
                inner.analysis_deadline_ms = synapse_deadline_ms;
                implementation::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
            },
        )?;

        let neuron_result = run_optional_analysis(
            neuron_input.is_some(),
            "analysis::analyze_all → neuron analysis starting",
            "analysis::analyze_all → neuron analysis finished",
            "analysis::analyze_all → neuron analysis skipped",
            || {
                let mut inner = neuron_input.expect("checked is_some");
                inner.analysis_deadline_ms = neuron_deadline_ms;
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
mod mod_tests;
