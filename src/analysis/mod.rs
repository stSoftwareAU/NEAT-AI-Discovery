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
//! - `utils.rs` - Utility functions (memory checks, deadlines)
//! - `activation.rs` - Activation function related code (Issue #266)
//!
//! For now, much of the code is still in `impl.rs` and will be gradually moved.

pub mod activation;
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
    // GPU timing types (Issue #195)
    AnalysisTiming,
    AnalyzeAllResult,
    AnalyzeNeuronsResult,
    AnalyzeSynapsesResult,
    CpuTimingBreakdown,
    // GPU info and zero-copy types (Issue #228)
    GpuAdapterInfo,
    GpuDeviceType,
    GpuTimingBreakdown,
    NeuronAnalysisMetadata,
    NeuronNoCandidateReason,
    NeuronNoCandidateSummary,
    ShaderTiming,
    SynapseAnalysisMetadata,
    SynapseNoCandidateReason,
    SynapseNoCandidateSummary,
    TimingCollector,
    TimingScope,
    ZeroCopyBufferConfig,
};

// Re-export from utils
pub use utils::{gpu_timing_enabled, verbose_enabled};

// Re-export memory functions from utils (Issue #267)
pub use utils::check_memory_for_parquet;

// Re-export Detail types from shared
pub use shared::{NeuronNoCandidateDetail, SynapseNoCandidateDetail};

// Re-export ACTIVATION_SPECS from activation module (Issue #266)
pub use activation::ACTIVATION_SPECS;

// Re-export focus_unused_observations_from_env for tests (Issue #182)
pub use implementation::focus_unused_observations_from_env;

// Implement analyze_all using the module functions
use crate::{
    AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput, CandidateNeuronJson,
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson,
};
use anyhow::Result;
use std::collections::HashMap;
use std::mem;
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

fn merge_coordinated_structural_replacements(
    synapse: &mut shared::AnalyzeSynapsesResult,
    mut replacements: Vec<CoordinatedStructuralCandidateJson>,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
) {
    if replacements.is_empty() {
        return;
    }

    synapse
        .coordinated_structural_candidates
        .append(&mut replacements);

    // Keep deterministic ordering for non-deadline runs.
    //
    // Deadline-diversified runs rely on upstream per-bucket ordering and round-robin selection,
    // so we avoid resorting here when `diversify` is enabled.
    if !diversify {
        synapse.coordinated_structural_candidates.sort_by(|a, b| {
            b.expected_creature_score_gain
                .partial_cmp(&a.expected_creature_score_gain)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    if let Some(limit) = max_synapse_candidates {
        let (helpful, harmful, coordinated) =
            implementation::truncate_combined_synapse_candidate_sets(
                mem::take(&mut synapse.helpful_synapses),
                mem::take(&mut synapse.harmful_synapses),
                mem::take(&mut synapse.coordinated_structural_candidates),
                limit,
                diversify,
            );
        synapse.helpful_synapses = helpful;
        synapse.harmful_synapses = harmful;
        synapse.coordinated_structural_candidates = coordinated;
    }

    // Ensure metadata reflects what we actually return to callers.
    synapse.metadata.candidates_returned = synapse.helpful_synapses.len()
        + synapse.harmful_synapses.len()
        + synapse.coordinated_structural_candidates.len();
}

fn apply_kept_neuron_candidates(
    neuron: &mut shared::AnalyzeNeuronsResult,
    kept_neurons: Vec<CandidateNeuronJson>,
) {
    // Regression fix (7-Jan-2026):
    // Post-processing may convert some add-neuron candidates into coordinated structural
    // replacements. When we filter those out of `helpful_neurons`, we must also update the
    // metadata so JSON output remains consistent with the returned arrays.
    neuron.helpful_neurons = kept_neurons;
    neuron.metadata.candidates_returned = neuron.helpful_neurons.len();
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

    // Post-process: convert certain add-neuron candidates into coordinated-structural replacements.
    //
    // Issue #173 (7-Jan-2026): When a direct synapse already exists between (source -> target),
    // applying an add-neuron candidate without removing the synapse is often not the intended edit.
    // We instead emit a coordinated group that removes the synapse and inserts the hidden neuron
    // path atomically.
    //
    // Important: This keeps normal add-neurons discovery working as-is for cases where no direct
    // synapse exists.
    let mut synapse_result = synapse_result;
    let mut neuron_result = neuron_result;

    if let (Some(syn), Some(neuron)) = (synapse_result.as_mut(), neuron_result.as_mut()) {
        // Build a quick lookup from (from_uuid,to_uuid) to existing weight.
        let mut direct_synapse_weight: HashMap<(String, String), f32> = HashMap::new();
        for s in &input.creature.synapses {
            direct_synapse_weight.insert((s.from_uuid.clone(), s.to_uuid.clone()), s.weight);
        }

        let mut kept_neurons: Vec<CandidateNeuronJson> =
            Vec::with_capacity(neuron.helpful_neurons.len());
        let mut replacements: Vec<CoordinatedStructuralCandidateJson> = Vec::new();

        for candidate in &neuron.helpful_neurons {
            let key = (
                candidate.source_neuron_uuid.clone(),
                candidate.target_neuron_uuid.clone(),
            );
            let Some(&old_weight) = direct_synapse_weight.get(&key) else {
                kept_neurons.push(candidate.clone());
                continue;
            };

            let new_neuron_uuid = implementation::deterministic_coordinated_neuron_uuid(
                &candidate.source_neuron_uuid,
                &candidate.target_neuron_uuid,
                &candidate.squash,
                candidate.incoming_weight,
                candidate.outgoing_weight,
                candidate.bias,
            );

            let mut expected_gain =
                implementation::expected_gain_replace_synapse_with_hidden_neuron(
                    shared_cache.as_ref(),
                    &candidate.source_neuron_uuid,
                    &candidate.target_neuron_uuid,
                    old_weight,
                    candidate.incoming_weight,
                    candidate.outgoing_weight,
                    candidate.bias,
                    &candidate.squash,
                )
                .unwrap_or(candidate.expected_creature_score_gain);

            // Apply a conservative impact discount when the target is hidden.
            // This mirrors the synapse/neurons discounting semantics without requiring deep graph analysis.
            let is_target_output = input
                .creature
                .neurons
                .iter()
                .find(|n| n.uuid == candidate.target_neuron_uuid)
                .map(|n| n.neuron_type == "output")
                .unwrap_or(false);
            if !is_target_output {
                expected_gain *= 0.1;
            }

            replacements.push(CoordinatedStructuralCandidateJson {
                operations: vec![
                    CoordinatedStructuralOpJson::RemoveSynapse {
                        from_neuron_uuid: candidate.source_neuron_uuid.clone(),
                        to_neuron_uuid: candidate.target_neuron_uuid.clone(),
                    },
                    CoordinatedStructuralOpJson::AddNeuron {
                        neuron_uuid: new_neuron_uuid.clone(),
                        neuron_type: "hidden".to_string(),
                        squash: candidate.squash.clone(),
                        bias: candidate.bias,
                        insert_before_neuron_uuid: Some(candidate.target_neuron_uuid.clone()),
                    },
                    CoordinatedStructuralOpJson::AddSynapse {
                        from_neuron_uuid: candidate.source_neuron_uuid.clone(),
                        to_neuron_uuid: new_neuron_uuid.clone(),
                        weight: candidate.incoming_weight,
                    },
                    CoordinatedStructuralOpJson::AddSynapse {
                        from_neuron_uuid: new_neuron_uuid,
                        to_neuron_uuid: candidate.target_neuron_uuid.clone(),
                        weight: candidate.outgoing_weight,
                    },
                ],
                expected_creature_score_gain: expected_gain,
                comment: Some(format!(
                    "Coordinated replacement: remove synapse and insert {} hidden neuron",
                    candidate.squash
                )),
            });
        }

        // Keep non-replacement add-neurons unchanged.
        apply_kept_neuron_candidates(neuron, kept_neurons);

        merge_coordinated_structural_replacements(
            syn,
            replacements,
            input.max_synapse_candidates,
            input.analysis_deadline_ms.is_some(),
        );
    }

    Ok(AnalyzeAllResult {
        synapse: synapse_result,
        neuron: neuron_result,
    })
}

// Re-export from modules
pub use gpu::supports_unified_memory;
pub use gpu::GpuAnalyzer;
pub use gpu::GpuAvailabilityResult;
pub use neuron::analyze_neurons;
pub use synapse::analyze_synapses;

#[cfg(test)]
mod mod_tests;
