//! Analysis module for NEAT-AI Discovery
//!
//! This module provides functions for analysing recorded discovery data to identify
//! beneficial new synapses and neurons that would reduce error.
//!
//! The module is organised into focused submodules:
//! - `shared.rs` - Common types, result structures, diagnostics
//! - `synapse.rs` - Synapse analysis functions
//! - `neuron.rs` - Neuron analysis functions
//! - `gpu.rs` - GPU infrastructure (GpuAnalyzer, GpuWorkQueue)
//! - `utils.rs` - Utility functions (memory checks, deadlines)
//! - `system.rs` - System utilities facade (memory, GPU tier detection) (Issue #239)
//! - `activation.rs` - Activation function related code (Issue #266)
//! - `samples.rs` - Sample data structures and GPU formats (Issue #269)
//! - `diagnostics.rs` - Diagnostic tracking and rejection reasons (Issue #271)
//! - `cache.rs` - Record caching for parquet files (Issue #185)
//! - `streaming.rs` - Streaming parquet loading with block-based caching (Issue #193)
//! - `early_termination.rs` - SPRT-based early termination for GPU evaluation (Issue #219)

pub mod activation;
pub mod cache;
pub mod confidence;
pub mod diagnostics;
pub mod early_termination;
pub mod epistatic;
pub mod error_distribution;
pub mod gpu;
pub mod neuron;
pub mod redundant_path;
pub mod samples;
pub mod shared;
pub mod streaming;
pub mod synapse;
pub mod system;
pub mod utils;
pub mod weights;

// Implementation module - helper functions for neuron/synapse analysis
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

// Re-export confidence interval types (Issue #194)
pub use confidence::{compute_confidence_metrics, PredictionConfidenceMetrics};

// Re-export from utils
pub use utils::{gpu_timing_enabled, verbose_enabled};

// Re-export memory functions from utils (Issue #267)
pub use utils::check_memory_for_parquet;

// Re-export system utilities for backward compatibility (Issue #239)
// These are the primary types and functions for memory detection and GPU performance
pub use system::{
    // GPU batch size utilities
    cap_gpu_batch_size_by_bytes,
    // Memory tier classification
    categorise_memory_tier,
    // System requirements
    check_system_memory_requirements,
    // GPU performance tier
    detect_gpu_tier,
    detect_memory_tier,
    detect_unified_memory,
    // Memory detection
    get_memory_info,
    get_work_queue_capacity,
    get_work_queue_capacity_for_tier,
    // Parquet memory validation
    validate_parquet_memory_requirements,
    GpuPerformanceTier,
    MemoryTier,
    // GPU batch size constants
    DEFAULT_GPU_BATCH_SIZE,
    HIGH_PERF_GPU_BATCH_SIZE,
    LOW_MEMORY_GPU_BATCH_SIZE,
};

// Re-export Detail types from shared
pub use shared::{NeuronNoCandidateDetail, SynapseNoCandidateDetail};

// Re-export activation-related items from activation module (Issue #266, #238)
pub use activation::{
    // Activation functions
    absolute_activation,
    // GPU ID mapping
    activation_name_to_gpu_id,
    arctan_activation,
    bent_identity_activation,
    bipolar_activation,
    // Target simulation functions (Issue #238)
    can_use_hard_tanh,
    clipped_activation,
    elu_activation,
    gelu_activation,
    // Bias helpers
    get_bias_range,
    get_bias_values,
    get_target_activation_fn,
    get_target_simulation_fn,
    get_target_simulation_mode,
    hard_tanh_activation,
    has_sufficient_output_variance,
    identity_activation,
    // Activation predicates
    is_threshold_activation,
    logistic_activation,
    mish_activation,
    relu6_activation,
    softplus_activation,
    softsign_activation,
    tanh_activation,
    // Activation candidate spec
    ActivationCandidateSpec,
    TargetSimulationMode,
    ACTIVATION_SPECS,
    ORIENTATIONS_BIDIRECTIONAL,
    SCALES_SMOOTH,
    SCALES_WIDE,
};

// Re-export sample data structures from samples module (Issue #269)
pub use samples::{
    compute_dynamic_constant_source_threshold, compute_source_std_dev,
    compute_source_variance_discount, constant_source_effect_threshold_from_env,
    get_constant_source_threshold, ActivationOutput, ActivationUniforms, BiasResult, BiasUniforms,
    GpuHelpfulSample, HarmfulContribution, HarmfulStats, HarmfulUniforms, HelpfulContribution,
    HelpfulSample, HelpfulStats, HelpfulUniforms, NeuronStats, ReluContribution, ReluOrientation,
    ReluStats, ReluUniforms, DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD, EPSILON,
};

// Re-export focus_unused_observations_from_env for tests (Issue #182)
// Moved to utils/deadline module in Issue #268
pub use utils::focus_unused_observations_from_env;

// Re-export weight calculation functions from weights module (Issue #270)
pub use weights::{
    calculate_optimal_bias, calculate_optimal_identity_outgoing_and_bias,
    calculate_optimal_outgoing_weight, clamp_weight_update_delta,
    coordinated_structural_activation_delta, MAX_OUTGOING_WEIGHT,
};

// Re-export error distribution types (Issue #192)
pub use error_distribution::{
    detect_error_modes, outlier_analysis_enabled, outlier_percentile_from_env, ErrorDistribution,
    ErrorMode, OutlierReductionInfo,
};

// Implement analyze_all using the module functions
use crate::observability::{
    global_gpu_metrics, profile_mode, report_global_gpu_metrics, PhaseTimer, ProfileData,
    ProfileMode,
};
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
    phase_name: &'static str,
    f: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    if enabled {
        crate::watchdog::beat(starting);
        let _timer = PhaseTimer::new(phase_name);
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
        let (helpful, harmful, coordinated) = synapse::truncate_combined_synapse_candidate_sets(
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
    // Phase timer for total analysis (Issue #214)
    let _total_timer = PhaseTimer::new("total_analysis");

    // Profile data collection (when NEAT_AI_DISCOVERY_PROFILE=json)
    let mut profile = ProfileData::new();
    profile.set_focus_neurons_requested(input.focus_neurons.len());

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
    let parquet_loading_start = std::time::Instant::now();
    let shared_cache = Arc::new(cache::RecordCache::new_adaptive(&input.parquet_file)?);
    profile.record_phase(
        "parquet_loading",
        parquet_loading_start.elapsed().as_millis() as u64,
    );
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
                "synapse_analysis",
                || {
                    let inner = synapse_input.expect("checked is_some");
                    synapse::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            let neuron_result = run_optional_analysis(
                neuron_input.is_some(),
                "analysis::analyze_all → neuron analysis starting",
                "analysis::analyze_all → neuron analysis finished",
                "analysis::analyze_all → neuron analysis skipped",
                "neuron_analysis",
                || {
                    let inner = neuron_input.expect("checked is_some");
                    neuron::analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            (synapse_result, neuron_result)
        } else {
            let neuron_result = run_optional_analysis(
                neuron_input.is_some(),
                "analysis::analyze_all → neuron analysis starting",
                "analysis::analyze_all → neuron analysis finished",
                "analysis::analyze_all → neuron analysis skipped",
                "neuron_analysis",
                || {
                    let inner = neuron_input.expect("checked is_some");
                    neuron::analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache))
                },
            )?;

            let synapse_result = run_optional_analysis(
                synapse_input.is_some(),
                "analysis::analyze_all → synapse analysis starting",
                "analysis::analyze_all → synapse analysis finished",
                "analysis::analyze_all → synapse analysis skipped",
                "synapse_analysis",
                || {
                    let inner = synapse_input.expect("checked is_some");
                    synapse::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
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
            "neuron_analysis",
            || {
                let inner = neuron_input.expect("checked is_some");
                neuron::analyze_neurons_with_cache(&inner, Arc::clone(&shared_cache))
            },
        )?;

        let synapse_result = run_optional_analysis(
            synapse_input.is_some(),
            "analysis::analyze_all → synapse analysis starting",
            "analysis::analyze_all → synapse analysis finished",
            "analysis::analyze_all → synapse analysis skipped",
            "synapse_analysis",
            || {
                let inner = synapse_input.expect("checked is_some");
                synapse::analyze_synapses_with_cache(&inner, Arc::clone(&shared_cache))
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

            let new_neuron_uuid = synapse::deterministic_coordinated_neuron_uuid(
                &candidate.source_neuron_uuid,
                &candidate.target_neuron_uuid,
                &candidate.squash,
                candidate.incoming_weight,
                candidate.outgoing_weight,
                candidate.bias,
            );

            let mut expected_gain = synapse::expected_gain_replace_synapse_with_hidden_neuron(
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

    // Collect final profile data (Issue #214)
    let synapse_candidates = synapse_result
        .as_ref()
        .map(|s| {
            s.helpful_synapses.len()
                + s.harmful_synapses.len()
                + s.coordinated_structural_candidates.len()
        })
        .unwrap_or(0);
    let neuron_candidates = neuron_result
        .as_ref()
        .map(|n| n.helpful_neurons.len())
        .unwrap_or(0);
    let total_candidates = synapse_candidates + neuron_candidates;
    profile.set_candidates_found(total_candidates);
    profile.set_candidates_returned(total_candidates);

    // Set focus neurons completed from metadata
    let synapse_completed = synapse_result
        .as_ref()
        .map(|s| s.metadata.completed_focus_neurons)
        .unwrap_or(0);
    let neuron_completed = neuron_result
        .as_ref()
        .map(|n| n.metadata.completed_focus_neurons)
        .unwrap_or(0);
    profile.set_focus_neurons_completed(synapse_completed.max(neuron_completed));

    // Get GPU device info if available
    if let Some(info) = synapse_result
        .as_ref()
        .and_then(|s| s.metadata.gpu_info.as_ref())
        .or_else(|| {
            neuron_result
                .as_ref()
                .and_then(|n| n.metadata.gpu_info.as_ref())
        })
    {
        profile.set_gpu_device(info.name.clone());
    }

    // Add GPU metrics to profile data (Issue #214)
    let gpu_metrics = global_gpu_metrics();
    profile.from_gpu_metrics(gpu_metrics);

    // Output GPU metrics if enabled (Issue #214)
    report_global_gpu_metrics();

    // Output profile data if JSON profiling is enabled (Issue #214)
    if profile_mode() == ProfileMode::Json {
        profile.report();
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
// Re-export benchmark helper function for use in benches/
pub use synapse::analyze_synapses_with_cache_and_gpu_queue;

// Re-export early termination types (Issue #219)
pub use early_termination::{
    check_batch_early_termination, EarlyTerminationConfig, EarlyTerminationDecision,
    EarlyTerminationResult, SequentialEvaluator,
};

#[cfg(test)]
mod mod_tests;
