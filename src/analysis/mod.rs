//! Analysis module for NEAT-AI Discovery
//!
//! This module provides functions for analysing recorded discovery data to identify
//! beneficial new synapses and neurons that would reduce error.
//!
//! The module is organised into thematic subdirectories (Issue #528):
//!
//! ## Thematic subdirectories
//! - `detection/` - Pattern detection modules (saturation, bottleneck, dead neuron, etc.)
//! - `recommendation/` - Candidate recommendation engines (epistatic, multi-hop, etc.)
//! - `scoring/` - Scoring, confidence, and validation (weights, confidence, etc.)
//!
//! ## Core modules (root level)
//! - `shared.rs` - Common types, result structures, diagnostics
//! - `synapse/` - Synapse analysis pipeline
//! - `neuron.rs` - Neuron analysis functions
//! - `gpu/` - GPU infrastructure (GpuAnalyzer, GpuWorkQueue)
//! - `utils/` - Utility functions (memory checks, deadlines)
//! - `diagnostics/` - Diagnostic tracking and rejection reasons
//! - `system.rs` - System utilities facade (memory, GPU tier detection)
//! - `activation.rs` - Activation function related code
//! - `samples.rs` - Sample data structures and GPU formats
//! - `constants.rs` - Central discovery thresholds and constants
//! - `cache.rs` - Record caching for parquet files
//! - `candidate_cache.rs` - Candidate outcome cache for success/failure tracking
//! - `streaming.rs` - Streaming parquet loading with block-based caching
//! - `discovery_dispatch.rs` - Generic discovery module dispatch pattern
//! - `candidate_clustering.rs` - Candidate clustering to reduce redundant ablation tests
//! - `module_weights.rs` - Per-module success rate tracking for adaptive weighting
//! - `neuron_fingerprint.rs` - Neuron structural fingerprinting for incremental analysis
//! - `early_termination.rs` - SPRT-based early termination for GPU evaluation

// Core modules (remain at root level)
pub mod activation;
pub mod cache;
pub mod candidate_cache;
pub mod candidate_clustering;
pub mod constants;
pub mod diagnostics;
pub mod discovery_dispatch;
pub mod early_termination;
pub mod gpu;
pub mod module_weights;
pub mod neuron;
pub mod neuron_fingerprint;
pub mod samples;
pub mod shared;
pub mod streaming;
pub mod synapse;
pub mod system;
pub mod utils;

// Thematic subdirectories (Issue #528)
pub mod detection;
pub mod recommendation;
pub mod scoring;

// Re-export detection modules at analysis level for backward compatibility
pub use detection::activation_mismatch;
pub use detection::bottleneck;
pub use detection::bounded_range;
pub use detection::correlated_error;
pub use detection::dead_neuron;
pub use detection::dormant_synapse;
pub use detection::error_plateau;
pub use detection::input_sensitivity;
pub use detection::noise_signal;
pub use detection::observation_range;
pub use detection::observation_utilisation;
pub use detection::operating_point;
pub use detection::opposing_synapse;
pub use detection::oscillating_neuron;
pub use detection::output_squash_mismatch;
pub use detection::redundant_path;
pub use detection::restricted_range;
pub use detection::saturation;
pub use detection::sentinel_gating;
pub use detection::squash_weight_rescale;
pub use detection::topology;
pub use detection::topology_diversification;
pub use detection::unbounded_capping;
pub use detection::weight_coherence;
pub use detection::weight_magnitude_reset;

// Re-export recommendation modules at analysis level for backward compatibility
pub use recommendation::activation_recommendation;
pub use recommendation::epistatic;
pub use recommendation::gradient_discovery;
pub use recommendation::multi_hop;
pub use recommendation::output_bias_drift;
pub use recommendation::sample_weighted;

// Re-export scoring modules at analysis level for backward compatibility
pub use scoring::confidence;
pub use scoring::cross_validation;
pub use scoring::error_distribution;
pub use scoring::weights;

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
pub use confidence::{PredictionConfidenceMetrics, compute_confidence_metrics};

// Re-export from utils
pub use utils::{gpu_timing_enabled, verbose_enabled};

// Re-export memory functions from utils (Issue #267)
pub use utils::check_memory_for_parquet;

// Re-export system utilities for backward compatibility (Issue #239)
// These are the primary types and functions for memory detection and GPU performance
pub use system::{
    // GPU batch size constants
    DEFAULT_GPU_BATCH_SIZE,
    GpuPerformanceTier,
    HIGH_PERF_GPU_BATCH_SIZE,
    LOW_MEMORY_GPU_BATCH_SIZE,
    MemoryTier,
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
};

// Re-export Detail types from shared
pub use shared::{NeuronNoCandidateDetail, SynapseNoCandidateDetail};

// Re-export activation-related items from activation module (Issue #266, #238)
pub use activation::{
    ACTIVATION_SPECS,
    // Activation candidate spec
    ActivationCandidateSpec,
    ORIENTATIONS_BIDIRECTIONAL,
    SCALES_SMOOTH,
    SCALES_WIDE,
    TargetSimulationMode,
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
};

// Re-export sample data structures from samples module (Issue #269)
pub use samples::{
    ActivationOutput, ActivationUniforms, BiasResult, BiasUniforms,
    DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD, EPSILON, GpuHelpfulSample, HarmfulContribution,
    HarmfulStats, HarmfulUniforms, HelpfulContribution, HelpfulSample, HelpfulStats,
    HelpfulUniforms, NeuronStats, ReluContribution, ReluOrientation, ReluStats, ReluUniforms,
    compute_dynamic_constant_source_threshold, compute_source_std_dev,
    compute_source_variance_discount, constant_source_effect_threshold_from_env,
    get_constant_source_threshold,
};

// Re-export focus_unused_observations_from_env for tests (Issue #182)
// Moved to utils/deadline module in Issue #268
pub use utils::focus_unused_observations_from_env;

// Re-export weight calculation functions from weights module (Issue #270, #402)
pub use weights::{
    DEFAULT_SENTINEL_TOLERANCE, MAX_OUTGOING_WEIGHT, calculate_optimal_bias,
    calculate_optimal_identity_outgoing_and_bias, calculate_optimal_outgoing_weight,
    calculate_range_aware_weight, clamp_weight_update_delta, compute_range_aware_sums,
    coordinated_structural_activation_delta,
};

// Re-export error distribution types (Issue #192)
pub use error_distribution::{
    ErrorDistribution, ErrorMode, OutlierReductionInfo, detect_error_modes,
    outlier_analysis_enabled, outlier_percentile_from_env,
};

// Implement analyze_all using the module functions
use crate::observability::{
    PhaseTimer, ProfileData, ProfileMode, global_gpu_metrics, profile_mode,
    report_global_gpu_metrics,
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

    // Issue #557: Filter out candidates with non-positive expected_creature_score_gain.
    // Only candidates predicted to improve the creature's score should be returned.
    replacements.retain(|c| c.expected_creature_score_gain > 0.0);
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
                .total_cmp(&a.expected_creature_score_gain)
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

    // Issue #490: Compute current fingerprints and filter unchanged neurons.
    let current_fingerprints = neuron_fingerprint::compute_neuron_fingerprints(&input.creature);
    let (effective_focus_neurons, fingerprint_cache_hits, fingerprint_cache_misses) = if let Some(
        prev_fp,
    ) =
        &input.previous_neuron_fingerprints
    {
        let filter_result = neuron_fingerprint::filter_changed_neurons(
            &input.focus_neurons,
            &input.creature,
            prev_fp,
        );

        if utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Incremental analysis (Issue #490): {}/{} focus neurons unchanged (skipped), {} to analyse",
                filter_result.cache_hits,
                filter_result.total_focus_neurons,
                filter_result.cache_misses,
            );
        }

        (
            filter_result.changed,
            filter_result.cache_hits,
            filter_result.cache_misses,
        )
    } else {
        let len = input.focus_neurons.len();
        (input.focus_neurons.clone(), 0, len)
    };

    if !include_synapse && !include_neuron {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
        });
    }

    // If all focus neurons were skipped by fingerprint filtering, return early.
    if effective_focus_neurons.is_empty() && (include_synapse || include_neuron) {
        if utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] All focus neurons unchanged — skipping GPU analysis",
            );
        }
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
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
            focus_neurons: effective_focus_neurons.clone(),
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
            focus_neurons: effective_focus_neurons.clone(),
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

    // Issue #375 / Issue #419: Discovery module dispatch using parallel pattern.
    // Each detection module is dispatched via `run_discovery_modules_parallel` which
    // runs all detection phases concurrently, then merges results sequentially.
    if let Some(syn) = synapse_result.as_mut() {
        let max_candidates = input.max_synapse_candidates;
        let diversify = input.analysis_deadline_ms.is_some();

        // Wrap shared data in Arc for Send closures (Issue #419)
        let creature = Arc::new(input.creature.clone());
        let hidden_neurons: Arc<Vec<(String, String, f32)>> = Arc::new(
            input
                .creature
                .neurons
                .iter()
                .filter(|n| n.neuron_type == "hidden")
                .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
                .collect(),
        );

        let mut modules: Vec<discovery_dispatch::DiscoveryModuleSpec> = Vec::with_capacity(28);

        // Issue #342: Saturated neuron detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "saturation detection".to_string(),
                phase_name: "saturation_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected = saturation::detect_saturated_neurons(&hidden, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        saturation::saturated_neurons_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #343: Bottleneck neuron detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "bottleneck detection".to_string(),
                phase_name: "bottleneck_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected = bottleneck::detect_bottleneck_neurons(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates = bottleneck::bottleneck_neurons_to_coordinated_candidates(
                        &detected, &creature,
                    );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #341: Dead neuron detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "dead neuron detection".to_string(),
                phase_name: "dead_neuron_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected = dead_neuron::detect_dead_neurons(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates = dead_neuron::dead_neurons_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #344: Correlated error detection
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "correlated error detection".to_string(),
                phase_name: "correlated_error_detection",
                detect_fn: Box::new(move || {
                    let output_count = creature
                        .neurons
                        .iter()
                        .filter(|n| n.neuron_type == "output")
                        .count();
                    if output_count < 2 {
                        return None;
                    }
                    let records =
                        cache.load_records_for_neuron_types(&creature, &["output", "input"]);
                    let detected =
                        correlated_error::detect_correlated_error_patterns(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates = correlated_error::correlated_errors_to_coordinated_candidates(
                        &detected, &creature,
                    );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #230: Multi-hop candidate analysis
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "multi-hop analysis".to_string(),
                phase_name: "multi_hop_analysis",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_all_neurons(&creature);
                    let detected = multi_hop::detect_multi_hop_candidates(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        multi_hop::multi_hop_to_coordinated_candidates(&detected, &creature);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #358: Oscillating neuron detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "oscillating neuron detection".to_string(),
                phase_name: "oscillating_neuron_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected =
                        oscillating_neuron::detect_oscillating_neurons(&hidden, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        oscillating_neuron::oscillating_neurons_to_coordinated_candidates(
                            &detected,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #359: Dormant synapse detection
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "dormant synapse detection".to_string(),
                phase_name: "dormant_synapse_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_synapse_sources(&creature);
                    let detected = dormant_synapse::detect_dormant_synapses(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        dormant_synapse::dormant_synapses_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #360: Opposing synapse detection
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "opposing synapse detection".to_string(),
                phase_name: "opposing_synapse_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    let detected = opposing_synapse::detect_opposing_synapses(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        opposing_synapse::opposing_synapses_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #361: Output bias drift detection
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "output bias drift detection".to_string(),
                phase_name: "output_bias_drift_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                    let detected = output_bias_drift::detect_output_bias_drift(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        output_bias_drift::output_bias_drift_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #395: Bounded range detection
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "bounded range detection".to_string(),
                phase_name: "bounded_range_detection",
                detect_fn: Box::new(move || {
                    let records =
                        cache.load_records_for_neuron_types(&creature, &["input", "hidden"]);
                    if records.is_empty() {
                        return None;
                    }
                    let detected = bounded_range::detect_bounded_range_neurons(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        bounded_range::bounded_range_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #400: Sentinel value gating
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "sentinel value gating".to_string(),
                phase_name: "sentinel_value_gating",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_neuron_types(&creature, &["input"]);
                    if records.is_empty() {
                        return None;
                    }
                    let detected =
                        sentinel_gating::detect_sentinel_gating_candidates(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates = sentinel_gating::sentinel_gating_to_coordinated_candidates(
                        &detected, &creature,
                    );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #543: Observation utilisation detection
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "observation utilisation detection".to_string(),
                phase_name: "observation_utilisation_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_neuron_types(&creature, &["input"]);
                    if records.is_empty() {
                        return None;
                    }
                    let detected = observation_utilisation::detect_underutilised_observations(
                        &creature, &records,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        observation_utilisation::observation_utilisation_to_coordinated_candidates(
                            &detected, &creature,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #399: Restricted activation range detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "restricted range detection".to_string(),
                phase_name: "restricted_range_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let config = restricted_range::RestrictedRangeConfig::default();
                    let detected = restricted_range::detect_restricted_range_neurons(
                        &creature, &records, &config,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates = restricted_range::restricted_range_to_coordinated_candidates(
                        &detected, &creature,
                    );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #401: Hidden neuron operating-point analysis
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "operating point analysis".to_string(),
                phase_name: "operating_point_analysis",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let config = operating_point::OperatingPointConfig::default();
                    let detected = operating_point::detect_operating_point_issues(
                        &creature, &records, &config,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates = operating_point::operating_point_to_coordinated_candidates(
                        &detected, &creature,
                    );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #441: Unbounded activation capping detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "unbounded capping detection".to_string(),
                phase_name: "unbounded_capping_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected =
                        unbounded_capping::detect_unbounded_capping_candidates(&hidden, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        unbounded_capping::unbounded_capping_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #434: Noise-to-signal ratio detection for neurons
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "noisy neuron detection".to_string(),
                phase_name: "noisy_neuron_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected = noise_signal::detect_noisy_neurons(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        noise_signal::noisy_neurons_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #434: Noise-to-signal ratio detection for synapses
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "noisy synapse detection".to_string(),
                phase_name: "noisy_synapse_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    let detected = noise_signal::detect_noisy_synapses(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        noise_signal::noisy_synapses_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #435: Input sensitivity analysis for dominant inputs
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "dominant input detection".to_string(),
                phase_name: "dominant_input_detection",
                detect_fn: Box::new(move || {
                    let records =
                        cache.load_records_for_neuron_types(&creature, &["input", "output"]);
                    if records.is_empty() {
                        return None;
                    }
                    let config = input_sensitivity::InputSensitivityConfig::default();
                    let detected =
                        input_sensitivity::detect_dominant_inputs(&creature, &records, &config);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        input_sensitivity::dominant_inputs_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #435: Input sensitivity analysis for threshold effects
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "threshold effect detection".to_string(),
                phase_name: "threshold_effect_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    let config = input_sensitivity::InputSensitivityConfig::default();
                    let detected =
                        input_sensitivity::detect_threshold_effects(&creature, &records, &config);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        input_sensitivity::threshold_effects_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #437: Weight coherence validation - incoherent weight ratios
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "weight coherence ratio detection".to_string(),
                phase_name: "weight_coherence_ratio_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let config = weight_coherence::WeightCoherenceConfig::default();
                    let detected = weight_coherence::detect_incoherent_weight_ratios(
                        &creature, &records, &config,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        weight_coherence::incoherent_ratios_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #437: Weight coherence validation - near-constant output paths
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "near-constant path detection".to_string(),
                phase_name: "near_constant_path_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let config = weight_coherence::WeightCoherenceConfig::default();
                    let detected =
                        weight_coherence::detect_near_constant_paths(&creature, &records, &config);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        weight_coherence::near_constant_paths_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #437: Weight coherence validation - symmetric weight cancellation
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "symmetric cancellation detection".to_string(),
                phase_name: "symmetric_cancellation_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    let config = weight_coherence::WeightCoherenceConfig::default();
                    let detected = weight_coherence::detect_symmetric_cancellation(
                        &creature, &records, &config,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        weight_coherence::symmetric_cancellation_to_coordinated_candidates(
                            &detected,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #417: Proactive activation function recommendation
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "activation recommendation".to_string(),
                phase_name: "activation_recommendation",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let mut recommendations = Vec::new();
                    for (uuid, squash, _bias) in hidden.iter() {
                        if let Some(neuron_records) = records.iter().find(|(u, _)| u == uuid)
                            && let Some(rec) =
                                activation_recommendation::recommend_activation_function(
                                    &neuron_records.1,
                                    squash,
                                )
                        {
                            recommendations.push(rec);
                        }
                    }
                    if recommendations.is_empty() {
                        return None;
                    }
                    let candidates =
                        activation_recommendation::recommendations_to_coordinated_candidates(
                            &recommendations,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: recommendations.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #543: Activation mismatch detection
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "activation mismatch detection".to_string(),
                phase_name: "activation_mismatch_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected =
                        activation_mismatch::detect_activation_mismatches(&hidden, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        activation_mismatch::activation_mismatch_to_coordinated_candidates(
                            &detected,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #422: Topology-aware network structure analysis
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "topology structure analysis".to_string(),
                phase_name: "topology_structure_analysis",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_all_neurons(&creature);
                    let detected = topology::detect_topology_issues(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        topology::topology_issues_to_coordinated_candidates(&detected, &creature);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #549: Topology diversification for structural jumps
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "topology diversification detection".to_string(),
                phase_name: "topology_diversification_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    let detected =
                        topology_diversification::detect_topology_diversification_candidates(
                            &creature, &records,
                        );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        topology_diversification::topology_diversification_to_coordinated_candidates(
                            &detected, &creature,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #423: Sample-weighted discovery — prioritise high-error samples
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "sample-weighted discovery".to_string(),
                phase_name: "sample_weighted_discovery",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    if records.is_empty() {
                        return None;
                    }
                    let config = sample_weighted::SampleWeightedConfig::default();
                    let detected = sample_weighted::detect_high_error_neurons(&records, &config);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        sample_weighted::high_error_neurons_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #421: Gradient-based synapse adjustment — directional improvement hints
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "gradient-based discovery".to_string(),
                phase_name: "gradient_based_discovery",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    if records.is_empty() {
                        return None;
                    }
                    let detected =
                        gradient_discovery::detect_gradient_candidates(&creature, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        gradient_discovery::gradient_candidates_to_coordinated(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #545: Output squash mismatch detection (local minimum escape)
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "output squash mismatch detection".to_string(),
                phase_name: "output_squash_mismatch_detection",
                detect_fn: Box::new(move || {
                    let output_neurons: Vec<(String, String, f32)> = creature
                        .neurons
                        .iter()
                        .filter(|n| n.neuron_type == "output")
                        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
                        .collect();
                    if output_neurons.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                    let detected = output_squash_mismatch::detect_output_squash_mismatches(
                        &output_neurons,
                        &records,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        output_squash_mismatch::output_squash_mismatch_to_coordinated_candidates(
                            &detected,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #548: Squash + weight rescale detection (coordinated multi-neuron squash exploration)
        {
            let cache = Arc::clone(&shared_cache);
            let hidden = Arc::clone(&hidden_neurons);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "squash weight rescale detection".to_string(),
                phase_name: "squash_weight_rescale_detection",
                detect_fn: Box::new(move || {
                    if hidden.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_hidden(&hidden);
                    let detected = squash_weight_rescale::detect_squash_weight_rescale_candidates(
                        &creature, &hidden, &records,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        squash_weight_rescale::squash_weight_rescale_to_coordinated_candidates(
                            &detected,
                        );
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #545: Error stagnation plateau detection (local minimum escape)
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "error plateau detection".to_string(),
                phase_name: "error_plateau_detection",
                detect_fn: Box::new(move || {
                    let output_neurons: Vec<(String, String, f32)> = creature
                        .neurons
                        .iter()
                        .filter(|n| n.neuron_type == "output")
                        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
                        .collect();
                    if output_neurons.is_empty() {
                        return None;
                    }
                    let records = cache.load_records_for_neuron_types(&creature, &["output"]);
                    let detected = error_plateau::detect_error_plateaus(&output_neurons, &records);
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        error_plateau::error_plateaus_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #550: Weight magnitude reset for stuck synapses (local minimum escape)
        {
            let cache = Arc::clone(&shared_cache);
            let creature = Arc::clone(&creature);
            modules.push(discovery_dispatch::DiscoveryModuleSpec {
                module_name: "weight magnitude reset detection".to_string(),
                phase_name: "weight_magnitude_reset_detection",
                detect_fn: Box::new(move || {
                    let records = cache.load_records_for_all_neurons(&creature);
                    if records.is_empty() {
                        return None;
                    }
                    let detected = weight_magnitude_reset::detect_stuck_synapse_weight_resets(
                        &creature, &records,
                    );
                    if detected.is_empty() {
                        return None;
                    }
                    let candidates =
                        weight_magnitude_reset::stuck_synapses_to_coordinated_candidates(&detected);
                    Some(discovery_dispatch::DiscoveryDetectionResult {
                        detected_count: detected.len(),
                        candidates,
                    })
                }),
            });
        }

        // Issue #419: Dispatch all detection modules in parallel, merge sequentially.
        discovery_dispatch::run_discovery_modules_parallel(syn, modules, max_candidates, diversify);
    }

    // Issue #489: Cross-module candidate deduplication.
    // After all discovery modules have contributed coordinated structural candidates,
    // deduplicate across module boundaries to avoid redundant ablation tests.
    if let Some(syn) = synapse_result.as_mut()
        && !syn.coordinated_structural_candidates.is_empty()
    {
        crate::watchdog::beat("analysis::analyze_all → cross-module deduplication starting");
        let _dedup_timer = PhaseTimer::new("cross_module_deduplication");

        let before_count = syn.coordinated_structural_candidates.len();
        let dedup_result = candidate_clustering::deduplicate_cross_module_candidates(mem::take(
            &mut syn.coordinated_structural_candidates,
        ));
        syn.coordinated_structural_candidates = dedup_result.candidates;

        if dedup_result.duplicates_removed > 0 && utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Cross-module deduplication: removed {} duplicate(s) from {} coordinated candidate(s) → {} remaining",
                dedup_result.duplicates_removed,
                before_count,
                syn.coordinated_structural_candidates.len()
            );
        }

        if dedup_result.conflicts_detected > 0 && utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Cross-module deduplication: {} neuron conflict(s) detected (remove vs modify)",
                dedup_result.conflicts_detected
            );
        }

        // Update metadata to reflect the deduplicated count.
        syn.metadata.candidates_returned = syn.helpful_synapses.len()
            + syn.harmful_synapses.len()
            + syn.coordinated_structural_candidates.len();

        crate::watchdog::beat("analysis::analyze_all → cross-module deduplication finished");
    }

    // Issue #224: Candidate clustering to reduce redundant ablation tests.
    // Groups similar candidates by target neuron, source type, and improvement
    // similarity so the controller can test a representative first and skip
    // redundant tests if it fails.
    if let Some(syn) = synapse_result.as_mut() {
        crate::watchdog::beat("analysis::analyze_all → candidate clustering starting");
        let _clustering_timer = PhaseTimer::new("candidate_clustering");

        // Collect all helpful and harmful synapse candidates as clusterable candidates
        let mut clusterable: Vec<candidate_clustering::ClusterableCandidate> = Vec::new();

        for c in &syn.helpful_synapses {
            clusterable.push(candidate_clustering::ClusterableCandidate {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
                expected_improvement: c.expected_creature_score_gain,
                neuron_type: input
                    .creature
                    .neurons
                    .iter()
                    .find(|n| n.uuid == c.from_neuron_uuid)
                    .map(|n| n.neuron_type.clone())
                    .unwrap_or_else(|| "input".to_string()),
            });
        }

        for c in &syn.harmful_synapses {
            clusterable.push(candidate_clustering::ClusterableCandidate {
                from_neuron_uuid: c.from_neuron_uuid.clone(),
                to_neuron_uuid: c.to_neuron_uuid.clone(),
                expected_improvement: c.expected_creature_score_gain,
                neuron_type: input
                    .creature
                    .neurons
                    .iter()
                    .find(|n| n.uuid == c.from_neuron_uuid)
                    .map(|n| n.neuron_type.clone())
                    .unwrap_or_else(|| "input".to_string()),
            });
        }

        let clusters = candidate_clustering::cluster_candidates(&clusterable);

        if !clusters.is_empty() && utils::verbose_enabled() {
            let total_clustered: usize = clusters.iter().map(|c| c.member_count).sum();
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Candidate clustering: {} cluster(s) covering {} candidate(s) of {} total",
                clusters.len(),
                total_clustered,
                clusterable.len()
            );
        }

        syn.candidate_clusters = clusters;

        crate::watchdog::beat("analysis::analyze_all → candidate clustering finished");
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
        neuron_fingerprints: Some(current_fingerprints),
        fingerprint_cache_hits,
        fingerprint_cache_misses,
    })
}

// Re-export from modules
pub use gpu::GpuAnalyzer;
pub use gpu::GpuAvailabilityResult;
pub use gpu::supports_unified_memory;
pub use neuron::analyze_neurons;
pub use synapse::analyze_synapses;
// Re-export benchmark helper function for use in benches/
pub use synapse::analyze_synapses_with_cache_and_gpu_queue;

// Re-export early termination types (Issue #219)
pub use early_termination::{
    EarlyTerminationConfig, EarlyTerminationDecision, EarlyTerminationResult, SequentialEvaluator,
    check_batch_early_termination,
};

// Re-export cross-validation types (Issue #436)
pub use cross_validation::{
    CrossValidationConfig, CrossValidationResult, FoldResult, PerformanceVariance,
    apply_brittleness_penalty, compute_cross_validation_score,
};

#[cfg(test)]
mod mod_tests;
