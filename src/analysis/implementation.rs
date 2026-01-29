use crate::intern::NeuronIndex;
use crate::types::DiscoverRecord;
use crate::{AnalyzeSynapsesInput, CandidateSynapseJson, SynapseJson};
use anyhow::{anyhow, Result};

// Import shared types from the new module structure
use crate::analysis::shared::{AnalyzeSynapsesResult, TimingScope};

// Import activation functions from the dedicated activation module (Issue #266, #238)
// Note: Activation functions and specs moved to neuron.rs for neuron analysis (Issue #185)
use crate::analysis::activation::get_target_simulation_fn;

// Import memory and platform utilities from dedicated modules (Issue #267)
// Note: cap_gpu_batch_size_by_bytes, check_system_memory_requirements, detect_memory_tier,
// ensure_xdg_runtime_dir, get_memory_info, suppress_mesa_warnings_if_requested, MemoryTier,
// DEFAULT_GPU_BATCH_SIZE, HIGH_PERF_GPU_BATCH_SIZE, LOW_MEMORY_GPU_BATCH_SIZE moved to
// gpu/analyzer.rs (Issue #273)
// get_work_queue_capacity moved to gpu/queue.rs usage (Issue #274)
use crate::analysis::utils::verbose_enabled;

// Import deadline handling and logging utilities from dedicated module (Issue #268)
// calculate_gpu_batch_timeout moved to gpu/queue.rs usage (Issue #274)
use crate::analysis::utils::{
    build_deadline, deadline_passed, log_analysis_start, log_analysis_timeout,
    order_eligible_sources, parse_input_index, shuffle_slice, shuffle_within_top_k, OrderedNeuron,
};

// Import sample data structures from dedicated module (Issue #269)
// Note: compute_source_variance_discount and more moved to neuron.rs (Issue #185)
use crate::analysis::samples::{
    compute_source_std_dev, get_constant_source_threshold, HelpfulSample, NeuronStats, EPSILON,
};

// Import confidence interval calculations (Issue #194)
use crate::analysis::confidence::compute_confidence_metrics;

// Import weight calculation functions from dedicated module (Issue #270)
use crate::analysis::weights::{
    calculate_optimal_outgoing_weight, clamp_weight_update_delta,
    coordinated_structural_activation_delta,
};

// Import diagnostics and rejection tracking from dedicated module (Issue #271)
// Note: NeuronDiagnostics, FocusTargetFilterResult, etc. moved to neuron.rs (Issue #185)
use crate::analysis::diagnostics::{
    compute_impact_scores_for_discounting, require_unique_focus, TargetDiagnostics, TargetMap,
    ThresholdContext,
};

// Import epistatic pair detection module (Issue #202) and synergistic discovery (Issue #189)
use crate::analysis::epistatic::{
    build_source_contribution, detect_epistatic_pairs, detect_synergistic_candidates,
    epistatic_pairs_to_coordinated_candidates, synergistic_to_coordinated_candidates,
    SourceContribution,
};

// Import redundant path pruning module (Issue #164)
use crate::analysis::redundant_path::{
    detect_redundant_paths, redundant_paths_to_coordinated_candidates, ExistingPathContribution,
};

// Import GPU infrastructure from dedicated modules (Issue #272, #273, #274)
use crate::analysis::gpu::{GpuAnalyzer, GpuWorkQueue};

// bytemuck::Zeroable moved to gpu/analyzer.rs (Issue #273)
// crossbeam_channel::{bounded, Receiver, Sender} moved to gpu/queue.rs (Issue #274)
// once_cell::sync::OnceCell moved to cache.rs (Issue #185)
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
// std::thread::{self, JoinHandle} moved to gpu/queue.rs (Issue #274)
// std::time::Duration moved to gpu/queue.rs usage (Issue #274)

// WORKGROUP_SIZE moved to gpu/analyzer.rs (Issue #273)

const MIN_NEURON_SAMPLE_COUNT: usize = 10;

// MIN_NEURON_OUTPUT_STD_DEV moved to crate::analysis::activation module (Issue #238)
// MAX_OUTGOING_WEIGHT moved to crate::analysis::weights module (Issue #270)
// has_sufficient_output_variance moved to crate::analysis::activation module (Issue #238)

// GPU_QUEUE_TIMEOUT_MIN_SECS and GPU_QUEUE_TIMEOUT_MAX_SECS moved to utils/deadline.rs (Issue #268)
// GPU_BUFFER_MAP_TIMEOUT_MARGIN_SECS, GPU_BUFFER_MAP_TIMEOUT_SECS moved to gpu/device.rs (Issue #272)

// GPU_MAX_BATCH_ALLOC_BYTES (256MB cap for GPU batch submissions) moved to gpu/analyzer.rs (Issue #273)

// GPU_INIT_TIMEOUT_SECS moved to gpu/device.rs (Issue #272)

// calculate_gpu_batch_timeout moved to utils/deadline.rs (Issue #268)

// poll_device_until_idle, wait_for_buffer_map, wait_for_buffer_maps_batch
// moved to gpu/device.rs (Issue #272)

// GpuPerformanceTier, detect_unified_memory, detect_gpu_tier
// moved to gpu/device.rs (Issue #272)

// Memory detection functions moved to crate::analysis::utils::memory (Issue #267)

// check_minimum_system_requirements, get_adjusted_batch_size, get_batch_size_override,
// get_batch_size_for_tier, log_gpu_info_once moved to gpu/analyzer.rs (Issue #273)

// Deadline handling, logging, and randomisation utilities moved to utils/deadline.rs (Issue #268)
// All functions now imported from crate::analysis::utils

// create_wgpu_instance_safely moved to gpu/device.rs (Issue #272)

// Types moved to shared.rs - using imports from there
// OrderedNeuron moved to utils/deadline.rs (Issue #268)

// RecordCache moved to cache.rs (Issue #185)
pub(crate) use super::cache::RecordCache;

// Import shared helper functions from synapse module (Issue #185)
// These are used by synapse analysis (neuron analysis is in neuron.rs)
use super::synapse::{
    build_ordered_neurons, build_samples_for_locality_group, compute_synapse_improvement_and_count,
    group_sources_by_locality, truncate_combined_synapse_candidate_sets,
    MIN_GROUP_SIZE_FOR_LOCALITY,
};

// RecordCacheProvider, compute_impact_scores_for_discounting, RejectionReason
// moved to crate::analysis::diagnostics module (Issue #271)

// RejectionDetail, ThresholdContext, TargetDiagnosticEntry, TargetDiagnostics
// moved to crate::analysis::diagnostics module (Issue #271)

// NeuronRejectionDetail, NeuronDiagnosticEntry, NeuronDiagnostics
// moved to crate::analysis::diagnostics module (Issue #271)

// RecordCache implementation moved to cache.rs (Issue #185)

// require_unique_focus moved to crate::analysis::diagnostics module (Issue #271)

// Sample data structures moved to crate::analysis::samples module (Issue #269)
// ReluStats::evaluate() moved to samples.rs (Issue #275)

// Note: ActivationCandidateSpec, ACTIVATION_SPECS, activation functions, and bias helpers
// have been moved to crate::analysis::activation module (Issue #266)

// MIN_WEIGHT_RATIO, coordinated_structural_activation_delta, clamp_weight_update_delta,
// calculate_optimal_outgoing_weight, calculate_optimal_identity_outgoing_and_bias,
// calculate_optimal_bias moved to crate::analysis::weights module (Issue #270)

// HarmfulStats moved to crate::analysis::samples module (Issue #269)

// GpuAnalyzer struct moved to gpu/analyzer.rs (Issue #273)
// Now imported from crate::analysis::gpu

// GpuAvailabilityResult moved to gpu/device.rs (Issue #272)
// Now imported from crate::analysis::gpu

// GpuEvaluator trait and impl GpuEvaluator for GpuAnalyzer moved to gpu/analyzer.rs (Issue #273)
// Now imported from crate::analysis::gpu

// GpuWorkQueue struct and impl moved to gpu/queue.rs (Issue #274)
// Now imported from crate::analysis::gpu
// The following items are now in the queue module:
// - GpuWorkRequest enum (pub(crate))
// - GpuWorkQueue struct
// - impl GpuWorkQueue (new, gpu_thread_loop, evaluate_helpful_batch, etc.)
// - impl Drop for GpuWorkQueue
// - impl GpuEvaluator for GpuWorkQueue
// - GPU_SHUTDOWN_TIMEOUT_SECS constant

// GpuAnalyzer implementation moved to gpu/analyzer.rs (Issue #273)
// The following functions are now in the analyzer module:
// - gpu_is_available()
// - check_gpu_availability()
// - supports_unified_memory()
// - get_adapter_info()
// - new()
// - build_helpful_pipeline()
// - build_harmful_pipeline()
// - build_relu_pipeline()
// - build_activation_pipeline()
// - build_bias_pipeline()
// - evaluate_harmful_batch()
// - evaluate_relu_gpu()
// - evaluate_activation_gpu()
// - evaluate_bias_gpu()
// - evaluate_helpful_batch()
// - merge_batch_results()

// Shader constants moved to gpu/analyzer.rs (Issue #273)

/// Internal implementation of synapse analysis with cache.
/// This is called from the synapse module which owns the public API.
///
/// The `gpu_queue` parameter is mandatory - callers should create it once and reuse it
/// across multiple calls for better performance (avoids ~100ms initialization overhead).
pub(crate) fn analyze_synapses_with_cache_impl(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<GpuWorkQueue>,
) -> Result<AnalyzeSynapsesResult> {
    let ordered_neurons = build_ordered_neurons(&input.creature);
    let order_map: HashMap<String, usize> = ordered_neurons
        .iter()
        .map(|neuron| (neuron.uuid.clone(), neuron.index))
        .collect();

    // Issue #210: Use NeuronIndex to intern UUID strings, reducing memory allocations.
    // Instead of cloning UUID strings for each synapse (72+ bytes per entry for String pairs),
    // we use u32 indices (8 bytes per entry) for ~89% memory reduction.
    let mut neuron_index = NeuronIndex::with_capacity(
        input.creature.neurons.len() + input.creature.input + input.creature.synapses.len() / 10,
    );

    // Pre-intern all neuron UUIDs (inputs + neurons from creature)
    for i in 0..input.creature.input {
        neuron_index.intern(&format!("input-{i}"));
    }
    for neuron in &input.creature.neurons {
        neuron_index.intern(&neuron.uuid);
    }

    // Build existing_synapses using interned indices instead of String clones
    let existing_synapses: HashSet<(u32, u32)> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| {
            (
                neuron_index.intern(&synapse.from_uuid),
                neuron_index.intern(&synapse.to_uuid),
            )
        })
        .collect();

    // Build existing_synapse_weights using interned indices
    let existing_synapse_weights: HashMap<(u32, u32), f32> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| {
            (
                (
                    neuron_index.intern(&synapse.from_uuid),
                    neuron_index.intern(&synapse.to_uuid),
                ),
                synapse.weight,
            )
        })
        .collect();

    // Build synapses_by_target using interned index as key
    let synapses_by_target: HashMap<u32, Vec<SynapseJson>> = input
        .creature
        .synapses
        .iter()
        .map(|synapse| (neuron_index.intern(&synapse.to_uuid), synapse.clone()))
        .fold(HashMap::new(), |mut acc, (key, val)| {
            acc.entry(key).or_default().push(val);
            acc
        });

    // Build a lookup map for neuron squash functions to identify discrete targets
    let neuron_squash_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.squash.clone()))
        .collect();

    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")?;

    // Issue #216: TargetDiagnostics uses DashMap internally for lock-free concurrent access.
    // No Mutex wrapper needed - the struct handles concurrency internally.
    let diagnostics = Arc::new(TargetDiagnostics::new(&unique_focus));

    // GPU timing collector (Issue #195)
    // Only collects timing data when NEAT_AI_DISCOVERY_GPU_TIMING=1 is set
    let timing_collector = Arc::new(super::shared::TimingCollector::new(
        super::utils::gpu_timing_enabled(),
    ));

    let deadline = build_deadline(input.analysis_deadline_ms);
    // Randomise the focus neuron order so that repeated runs with timeouts will
    // eventually cover all neurons. Convert to owned strings, shuffle, then use.
    // STEP/BIPOLAR neurons are now included - both get proper simulation functions
    // that accurately predict output flips when synapse contributions cross the threshold.
    let mut focus_order: Vec<String> = unique_focus.iter().map(|s| (*s).clone()).collect();

    shuffle_slice(&mut focus_order, input.random_seed, "synapse:focus_order");

    // Log analysis start with timeout duration and randomised order
    log_analysis_start(
        "synapse",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    // Track completed focus neurons for timeout logging
    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // v0.1.134: Return ALL positive improvements.
    // NEAT-AI applies the cost-of-growth gate during evaluation.
    let threshold = 0.0;

    // Collect all helpful evaluation work first for batching
    struct HelpfulWork {
        source_uuid: String,
        target_uuid: String,
        samples: Vec<HelpfulSample>,
        /// Existing synapse weight (when the synapse already exists).
        ///
        /// When set, we propose a delta-based weight update rather than adding a new synapse.
        existing_weight: Option<f32>,
    }

    // GPU is always required - TypeScript layer calls check_gpu_available() and skips
    // discovery entirely on machines without GPU. Reaching here without GPU is a bug.
    assert!(
        GpuAnalyzer::gpu_is_available(),
        "Discovery logic called without GPU - check_gpu_available should have prevented this"
    );
    let gpu_used = true;

    let helpful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let harmful_results = Arc::new(Mutex::new(Vec::<CandidateSynapseJson>::new()));
    let coordinated_structural_results = Arc::new(Mutex::new(Vec::<
        crate::CoordinatedStructuralCandidateJson,
    >::new()));
    let helpful_fallback = Arc::new(Mutex::new(Option::<CandidateSynapseJson>::None));
    let analysis_timed_out = Arc::new(Mutex::new(false));

    // Metadata tracking for observability (v0.2.17+)
    // These track whether target_value was available and whether saturation-aware simulation was used
    let metadata_target_value_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_saturation_aware_used = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_seen_any_input_with_records = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let metadata_input_min_with_records = Arc::new(std::sync::atomic::AtomicUsize::new(usize::MAX));
    let metadata_input_max_with_records = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    // Issue #192: Collect error values for error distribution analysis
    // We collect errors from all target neurons to compute aggregate distribution statistics
    let error_values_for_distribution = Arc::new(Mutex::new(Vec::<f32>::new()));

    let focus_order_arc = Arc::new(focus_order);
    let ordered_neurons_arc = Arc::new(ordered_neurons);
    let existing_synapses_arc = Arc::new(existing_synapses);
    let existing_synapse_weights_arc = Arc::new(existing_synapse_weights);
    let synapses_by_target_arc = Arc::new(synapses_by_target);
    let order_map_arc = Arc::new(order_map);
    let neuron_squash_map_arc = Arc::new(neuron_squash_map);
    // Issue #210: Share neuron index for interned UUID lookups across parallel threads
    let neuron_index_arc = Arc::new(neuron_index);

    // Issue #182: Build a set of "used" input neurons (those with at least one outgoing synapse)
    // for the focus_unused_observations feature. Unused inputs will be prioritised in source ordering.
    let used_inputs: HashSet<String> = input
        .creature
        .synapses
        .iter()
        .filter(|s| parse_input_index(&s.from_uuid).is_some())
        .map(|s| s.from_uuid.clone())
        .collect();
    let used_inputs_arc = Arc::new(used_inputs);

    // Map of neuron UUID -> bias, used when folding constant-source synapses into `setBias`.
    // Inputs are not present here (they have no bias).
    let neuron_bias_map: HashMap<String, f32> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.bias))
        .collect();
    let neuron_bias_map_arc = Arc::new(neuron_bias_map);

    // Issue #199: Compute source variance profile for dynamic constant-source threshold.
    // We sample source neurons to compute an average standard deviation, which is used
    // to scale the threshold for folding constant sources into setBias operations.
    // This captures more coordinated candidates in creatures where "constant" is relative.
    let source_std_dev_avg: Option<f32> = {
        // Sample input neurons to compute average std dev
        let mut std_dev_sum = 0.0f64;
        let mut std_dev_count = 0u32;
        let max_samples = input.creature.input.min(50); // Sample up to 50 input neurons

        for input_idx in 0..max_samples {
            let input_uuid = format!("input-{input_idx}");
            if let Ok(records) = cache.get(&input_uuid) {
                if records.len() >= 2 {
                    let std_dev = compute_source_std_dev(&records);
                    if std_dev.is_finite() {
                        std_dev_sum += std_dev as f64;
                        std_dev_count += 1;
                    }
                }
            }
        }

        if std_dev_count > 0 {
            let avg = (std_dev_sum / std_dev_count as f64) as f32;
            if verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Source variance profile: avg_std_dev={avg:.4} (sampled {std_dev_count} sources)"
                );
            }
            Some(avg)
        } else {
            None
        }
    };

    // Issue #178, #199: threshold for folding constant-ish sources into bias adjustments.
    // Now uses dynamic threshold based on source variance profile (Issue #199).
    let constant_source_effect_threshold = get_constant_source_threshold(source_std_dev_avg);

    // Build a comprehensive map of ALL neuron UUIDs to their types
    // This includes: input neurons, and all neurons from creature.neurons (hidden, output, constant)
    // If a UUID is not in this map, it's an invalid UUID (bug)
    let mut neuron_type_map: HashMap<String, String> = HashMap::new();

    // Add input neurons (they're not in creature.neurons, only represented by creature.input count)
    for input_index in 0..input.creature.input {
        neuron_type_map.insert(format!("input-{input_index}"), "input".to_string());
    }

    // Add all neurons from creature.neurons (hidden, output, constant)
    for neuron in &input.creature.neurons {
        neuron_type_map.insert(neuron.uuid.clone(), neuron.neuron_type.clone());
    }

    let neuron_type_map_arc = Arc::new(neuron_type_map);

    // Keep input neuron UUIDs set for quick checks (backwards compatibility)
    let input_neuron_uuids: HashSet<String> = (0..input.creature.input)
        .map(|i| format!("input-{i}"))
        .collect();
    let input_neuron_uuids_arc = Arc::new(input_neuron_uuids);

    // Use the provided GPU work queue.
    // This eliminates the overhead of creating multiple GPU devices (one per thread).
    // All GPU operations are processed by a single dedicated thread, improving utilisation.
    // CRITICAL: The GpuAnalyzer is created INSIDE the GPU thread to avoid wgpu deadlocks.

    // Process each focus neuron in parallel
    focus_order_arc
        .par_iter()
        .try_for_each(|target_uuid| -> Result<()> {
            if *analysis_timed_out.lock().expect("Mutex poisoned") || deadline_passed(&deadline) {
                *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                return Ok(());
            }

            crate::watchdog::beat(format!(
                "synapse analysis → processing target {target_uuid}"
            ));

            // Use the shared GPU work queue instead of creating a new GpuAnalyzer per thread.
            // This eliminates device creation overhead and improves GPU utilisation.
            let gpu = &*gpu_queue;

            let target_records_arc = cache.get(target_uuid.as_str())?;
            if target_records_arc.is_empty() {
                diagnostics.set_target_record_count(target_uuid, 0);
                return Ok(());
            }
            let target_records = target_records_arc.as_ref();
            diagnostics.set_target_record_count(target_uuid, target_records.len());

            // Issue #192: Collect error values for distribution analysis
            // Extract errors from target records and add to shared collection
            {
                let errors: Vec<f32> = target_records
                    .iter()
                    .flat_map(|r| r.errors.iter().filter(|e| e.is_finite()).copied())
                    .collect();
                if !errors.is_empty() {
                    let mut error_vec = error_values_for_distribution
                        .lock()
                        .expect("Mutex poisoned: error_values_for_distribution");
                    error_vec.extend(errors);
                }
            }

            let target_index = match order_map_arc.get(target_uuid.as_str()) {
                Some(index) => *index,
                None => {
                    // Target neuron not found in order map - this indicates a data integrity issue
                    // This should never happen for valid hidden/output neurons
                    if verbose_enabled() {
                        eprintln!(
                            "[NEAT-AI-Discovery][verbose] Target {target_uuid} not found in creature neuron order map (neuron may not exist in creature definition). Skipping."
                        );
                    }
                    return Ok(());
                }
            };

            // Early validation: skip input and constant neurons (they have no upstream sources)
            // Validate target UUID exists in comprehensive neuron type map
            let target_neuron_type = neuron_type_map_arc.get(target_uuid.as_str())
                .ok_or_else(|| anyhow!(
                    "Invalid target neuron UUID '{}': not found in neuron type map. \
                    This indicates a serious data integrity bug. All valid neurons must be in the type map \
                    (input neurons: input-0..input-{}, or neurons from creature.neurons array).",
                    target_uuid,
                    input_neuron_uuids_arc.len().saturating_sub(1)
                ))?;

            let input_count = input_neuron_uuids_arc.len();
            let is_input_neuron = target_neuron_type == "input";
            let is_constant_neuron = target_neuron_type == "constant";

            // Skip actual input/constant neurons (expected - they have no upstream sources)
            // Only skip by UUID check, not by index, to avoid incorrectly skipping hidden neurons
            // that might have been assigned incorrect indices due to ordering bugs
            if is_input_neuron || is_constant_neuron {
                return Ok(());
            }

            // Filter eligible sources: must have index < target_index and not be a constant
            // All neurons should be in the comprehensive neuron_type_map
            let mut eligible_sources: Vec<&OrderedNeuron> = ordered_neurons_arc
                .iter()
                .filter(|neuron| {
                    neuron.index < target_index
                        && {
                            // Look up neuron type - if missing, it's a serious bug
                            match neuron_type_map_arc.get(&neuron.uuid) {
                                Some(neuron_type) => {
                                    // Valid neuron - exclude constants, include everything else (input, hidden, output)
                                    neuron_type != "constant"
                                }
                                None => {
                                    // Invalid UUID - serious data integrity bug
                                    eprintln!(
                                        "[NEAT-AI-Discovery] ERROR: Invalid neuron UUID '{}' found in ordered_neurons. \
                                        Not found in comprehensive neuron type map. This indicates a serious data integrity bug.",
                                        neuron.uuid
                                    );
                                    false // Exclude invalid neurons
                                }
                            }
                        }
                })
                .collect();

            // Track total eligible sources before filtering
            let total_eligible = eligible_sources.len() as u32;

            // For hidden/output neurons with index >= input_count, there should always be at least the input neurons as eligible sources
            // If total_eligible == 0, this indicates a serious bug
            if total_eligible == 0 {
                // This should be impossible - we've already filtered out input/constant neurons
                // Log detailed diagnostics to help debug
                let neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index)
                    .count();
                let constants_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| {
                        n.index < target_index
                            && neuron_type_map_arc
                                .get(&n.uuid)
                                .map(|t| t == "constant")
                                .unwrap_or(false)
                    })
                    .count();
                let input_neurons_before_index = ordered_neurons_arc
                    .iter()
                    .filter(|n| n.index < target_index && input_neuron_uuids_arc.contains(&n.uuid))
                    .count();

                eprintln!(
                    "[NEAT-AI-Discovery] BUG: Target {target_uuid} (type: {target_neuron_type}, index: {target_index}) has no eligible upstream neurons. \
                    creature.input: {input_count}, neurons before target: {neurons_before_index}, constants before target: {constants_before_index}, \
                    input neurons before target: {input_neurons_before_index}. This should not happen for hidden/output neurons with index >= creature.input."
                );

                // Still skip to avoid crashing, but log the bug
                return Ok(());
            }
            // Count how many eligible sources are input neurons
            let input_neuron_count = eligible_sources
                .iter()
                .filter(|neuron| input_neuron_uuids_arc.contains(&neuron.uuid))
                .count() as u32;
            diagnostics.set_total_eligible_sources(target_uuid, total_eligible);
            diagnostics.set_input_neuron_count(target_uuid, input_neuron_count);

            let context = format!("synapse:eligible_sources:{target_uuid}");
            order_eligible_sources(
                &mut eligible_sources,
                input.random_seed,
                &context,
                input.creature.input,
                Some(&*used_inputs_arc),
            );

            // Improved GPU utilisation: Build samples on CPU in parallel, then batch GPU evaluation
            // This avoids the GPU sync overhead of calling build_samples_gpu for each source.
            // The heavy computation is in evaluate_helpful_batch which is properly batched.
            struct SourceWorkResult {
                work: Option<HelpfulWork>,
                had_samples: bool,
                source_uuid: String,
                record_count: usize,
            }

            // Pre-filter sources and collect their records (cache is thread-safe)
            // Track already-connected, load-failure, and empty-record counts separately
            let mut already_connected_count = 0u32;
            let mut load_failure_count = 0u32;
            let mut empty_record_sources: Vec<String> = Vec::new();
            struct ExistingSourceToProcess<'a> {
                source: &'a OrderedNeuron,
                records: Arc<Vec<DiscoverRecord>>,
                old_weight: f32,
            }

            // Note: This vector is for *add-synapse* candidates only.
            // Existing edges are intentionally excluded to preserve the original
            // `NoEligibleSources` diagnostics semantics (tests rely on this).
            let mut sources_to_process: Vec<(&OrderedNeuron, Arc<Vec<DiscoverRecord>>)> =
                Vec::with_capacity(eligible_sources.len());

            // Existing edges are evaluated separately as weight-update candidates.
            let mut existing_sources_to_process: Vec<ExistingSourceToProcess> =
                Vec::with_capacity(eligible_sources.len());

            for source in &eligible_sources {
                if deadline_passed(&deadline) {
                    *analysis_timed_out.lock().expect("Mutex poisoned") = true;
                    break;
                }
                let source_uuid = source.uuid.as_str();

                // Issue #210: Use interned indices for efficient lookup (avoids String allocations)
                let source_idx = neuron_index_arc.get_index(source_uuid);
                let target_idx = neuron_index_arc.get_index(target_uuid.as_str());
                let is_connected = match (source_idx, target_idx) {
                    (Some(s), Some(t)) => existing_synapses_arc.contains(&(s, t)),
                    _ => false, // UUID not interned means it's not in the creature
                };
                if is_connected {
                    already_connected_count += 1;
                };
                let existing_weight = if is_connected {
                    match (source_idx, target_idx) {
                        (Some(s), Some(t)) => existing_synapse_weights_arc.get(&(s, t)).copied(),
                        _ => None,
                    }
                } else {
                    None
                };

                match cache.get(source_uuid) {
                    Ok(records) => {
                        if !records.is_empty() {
                            if let Some(input_index) = parse_input_index(source_uuid) {
                                metadata_seen_any_input_with_records.store(
                                    true,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                // Update min/max atomically (best-effort).
                                let _ = metadata_input_min_with_records.fetch_min(
                                    input_index,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                                let _ = metadata_input_max_with_records.fetch_max(
                                    input_index,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                            if let Some(old_weight) = existing_weight {
                                existing_sources_to_process.push(ExistingSourceToProcess {
                                    source,
                                    records,
                                    old_weight,
                                });
                            } else if !is_connected {
                                sources_to_process.push((source, records));
                            }
                        } else {
                            // Empty records - track for diagnostics
                            empty_record_sources.push(source_uuid.to_string());
                            let is_input_neuron = input_neuron_uuids_arc.contains(source_uuid);
                            // Log non-input neurons with empty records (input neurons are logged as summary below)
                            if verbose_enabled() && !is_input_neuron {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Source {source_uuid} (target {target_uuid}) has no records in parquet file."
                                );
                            }
                        }
                    }
                    Err(err) => {
                        load_failure_count += 1;
                        if verbose_enabled() {
                            eprintln!(
                                "[NEAT-AI-Discovery][verbose] Failed to load records for source {source_uuid} (target {target_uuid}): {err}"
                            );
                        }
                    }
                };
            }

            // Count how many empty record sources are input neurons (helps diagnose parquet data issues)
            let empty_input_neuron_count = empty_record_sources
                .iter()
                .filter(|uuid| input_neuron_uuids_arc.contains(uuid.as_str()))
                .count();
            let empty_non_input_count = empty_record_sources.len() - empty_input_neuron_count;

            // Log summary if many input neurons have empty records (indicates data issue)
            if verbose_enabled() && empty_input_neuron_count > 0 {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: {} of {} input neurons have no records in parquet file (plus {} non-input sources). This may indicate incomplete parquet data.",
                    empty_input_neuron_count,
                    input_neuron_uuids_arc.len(),
                    empty_non_input_count
                );
            }

            // Update diagnostics for already-connected, load failures, and empty records
            // Issue #216: Direct method calls - no lock needed with DashMap-based diagnostics
            if already_connected_count > 0
                || load_failure_count > 0
                || !empty_record_sources.is_empty()
            {
                for _ in 0..already_connected_count {
                    diagnostics.record_already_connected(target_uuid);
                }
                for _ in 0..load_failure_count {
                    diagnostics.record_load_failure(target_uuid);
                }
                // Record diagnostics for sources with empty records (matches old sequential behaviour)
                for source_uuid in &empty_record_sources {
                    diagnostics.record_candidate_attempt(target_uuid, false);
                    diagnostics.record_no_samples(target_uuid, source_uuid, 0);
                }
            }

            // Build samples on CPU (fast hashmap matching, no GPU sync overhead)
            // OPTIMIZATION: Pre-build target map ONCE, reuse for all sources.
            // This avoids rebuilding the HashMap for each of ~1000+ source neurons.
            let target_map = TargetMap::from_records(target_records);
            let target_map_ref = &target_map;

            // ================================================================
            // Coordinated Structural Discovery (Issue #165): noisy vs trusted
            // ================================================================
            //
            // This targets the "simple case" described in the docs:
            // - Two input signals with the same mean and the same starting synapse weight.
            // - One signal is much noisier (higher activation variance).
            //
            // The intended coordinated fix is:
            // - remove the noisy synapse completely
            // - remove the trusted synapse
            // - add the trusted synapse back with a higher weight (typically doubled)
            //
            // This is designed for large-scale feature inputs (eg market data), where variance
            // differences can reflect noise rather than signal.
            let coordinated_candidate = (|| -> Option<crate::CoordinatedStructuralCandidateJson> {
                fn activation_mean_and_variance(records: &[DiscoverRecord]) -> Option<(f32, f32)> {
                    let mut n = 0.0f32;
                    let mut sum = 0.0f32;
                    let mut sum_sq = 0.0f32;
                    for r in records {
                        if r.activation.is_finite() {
                            n += 1.0;
                            sum += r.activation;
                            sum_sq += r.activation * r.activation;
                        }
                    }
                    if n <= 0.0 {
                        return None;
                    }
                    let mean = sum / n;
                    let var = (sum_sq / n) - (mean * mean);
                    Some((mean, var.max(0.0)))
                }

                fn activation_map(records: &[DiscoverRecord]) -> HashMap<u32, f32> {
                    let mut map = HashMap::with_capacity(records.len());
                    for r in records {
                        if r.activation.is_finite() {
                            map.insert(r.obs_index, r.activation);
                        }
                    }
                    map
                }

                // Issue #210: Use interned index for synapses_by_target lookup
                let target_idx_for_synapses = neuron_index_arc.get_index(target_uuid.as_str())?;
                let existing = synapses_by_target_arc.get(&target_idx_for_synapses)?;

                #[derive(Clone)]
                struct IncomingInput {
                    from_uuid: String,
                    weight: f32,
                    mean: f32,
                    var: f32,
                }

                let mut incoming_inputs: Vec<IncomingInput> = Vec::new();
                for syn in existing.iter() {
                    if !syn.from_uuid.starts_with("input-") {
                        continue;
                    }
                    let Ok(from_records_arc) = cache.get(&syn.from_uuid) else {
                        continue;
                    };
                    if from_records_arc.is_empty() {
                        continue;
                    }
                    let Some((mean, var)) = activation_mean_and_variance(from_records_arc.as_ref())
                    else {
                        continue;
                    };
                    incoming_inputs.push(IncomingInput {
                        from_uuid: syn.from_uuid.clone(),
                        weight: syn.weight,
                        mean,
                        var,
                    });
                }

                if incoming_inputs.len() < 2 || target_map_ref.map.is_empty() {
                    None
                } else {
                    // Strict matching for the simple-case test: same weights and same means.
                    const WEIGHT_EPS: f32 = 1e-6;
                    const MEAN_EPS: f32 = 1e-3;
                    const MIN_VAR_RATIO: f32 = 10.0;

                    let target_squash = neuron_squash_map_arc
                        .get(target_uuid.as_str())
                        .map(|s| s.as_str());

                    let mut best: Option<(IncomingInput, IncomingInput, f32)> = None; // (noisy, trusted, gain)

                    for i in 0..incoming_inputs.len() {
                        for j in (i + 1)..incoming_inputs.len() {
                            let a = incoming_inputs[i].clone();
                            let b = incoming_inputs[j].clone();

                            if (a.weight - b.weight).abs() > WEIGHT_EPS {
                                continue;
                            }
                            if (a.mean - b.mean).abs() > MEAN_EPS {
                                continue;
                            }

                            let (noisy, trusted) = if a.var >= b.var { (a, b) } else { (b, a) };
                            let ratio = noisy.var / trusted.var.max(EPSILON);
                            if ratio < MIN_VAR_RATIO {
                                continue;
                            }

                            let Ok(noisy_records_arc) = cache.get(&noisy.from_uuid) else {
                                continue;
                            };
                            let Ok(trusted_records_arc) = cache.get(&trusted.from_uuid) else {
                                continue;
                            };

                            let noisy_map = activation_map(noisy_records_arc.as_ref());
                            let trusted_map = activation_map(trusted_records_arc.as_ref());

                            let mut delta_samples: Vec<HelpfulSample> =
                                Vec::with_capacity(target_map_ref.map.len());
                            for (obs_index, target) in target_map_ref.map.iter() {
                                let Some(noisy_act) = noisy_map.get(obs_index) else {
                                    continue;
                                };
                                let Some(trusted_act) = trusted_map.get(obs_index) else {
                                    continue;
                                };
                                let Some(activation) = coordinated_structural_activation_delta(
                                    *trusted_act,
                                    *noisy_act,
                                    noisy.weight,
                                    trusted.weight,
                                ) else {
                                    continue;
                                };
                                if !activation.is_finite() || !target.avg_error.is_finite() {
                                    continue;
                                }
                                delta_samples.push(HelpfulSample {
                                    activation,
                                    avg_error: target.avg_error,
                                    target_value: target.value,
                                    target_activation: Some(target.activation),
                                });
                            }

                            if delta_samples.is_empty() {
                                continue;
                            }

                            let baseline_sq: f32 = delta_samples
                                .iter()
                                .map(|s| s.avg_error * s.avg_error)
                                .sum();

                            // Move the noisy weight onto the trusted input:
                            // Δoutput = w_noisy * (trusted - noisy)
                            let moved_weight = noisy.weight;
                            let (improvement, _, _, _) = compute_synapse_improvement_and_count(
                                delta_samples.as_slice(),
                                moved_weight,
                                baseline_sq,
                                target_squash,
                            );

                            if improvement <= 0.0 {
                                continue;
                            }

                            match &best {
                                Some((_, _, best_gain)) if *best_gain >= improvement => {}
                                _ => best = Some((noisy, trusted, improvement)),
                            }
                        }
                    }

                    best.map(|(noisy, trusted, gain)| {
                        let new_weight = trusted.weight + noisy.weight;
                        crate::CoordinatedStructuralCandidateJson {
                            operations: vec![
                                crate::CoordinatedStructuralOpJson::RemoveSynapse {
                                    from_neuron_uuid: noisy.from_uuid,
                                    to_neuron_uuid: target_uuid.to_string(),
                                },
                                crate::CoordinatedStructuralOpJson::RemoveSynapse {
                                    from_neuron_uuid: trusted.from_uuid.clone(),
                                    to_neuron_uuid: target_uuid.to_string(),
                                },
                                crate::CoordinatedStructuralOpJson::AddSynapse {
                                    from_neuron_uuid: trusted.from_uuid,
                                    to_neuron_uuid: target_uuid.to_string(),
                                    weight: new_weight,
                                },
                            ],
                            expected_creature_score_gain: gain,
                            comment: Some(
                                "Coordinated: prune noisy input (high variance), strengthen trusted input"
                                    .to_string(),
                            ),
                        }
                    })
                }
            })();

            if let Some(candidate) = coordinated_candidate {
                let mut results = coordinated_structural_results
                    .lock()
                    .expect("Mutex poisoned: coordinated_structural_results");
                results.push(candidate);
            }

            // Even if target_map is empty, we continue to record diagnostics
            // about what sources were evaluated (important for debugging).
            //
            // Issue #221: Sample Locality Optimisation
            // Group sources by obs_index overlap to reduce redundant sample building.
            // Sources with ≥80% obs_index overlap share sample building in a single pass.
            let source_results: Vec<SourceWorkResult> = {
                let _timing = TimingScope::sample_building(&timing_collector);

                // Group sources by sample locality
                let locality_groups = group_sources_by_locality(&sources_to_process);

                // Log locality grouping stats if verbose
                if verbose_enabled() && sources_to_process.len() >= MIN_GROUP_SIZE_FOR_LOCALITY {
                    let group_sizes: Vec<usize> = locality_groups.iter().map(|g| g.sources.len()).collect();
                    let max_group = group_sizes.iter().max().copied().unwrap_or(0);
                    let avg_group = if !group_sizes.is_empty() {
                        group_sizes.iter().sum::<usize>() as f32 / group_sizes.len() as f32
                    } else {
                        0.0
                    };
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {}: {} sources grouped into {} locality groups (max={}, avg={:.1})",
                        target_uuid,
                        sources_to_process.len(),
                        locality_groups.len(),
                        max_group,
                        avg_group
                    );
                }

                // Build samples for each group (groups with multiple sources use batched building)
                locality_groups
                    .par_iter()
                    .flat_map(|group| {
                        let group_results = build_samples_for_locality_group(group, target_map_ref);
                        group_results
                            .into_iter()
                            .map(|(source_uuid, samples, record_count)| {
                                let had_samples = !samples.is_empty();
                                let work = if had_samples {
                                    Some(HelpfulWork {
                                        source_uuid: source_uuid.clone(),
                                        target_uuid: target_uuid.to_string(),
                                        samples,
                                        existing_weight: None,
                                    })
                                } else {
                                    None
                                };
                                SourceWorkResult {
                                    work,
                                    had_samples,
                                    source_uuid,
                                    record_count,
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect()
            };

            // Extract work batch and batch diagnostics updates
            let mut helpful_work_batch: Vec<HelpfulWork> = Vec::new();
            let mut diagnostics_updates: Vec<(String, String, bool, usize)> = Vec::new();

            for result in source_results {
                if let Some(work) = result.work {
                    helpful_work_batch.push(work);
                }
                diagnostics_updates.push((
                    target_uuid.to_string(),
                    result.source_uuid,
                    result.had_samples,
                    result.record_count,
                ));
            }

            // Apply all diagnostics updates directly (no lock needed with DashMap - Issue #216)
            for (target, source, had_samples, record_count) in diagnostics_updates {
                diagnostics.record_candidate_attempt(&target, had_samples);
                if !had_samples {
                    diagnostics.record_no_samples(&target, &source, record_count);
                }
            }

            // Append existing edges for weight-update evaluation.
            //
            // Important: we do NOT record these as "candidate attempts" in TargetDiagnostics because
            // `no_candidate_reasons` is reporting add-synapse eligibility (existing edges are not
            // eligible for add-synapse). This preserves the historical semantics and unit tests.
            //
            // Issue #164: Also collect ExistingPathContribution for redundant path detection.
            let mut existing_path_contributions: Vec<ExistingPathContribution> = Vec::new();
            if !existing_sources_to_process.is_empty() {
                let existing_work: Vec<HelpfulWork> = existing_sources_to_process
                    .par_iter()
                    .filter_map(|item| {
                        let source_uuid = item.source.uuid.as_str();
                        let from_records = item.records.as_ref();
                        let samples = target_map_ref.build_samples_from(from_records);
                        if samples.is_empty() {
                            return None;
                        }
                        Some(HelpfulWork {
                            source_uuid: source_uuid.to_string(),
                            target_uuid: target_uuid.to_string(),
                            samples,
                            existing_weight: Some(item.old_weight),
                        })
                    })
                    .collect();

                // Issue #164: Collect existing path contributions for redundant path detection
                for work in &existing_work {
                    existing_path_contributions.push(ExistingPathContribution {
                        source_uuid: work.source_uuid.clone(),
                        existing_weight: work.existing_weight.unwrap_or(0.0),
                        samples: work.samples.clone(),
                    });
                }

                helpful_work_batch.extend(existing_work);
            }

            // Process helpful work in batches for better GPU utilization
            // Vertical timeout: Complete all GPU batch processing for the current focus neuron
            if !helpful_work_batch.is_empty() {
                // Clone samples for the GPU queue (queue takes ownership)
                let helpful_samples: Vec<Vec<HelpfulSample>> = helpful_work_batch
                    .iter()
                    .map(|w| w.samples.clone())
                    .collect();

                // Track metadata: check if any samples have target_value data
                // This is used to determine if saturation-aware simulation was possible.
                for samples in &helpful_samples {
                    if samples.iter().any(|s| s.target_value.is_some()) {
                        metadata_target_value_seen.store(true, std::sync::atomic::Ordering::Relaxed);
                        break;
                    }
                }

                let helpful_stats_batch = {
                    let _timing = TimingScope::shader(&timing_collector, "helpful");
                    gpu.evaluate_helpful_batch(helpful_samples, &deadline)?
                };

                // Process results - collect all updates first, then apply in batches (reduces mutex contention)
                let mut candidates_to_add = Vec::new();
                let mut coordinated_to_add = Vec::new();
                let mut diagnostics_zero_improvements = Vec::new();
                let mut diagnostics_below_threshold = Vec::new();
                let mut diagnostics_selected = Vec::new();

                // Issue #202: Track source contributions for epistatic pair detection
                let mut source_contributions: Vec<SourceContribution> = Vec::new();

                {
                    let _timing = TimingScope::result_processing(&timing_collector);
                    for (work, stats) in helpful_work_batch.iter().zip(helpful_stats_batch.iter()) {
                    let positive_is_better = stats.positive_count >= stats.negative_count;
                    let gpu_improved_count = if positive_is_better {
                        stats.positive_count
                    } else {
                        stats.negative_count
                    };
                    if gpu_improved_count == 0 {
                        diagnostics_zero_improvements.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            work.samples.len(),
                            stats.positive_count,
                            stats.negative_count,
                        ));
                        continue;
                    }

                    let total_count = work.samples.len() as u32;
                    if total_count == 0 {
                        continue;
                    }

                    // Use shared weight calculation (synapse = direct connection, so incoming_weight = 1.0)
                    // The shared function ensures consistent weight calculation across synapse and neuron analysis
                    let weight = match calculate_optimal_outgoing_weight(
                        stats.error_activation_sum,
                        stats.activation_sq_sum,
                        1.0, // Synapses are direct connections, no intermediate neuron
                    ) {
                        Some(w) => w,
                        None => continue, // Skip if weight is invalid
                    };

                    // Get target's squash function for saturation-aware improvement calculation.
                    // For saturating activations (HARD_TANH, TANH, LOGISTIC, etc.), the linear model
                    // overpredicts improvement near saturation. Using the actual activation function
                    // gives accurate predictions that match real-world results.
                    let target_squash = neuron_squash_map_arc
                        .get(&work.target_uuid)
                        .map(|s| s.as_str());

                    // Track metadata: check if saturation-aware simulation is used for this candidate.
                    // get_target_simulation_fn returns Some when:
                    // 1. The target squash is a supported saturating activation, AND
                    // 2. All samples have target_value/target_activation data
                    if get_target_simulation_fn(&work.samples, target_squash).is_some() {
                        metadata_saturation_aware_used.store(true, std::sync::atomic::Ordering::Relaxed);
                    }

                    // Compute expected improvement using the linear error model.
                    // This works for ALL squash types because we're measuring actual errors
                    // from recordings, not predicting theoretical errors. The correlation
                    // between source activation and target error determines improvement.
                    // Issue #128: This is neuron-level improvement - impact discounting converts to creature-level.
                    let baseline_error_sq = stats.error_sq_sum;

                    // IMPORTANT (3-Jan-2026):
                    // For weight updates, we must compute expected improvement using the *effective*
                    // (clamped) delta weight, not the proposed delta weight. Otherwise, expected gains
                    // are overstated and candidates are mis-prioritised.
                    let (applied_weight, neuron_error_improvement, improved_count, worsened_count) =
                        if let Some(old_weight) = work.existing_weight {
                            let Some((_new_weight, delta_weight)) =
                                clamp_weight_update_delta(old_weight, weight)
                            else {
                                continue;
                            };
                            let (improvement, improved, worsened, _) =
                                compute_synapse_improvement_and_count(
                                    &work.samples,
                                    delta_weight,
                                    baseline_error_sq,
                                    target_squash,
                                );
                            (delta_weight, improvement, improved, worsened)
                        } else {
                            let (improvement, improved, worsened, _) =
                                compute_synapse_improvement_and_count(
                                    &work.samples,
                                    weight,
                                    baseline_error_sq,
                                    target_squash,
                                );
                            (weight, improvement, improved, worsened)
                        };

                    // Issue #202: Track source contribution for epistatic pair detection
                    // Collect ALL sources (including non-positive improvements) because
                    // epistatic pairs may have low individual improvements but high combined
                    if work.existing_weight.is_none() {
                        source_contributions.push(build_source_contribution(
                            &work.source_uuid,
                            work.samples.clone(),
                            stats.clone(),
                            applied_weight,
                            neuron_error_improvement,
                        ));
                    }

                    // Accept all positive improvements as candidates (not just those above threshold)
                    // Only reject if improvement is non-positive (<= 0.0)
                    if neuron_error_improvement <= 0.0 {
                        // Skip non-positive improvements
                        continue;
                    }

                    // If positive but below threshold, still accept as candidate but log for diagnostics
                    if neuron_error_improvement <= threshold {
                        diagnostics_below_threshold.push((
                            work.target_uuid.clone(),
                            work.source_uuid.clone(),
                            ThresholdContext {
                                sample_count: work.samples.len(),
                                expected_improvement: neuron_error_improvement,
                                threshold,
                                improved_count,
                                worsened_count,
                                weight: applied_weight,
                            },
                        ));
                    }

                    let target_stats = cache
                        .get(&work.target_uuid)
                        .ok()
                        .and_then(|records| NeuronStats::from_records(records.as_ref()))
                        .map(|s| s.to_json());
                    if let Some(old_weight) = work.existing_weight {
                        // Issue #180 (9-Jan-2026): Weight update using setWeight operation.
                        // Previously used remove+add pattern; now use a single setWeight op
                        // for simplicity and directness.
                        let Some((new_weight, delta_weight)) =
                            clamp_weight_update_delta(old_weight, weight)
                        else {
                            continue;
                        };
                        // NOTE: We keep the computed improvement based on the effective (clamped)
                        // delta (`delta_weight`) but apply the absolute `new_weight` in the op.
                        coordinated_to_add.push(crate::CoordinatedStructuralCandidateJson {
                            operations: vec![crate::CoordinatedStructuralOpJson::SetWeight {
                                from_neuron_uuid: work.source_uuid.clone(),
                                to_neuron_uuid: work.target_uuid.clone(),
                                weight: new_weight,
                            }],
                            expected_creature_score_gain: neuron_error_improvement,
                            comment: Some(format!(
                                "Adjust synapse weight: old={old_weight:.6}, new={new_weight:.6}, delta={delta_weight:.6}"
                            )),
                        });
                    } else {
                        diagnostics_selected.push(work.target_uuid.clone());
                        // Issue #178 (7-Jan-2026): If the source activation is constant/near-constant,
                        // an add-synapse behaves like a bias shift on the target. Prefer `setBias`
                        // to avoid paying complexity cost for what is effectively a constant offset.
                        if let Some(threshold) = constant_source_effect_threshold {
                            let mut act_min = f32::INFINITY;
                            let mut act_max = f32::NEG_INFINITY;
                            let mut act_sum = 0.0f64;
                            let mut act_count: u32 = 0;
                            for s in &work.samples {
                                if s.activation.is_finite() {
                                    act_min = act_min.min(s.activation);
                                    act_max = act_max.max(s.activation);
                                    act_sum += s.activation as f64;
                                    act_count += 1;
                                }
                            }

                            if act_count > 0 {
                                let mean_activation = (act_sum / act_count as f64) as f32;
                                let activation_range = (act_max - act_min).abs();
                                let effect_range = weight.abs() * activation_range;

                                if mean_activation.is_finite()
                                    && activation_range.is_finite()
                                    && effect_range.is_finite()
                                    && effect_range <= threshold
                                {
                                    let old_bias = neuron_bias_map_arc
                                        .get(&work.target_uuid)
                                        .copied()
                                        .unwrap_or(0.0);
                                    let new_bias = old_bias + (weight * mean_activation);
                                    if new_bias.is_finite() {
                                        coordinated_to_add.push(crate::CoordinatedStructuralCandidateJson {
                                            operations: vec![crate::CoordinatedStructuralOpJson::SetBias {
                                                neuron_uuid: work.target_uuid.clone(),
                                                bias: new_bias,
                                            }],
                                            expected_creature_score_gain: neuron_error_improvement,
                                            comment: Some(format!(
                                                "Fold constant source into setBias: old_bias={old_bias:.6}, new_bias={new_bias:.6}, weight={weight:.6}, mean_act={mean_activation:.6}, act_range={activation_range:.6e}, effect_range={effect_range:.6e}"
                                            )),
                                        });
                                        continue;
                                    }
                                }
                            }
                        }

                        // Issue #128: Use creature-level metrics (impact discounting applied later)
                        // Issue #194: Compute confidence metrics for this prediction
                        let confidence_metrics = compute_confidence_metrics(
                            &work.samples,
                            neuron_error_improvement,
                            None, // R² not available for synapse candidates
                        );
                        candidates_to_add.push(CandidateSynapseJson {
                            from_neuron_uuid: work.source_uuid.clone(),
                            to_neuron_uuid: work.target_uuid.clone(),
                            from_neuron_index: None,
                            to_neuron_index: None,
                            weight,
                            target_neuron_impact: 1.0,
                            expected_creature_error_reduction: neuron_error_improvement,
                            expected_creature_score_gain: neuron_error_improvement,
                            improved_count,
                            total_count,
                            target_neuron_stats: target_stats,
                            outlier_reduction_info: None, // Set during outlier analysis pass if enabled (Issue #192)
                            prediction_confidence: confidence_metrics.prediction_confidence,
                            expected_score_gain_confidence_interval: confidence_metrics.expected_score_gain_confidence_interval,
                        });
                    }
                    } // End timing scope for result processing
                }

                // Apply all diagnostics updates directly (no lock needed with DashMap - Issue #216)
                for (target, source, sample_count, pos, neg) in diagnostics_zero_improvements {
                    diagnostics.record_zero_improvement(&target, &source, sample_count, pos, neg);
                }
                for (target, source, context) in diagnostics_below_threshold {
                    diagnostics.record_below_threshold(&target, &source, context);
                }
                for target in diagnostics_selected {
                    diagnostics.mark_candidate_selected(&target);
                }
                if !candidates_to_add.is_empty() {
                    let mut results = helpful_results
                        .lock()
                        .expect("Mutex poisoned: helpful_results");
                    results.extend(candidates_to_add);
                }
                if !coordinated_to_add.is_empty() {
                    let mut results = coordinated_structural_results
                        .lock()
                        .expect("Mutex poisoned: coordinated_structural_results");
                    results.extend(coordinated_to_add);
                }

                // Issue #202: Detect epistatic neuron pairs for this target
                // Epistatic pairs are sources where neither improves individually, but
                // both together could improve the target due to complementary patterns.
                if verbose_enabled() {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] Target {target_uuid}: collected {} source contributions for epistatic detection",
                        source_contributions.len()
                    );
                }
                if source_contributions.len() >= 2 {
                    // Get target neuron impact for discounting
                    let target_is_output = neuron_type_map_arc
                        .get(target_uuid.as_str())
                        .map(|t| t == "output")
                        .unwrap_or(false);
                    let target_impact = if target_is_output {
                        1.0
                    } else {
                        // Use a default impact for hidden neurons (will be recalculated later)
                        0.5
                    };

                    let epistatic_pairs = detect_epistatic_pairs(
                        target_uuid.as_str(),
                        &source_contributions,
                        target_impact,
                    );

                    if !epistatic_pairs.is_empty() {
                        let epistatic_candidates = epistatic_pairs_to_coordinated_candidates(&epistatic_pairs);
                        if !epistatic_candidates.is_empty() {
                            let mut results = coordinated_structural_results
                                .lock()
                                .expect("Mutex poisoned: coordinated_structural_results");
                            results.extend(epistatic_candidates);

                            if verbose_enabled() {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: found {} epistatic pair(s)",
                                    epistatic_pairs.len()
                                );
                            }
                        }
                    }

                    // Issue #189: Detect synergistic candidates via residual analysis
                    // This detects XOR-like patterns where:
                    // - Neither source alone provides strong improvement
                    // - Together they reduce error better than either alone
                    let synergistic_candidates = detect_synergistic_candidates(
                        target_uuid.as_str(),
                        &source_contributions,
                        target_impact,
                    );

                    if !synergistic_candidates.is_empty() {
                        let synergistic_coordinated = synergistic_to_coordinated_candidates(&synergistic_candidates);
                        if !synergistic_coordinated.is_empty() {
                            let mut results = coordinated_structural_results
                                .lock()
                                .expect("Mutex poisoned: coordinated_structural_results");
                            results.extend(synergistic_coordinated);

                            if verbose_enabled() {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: found {} synergistic candidate(s)",
                                    synergistic_candidates.len()
                                );
                            }
                        }
                    }
                }

                // Issue #164: Detect redundant paths feeding the same target.
                // Two existing synapses with highly correlated activations are redundant –
                // prune the weaker path and renormalise the survivor's weight.
                if existing_path_contributions.len() >= 2 {
                    let target_is_output = neuron_type_map_arc
                        .get(target_uuid.as_str())
                        .map(|t| t == "output")
                        .unwrap_or(false);
                    let target_impact = if target_is_output {
                        1.0
                    } else {
                        0.5
                    };

                    let redundant_paths = detect_redundant_paths(
                        target_uuid.as_str(),
                        &existing_path_contributions,
                        target_impact,
                    );

                    if !redundant_paths.is_empty() {
                        let redundant_coordinated =
                            redundant_paths_to_coordinated_candidates(&redundant_paths);
                        if !redundant_coordinated.is_empty() {
                            let mut results = coordinated_structural_results
                                .lock()
                                .expect("Mutex poisoned: coordinated_structural_results");
                            results.extend(redundant_coordinated);

                            if verbose_enabled() {
                                eprintln!(
                                    "[NEAT-AI-Discovery][verbose] Target {target_uuid}: found {} redundant path(s) for pruning",
                                    redundant_paths.len()
                                );
                            }
                        }
                    }
                }
            }

            // Process harmful synapses for this target - BATCHED GPU evaluation for better utilisation
            // Vertical timeout: Complete all harmful synapse processing for the current focus neuron
            if !*analysis_timed_out.lock().expect("Mutex poisoned") {
                // Issue #210: Use interned index for synapses_by_target lookup
                let target_idx_for_harmful = neuron_index_arc.get_index(target_uuid.as_str());
                if let Some(existing) = target_idx_for_harmful.and_then(|idx| synapses_by_target_arc.get(&idx)) {
                    // Phase 1: Build all samples on CPU (fast, parallel-friendly)
                    // Reuse the target_map we already built for helpful synapse processing
                    struct HarmfulWork {
                        synapse: SynapseJson,
                        samples: Vec<HelpfulSample>,
                    }
                    let mut harmful_work: Vec<HarmfulWork> = Vec::with_capacity(existing.len());

                    for synapse in existing {
                        let from_records_arc = match cache.get(&synapse.from_uuid) {
                            Ok(records) => records,
                            Err(_) => continue,
                        };
                        if from_records_arc.is_empty() {
                            continue;
                        }
                        let from_records = from_records_arc.as_ref();
                        // Use pre-built target map - avoids rebuilding HashMap for each synapse
                        let samples = target_map_ref.build_samples_from(from_records);
                        if samples.is_empty() {
                            continue;
                        }

                        harmful_work.push(HarmfulWork {
                            synapse: synapse.clone(),
                            samples,
                        });
                    }

                    // Phase 2: Batch GPU evaluation (single submission for all synapses)
                    if !harmful_work.is_empty() {
                        // Clone samples for the GPU queue (queue takes ownership)
                        let batch_input: Vec<(Vec<HelpfulSample>, f32)> = harmful_work
                            .iter()
                            .map(|w| (w.samples.clone(), w.synapse.weight))
                            .collect();

                        let batch_stats = {
                            let _timing = TimingScope::shader(&timing_collector, "harmful");
                            gpu.evaluate_harmful_batch(batch_input, &deadline)?
                        };

                        // Phase 3: Process results
                        let mut harmful_candidates = Vec::with_capacity(batch_stats.len());
                        let target_stats = cache
                            .get(target_uuid.as_str())
                            .ok()
                            .and_then(|records| NeuronStats::from_records(records.as_ref()))
                            .map(|s| s.to_json());

                        for (work, stats) in harmful_work.iter().zip(batch_stats.iter()) {
                            let total_count = work.samples.len() as u32;
                            if total_count == 0 {
                                continue;
                            }

                            // Issue #128: This is neuron-level - impact discounting converts to creature-level
                            let neuron_error_improvement = (stats.harmful_count as f32
                                - stats.helpful_count as f32)
                                / total_count as f32;

                            // Issue #128: Use creature-level metrics (impact discounting applied later)
                            // Issue #194: Compute confidence metrics for this prediction
                            let confidence_metrics = compute_confidence_metrics(
                                &work.samples,
                                neuron_error_improvement,
                                None, // R² not available for synapse candidates
                            );
                            harmful_candidates.push(CandidateSynapseJson {
                                from_neuron_uuid: work.synapse.from_uuid.clone(),
                                to_neuron_uuid: work.synapse.to_uuid.clone(),
                                from_neuron_index: None,
                                to_neuron_index: None,
                                weight: work.synapse.weight,
                                target_neuron_impact: 1.0,
                                expected_creature_error_reduction: neuron_error_improvement,
                                expected_creature_score_gain: neuron_error_improvement,
                                improved_count: stats.harmful_count,
                                total_count,
                                target_neuron_stats: target_stats.clone(),
                                outlier_reduction_info: None, // Set during outlier analysis pass if enabled (Issue #192)
                                prediction_confidence: confidence_metrics.prediction_confidence,
                                expected_score_gain_confidence_interval: confidence_metrics.expected_score_gain_confidence_interval,
                            });
                        }

                        // Batch push all harmful candidates (single lock)
                        if !harmful_candidates.is_empty() {
                            let mut results = harmful_results
                                .lock()
                                .expect("Mutex poisoned: harmful_results");
                            results.extend(harmful_candidates);
                        }
                    }
                }
            }

            // Track completion of this focus neuron for timeout reporting.
            let completed =
                completed_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "synapse analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(())
        })?;

    let analysis_timed_out = *analysis_timed_out.lock().expect("Mutex poisoned");
    let mut helpful_results = helpful_results.lock().expect("Mutex poisoned").clone();
    let mut harmful_results = harmful_results.lock().expect("Mutex poisoned").clone();
    let mut coordinated_structural_results = coordinated_structural_results
        .lock()
        .expect("Mutex poisoned")
        .clone();
    let mut helpful_fallback = helpful_fallback.lock().expect("Mutex poisoned").take();

    // Log timeout with completion stats (always visible, not just verbose)
    if analysis_timed_out {
        let completed = completed_count.load(std::sync::atomic::Ordering::Relaxed);
        log_analysis_timeout("synapse", completed, total_focus_count);
    }

    if helpful_results.is_empty() {
        if let Some(candidate) = helpful_fallback.take() {
            // Issue #216: Direct method call - no lock needed with DashMap-based diagnostics
            diagnostics.mark_candidate_selected(&candidate.to_neuron_uuid);
            helpful_results.push(candidate);
        }
    }

    // Coordinated structural discovery (7-Jan-2026): collapse a simple hidden neuron into a single synapse.
    //
    // This is the "reverse" of synapse→neuron insertion: when a hidden neuron forms a simple 1-in/1-out
    // chain (a → h → b), we can propose removing that neuron and replacing the chain with a direct
    // synapse (a → b). This must be applied atomically, so it is emitted as a coordinated-structural
    // candidate group.
    //
    // Initial scope: only hidden neurons with exactly one incoming and one outgoing synapse.
    // This keeps the candidate safe and deterministic; broader graph rewrites can be added later.
    {
        // Build incoming/outgoing synapse lists per neuron.
        let mut incoming: HashMap<String, Vec<SynapseJson>> = HashMap::new();
        let mut outgoing: HashMap<String, Vec<SynapseJson>> = HashMap::new();
        for s in &input.creature.synapses {
            incoming
                .entry(s.to_uuid.clone())
                .or_default()
                .push(s.clone());
            outgoing
                .entry(s.from_uuid.clone())
                .or_default()
                .push(s.clone());
        }

        // Quick neuron-type lookup (only creature.neurons; inputs are not here).
        let neuron_type_map_local: HashMap<String, String> = input
            .creature
            .neurons
            .iter()
            .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
            .collect();

        // Precompute existing direct synapses so we don't propose duplicates.
        let mut existing_edges: HashSet<(String, String)> = HashSet::new();
        for s in &input.creature.synapses {
            existing_edges.insert((s.from_uuid.clone(), s.to_uuid.clone()));
        }

        for neuron in &input.creature.neurons {
            if neuron.neuron_type != "hidden" {
                continue;
            }
            let h = neuron.uuid.as_str();
            let Some(ins) = incoming.get(h) else { continue };
            let Some(outs) = outgoing.get(h) else {
                continue;
            };
            if ins.len() != 1 || outs.len() != 1 {
                continue;
            }

            let a_syn = &ins[0];
            let b_syn = &outs[0];
            let a = a_syn.from_uuid.as_str();
            let b = b_syn.to_uuid.as_str();

            // Skip degenerate / non-actionable cases.
            if a == b || a == h || b == h {
                continue;
            }
            if existing_edges.contains(&(a.to_string(), b.to_string())) {
                // A direct synapse already exists; collapsing would need additional ops (future work).
                continue;
            }

            // Ensure the target exists (either input-* or a neuron) so the op is not stale.
            let target_is_known = b.starts_with("input-") || neuron_type_map_local.contains_key(b);
            if !target_is_known {
                continue;
            }

            // Build samples: correlate a's activation to b's adjusted error after removing h→b.
            let Ok(a_records) = cache.get(a) else {
                continue;
            };
            let Ok(h_records) = cache.get(h) else {
                continue;
            };
            let Ok(b_records) = cache.get(b) else {
                continue;
            };
            if a_records.is_empty() || h_records.is_empty() || b_records.is_empty() {
                continue;
            }

            let target_map_b = TargetMap::from_records(b_records.as_ref());
            if target_map_b.map.is_empty() {
                continue;
            }
            let build_act_map = |records: &[DiscoverRecord]| -> HashMap<u32, f32> {
                let mut map: HashMap<u32, f32> = HashMap::with_capacity(records.len());
                for r in records {
                    if r.activation.is_finite() {
                        map.insert(r.obs_index, r.activation);
                    }
                }
                map
            };
            let a_map = build_act_map(a_records.as_ref());
            let h_map = build_act_map(h_records.as_ref());

            let mut samples: Vec<HelpfulSample> = Vec::with_capacity(target_map_b.map.len());
            for (obs_index, target) in target_map_b.map.iter() {
                let Some(a_act) = a_map.get(obs_index) else {
                    continue;
                };
                let Some(h_act) = h_map.get(obs_index) else {
                    continue;
                };
                if !a_act.is_finite() || !h_act.is_finite() || !target.avg_error.is_finite() {
                    continue;
                }
                let adjusted_error = target.avg_error + b_syn.weight * (*h_act);
                if !adjusted_error.is_finite() {
                    continue;
                }
                samples.push(HelpfulSample {
                    activation: *a_act,
                    avg_error: adjusted_error,
                    target_value: None,
                    target_activation: None,
                });
            }

            if samples.len() < MIN_NEURON_SAMPLE_COUNT {
                continue;
            }

            let mut sum_act_sq = 0.0f32;
            let mut sum_err_act = 0.0f32;
            let mut baseline_sq = 0.0f32;
            for s in &samples {
                sum_act_sq += s.activation * s.activation;
                sum_err_act += s.activation * s.avg_error;
                baseline_sq += s.avg_error * s.avg_error;
            }
            if baseline_sq <= EPSILON {
                continue;
            }

            let Some(weight) = calculate_optimal_outgoing_weight(sum_err_act, sum_act_sq, 1.0)
            else {
                continue;
            };

            let (improvement, improved, worsened, total) =
                compute_synapse_improvement_and_count(&samples, weight, baseline_sq, None);
            let _ = (improved, worsened, total);
            if improvement <= 0.0 {
                continue;
            }

            coordinated_structural_results.push(crate::CoordinatedStructuralCandidateJson {
                operations: vec![
                    crate::CoordinatedStructuralOpJson::RemoveSynapse {
                        from_neuron_uuid: a.to_string(),
                        to_neuron_uuid: h.to_string(),
                    },
                    crate::CoordinatedStructuralOpJson::RemoveSynapse {
                        from_neuron_uuid: h.to_string(),
                        to_neuron_uuid: b.to_string(),
                    },
                    crate::CoordinatedStructuralOpJson::RemoveNeuron {
                        neuron_uuid: h.to_string(),
                    },
                    crate::CoordinatedStructuralOpJson::AddSynapse {
                        from_neuron_uuid: a.to_string(),
                        to_neuron_uuid: b.to_string(),
                        weight,
                    },
                ],
                expected_creature_score_gain: improvement,
                comment: Some(
                    "Coordinated collapse: remove 1-in/1-out hidden neuron and add bypass synapse"
                        .to_string(),
                ),
            });
        }
    }

    // Issue #128: Apply impact-based discounting and set creature-level metrics.
    // Output neurons have impact = 1.0 (no discount).
    // Hidden neurons have impact in [0, 1] based on their weighted paths to outputs.
    let impact_scores = compute_impact_scores_for_discounting(&input.creature, cache.as_ref());
    let neuron_type_map: HashMap<String, String> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.clone(), n.neuron_type.clone()))
        .collect();

    // Apply impact discounting to helpful synapse candidates
    for candidate in &mut helpful_results {
        // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
        candidate.from_neuron_index = order_map_arc.get(&candidate.from_neuron_uuid).copied();
        candidate.to_neuron_index = order_map_arc.get(&candidate.to_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.to_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true); // Default to hidden if type unknown

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.to_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                // No impact score means disconnected from outputs - heavy discount
                0.1
            }
        } else {
            // Output neuron - full impact
            1.0
        };

        // Update creature-level metrics
        candidate.target_neuron_impact = impact;
        let original = candidate.expected_creature_error_reduction;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;

        if verbose_enabled() && is_hidden {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] Synapse candidate → {} impact {:.3}: \
                {:.4}% → {:.4}%",
                &candidate.to_neuron_uuid[..12.min(candidate.to_neuron_uuid.len())],
                impact,
                original * 100.0,
                candidate.expected_creature_score_gain * 100.0
            );
        }
    }

    // Apply impact discounting to harmful synapse candidates (same logic)
    for candidate in &mut harmful_results {
        // Populate indices for debugging/analysis (consistent with CandidateNeuronJson).
        candidate.from_neuron_index = order_map_arc.get(&candidate.from_neuron_uuid).copied();
        candidate.to_neuron_index = order_map_arc.get(&candidate.to_neuron_uuid).copied();

        let is_hidden = neuron_type_map
            .get(&candidate.to_neuron_uuid)
            .map(|t| t != "output")
            .unwrap_or(true);

        let impact = if is_hidden {
            if let Some(&impact) = impact_scores.get(&candidate.to_neuron_uuid) {
                impact.clamp(0.0, 1.0)
            } else {
                0.1
            }
        } else {
            1.0
        };

        candidate.target_neuron_impact = impact;
        candidate.expected_creature_error_reduction *= impact;
        candidate.expected_creature_score_gain = candidate.expected_creature_error_reduction;
    }

    // Apply impact discounting to coordinated candidates (based on target neuron UUID).
    for candidate in &mut coordinated_structural_results {
        // Heuristic: use the last op that targets a concrete neuron, so multi-op groups
        // (remove+add synapses, add/remove neuron, etc.) get discounted by the final target.
        // This aligns with coordinated groups that ultimately adjust the inputs of a target neuron.
        let target_uuid = candidate
            .operations
            .iter()
            .rev()
            .map(|op| match op {
                crate::CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. } => {
                    to_neuron_uuid.as_str()
                }
                crate::CoordinatedStructuralOpJson::RemoveSynapse { to_neuron_uuid, .. } => {
                    to_neuron_uuid.as_str()
                }
                crate::CoordinatedStructuralOpJson::SetWeight { to_neuron_uuid, .. } => {
                    to_neuron_uuid.as_str()
                }
                crate::CoordinatedStructuralOpJson::ChangeSquash { neuron_uuid, .. } => {
                    neuron_uuid.as_str()
                }
                crate::CoordinatedStructuralOpJson::SetBias { neuron_uuid, .. } => {
                    neuron_uuid.as_str()
                }
                crate::CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => {
                    neuron_uuid.as_str()
                }
                crate::CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid } => {
                    neuron_uuid.as_str()
                }
            })
            .next()
            .unwrap_or("");

        let is_hidden = neuron_type_map
            .get(target_uuid)
            .map(|t| t != "output")
            .unwrap_or(true);
        let impact = if is_hidden {
            impact_scores
                .get(target_uuid)
                .copied()
                .unwrap_or(0.1)
                .clamp(0.0, 1.0)
        } else {
            1.0
        };

        candidate.expected_creature_score_gain *= impact;
    }

    // Note: helpful_fallback does NOT need separate discounting here.
    // If helpful_results was empty, the fallback was already moved into it via .take()
    // at line ~6877 and gets discounted in the loop above. If helpful_results was NOT
    // empty, the fallback is intentionally not returned (we have better candidates).

    helpful_results.sort_by(|a, b| {
        // Issue #128: Sort by expected creature score gain (highest first)
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(Ordering::Equal)
    });
    harmful_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(Ordering::Equal)
    });
    coordinated_structural_results.sort_by(|a, b| {
        b.expected_creature_score_gain
            .partial_cmp(&a.expected_creature_score_gain)
            .unwrap_or(Ordering::Equal)
    });

    // Deadline coverage (Jan 2026): diversify within the top-K so repeated runs explore different
    // high-quality candidates over time (helps with failure caches and avoids category starvation).
    if input.analysis_deadline_ms.is_some() {
        const DIVERSIFY_TOP_K: usize = 64;
        shuffle_within_top_k(
            helpful_results.as_mut_slice(),
            input.random_seed,
            "synapse:helpful_candidates:top_k",
            DIVERSIFY_TOP_K,
        );
        shuffle_within_top_k(
            harmful_results.as_mut_slice(),
            input.random_seed,
            "synapse:harmful_candidates:top_k",
            DIVERSIFY_TOP_K,
        );
        shuffle_within_top_k(
            coordinated_structural_results.as_mut_slice(),
            input.random_seed,
            "synapse:coordinated_structural_candidates:top_k",
            DIVERSIFY_TOP_K,
        );
    }

    // Track candidates_found before truncation for metadata
    let candidates_found =
        helpful_results.len() + harmful_results.len() + coordinated_structural_results.len();

    if let Some(limit) = input.max_candidates {
        let (h1, h2, c) = truncate_combined_synapse_candidate_sets(
            std::mem::take(&mut helpful_results),
            std::mem::take(&mut harmful_results),
            std::mem::take(&mut coordinated_structural_results),
            limit,
            input.analysis_deadline_ms.is_some(),
        );
        helpful_results = h1;
        harmful_results = h2;
        coordinated_structural_results = c;
    }

    // Track candidates_returned after truncation
    let candidates_returned =
        helpful_results.len() + harmful_results.len() + coordinated_structural_results.len();

    let no_candidate_reasons = diagnostics.no_candidate_summaries();
    diagnostics.emit_logs();

    // Build metadata for observability (v0.2.17+)
    // Note: target_value_available and saturation_aware_simulation_used are tracked
    // during the inner analysis loop via atomic flags.
    let saw_any_input =
        metadata_seen_any_input_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let input_min = metadata_input_min_with_records.load(std::sync::atomic::Ordering::Relaxed);
    let input_max = metadata_input_max_with_records.load(std::sync::atomic::Ordering::Relaxed);

    // Issue #192: Compute error distribution from collected error values
    let error_distribution = {
        let error_vec = error_values_for_distribution
            .lock()
            .expect("Mutex poisoned: error_values_for_distribution");
        super::error_distribution::ErrorDistribution::from_errors(&error_vec)
    };

    let metadata = super::shared::SynapseAnalysisMetadata {
        target_value_available: metadata_target_value_seen
            .load(std::sync::atomic::Ordering::Relaxed),
        saturation_aware_simulation_used: metadata_saturation_aware_used
            .load(std::sync::atomic::Ordering::Relaxed),
        candidates_found,
        candidates_returned,
        timed_out: analysis_timed_out,
        completed_focus_neurons: completed_count.load(std::sync::atomic::Ordering::Relaxed),
        total_focus_neurons: total_focus_count,
        input_index_min_seen_with_records: if saw_any_input { Some(input_min) } else { None },
        input_index_max_seen_with_records: if saw_any_input { Some(input_max) } else { None },
        timing: timing_collector.finalize(),
        gpu_info: GpuAnalyzer::get_adapter_info(),
        error_distribution,
    };

    Ok(AnalyzeSynapsesResult {
        helpful_synapses: helpful_results,
        harmful_synapses: harmful_results,
        // Weight updates are represented as remove+add coordinated candidates for KISS.
        // This keeps NEAT-AI's apply/ablate pipeline limited to existing structural ops.
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: coordinated_structural_results,
        candidate_clusters: Vec::new(), // populated by analyze_all post-processing (Issue #224)
        gpu_used,
        no_candidate_reasons,
        metadata,
    })
}

// analyze_synapses has been moved to src/analysis/synapse.rs (Issue #275)

#[cfg(test)]
#[path = "implementation_tests.rs"]
mod tests;
