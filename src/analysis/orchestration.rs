//! Top-level analysis orchestration (Issue #562).
//!
//! Contains the `analyze_all` entry point and its supporting helpers:
//! - `run_optional_analysis` — guarded analysis phase execution
//! - `dispatch_analyses` — concurrent synapse/neuron dispatch (Issue #1002)

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::sync::Arc;

use anyhow::Result;

use crate::observability::{
    PhaseTimer, ProfileData, ProfileMode, global_gpu_metrics, profile_mode,
    report_global_gpu_metrics,
};
use crate::{AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput};

use super::shared::{AnalyzeAllResult, AnalyzeNeuronsResult, AnalyzeSynapsesResult};
use super::{
    cache, candidate_aggregation, candidate_compression, discovery_dispatch, module_dispatch_specs,
    module_weights, neuron, neuron_fingerprint, synapse, utils,
};

pub(crate) fn run_optional_analysis<T>(
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

/// Run synapse and neuron analyses concurrently when both are enabled (Issue #1002).
///
/// Uses `rayon::join` to run both analyses in parallel with a shared GPU queue,
/// reducing wall-clock time. When only one analysis is enabled, it runs alone.
fn dispatch_analyses(
    synapse_input: Option<AnalyzeSynapsesInput>,
    neuron_input: Option<AnalyzeNeuronsInput>,
    shared_cache: &Arc<cache::RecordCache>,
    shared_gpu_queue: &Arc<super::gpu::GpuWorkQueue>,
) -> Result<(Option<AnalyzeSynapsesResult>, Option<AnalyzeNeuronsResult>)> {
    let both_enabled = synapse_input.is_some() && neuron_input.is_some();

    if both_enabled {
        // Issue #1002: Run both analyses concurrently via rayon::join with a
        // shared GPU queue. Both analyses are independent — they share only the
        // read-only RecordCache and the thread-safe GpuWorkQueue.
        let syn_input = synapse_input.expect("checked is_some");
        let neu_input = neuron_input.expect("checked is_some");
        let cache_for_syn = Arc::clone(shared_cache);
        let cache_for_neu = Arc::clone(shared_cache);
        let gpu_for_syn = Arc::clone(shared_gpu_queue);
        let gpu_for_neu = Arc::clone(shared_gpu_queue);

        let (syn_result, neu_result) = rayon::join(
            || {
                run_optional_analysis(
                    true,
                    "analysis::analyze_all → synapse analysis starting",
                    "analysis::analyze_all → synapse analysis finished",
                    "analysis::analyze_all → synapse analysis skipped",
                    "synapse_analysis",
                    || {
                        synapse::analyze_synapses_with_cache_and_gpu_queue(
                            &syn_input,
                            cache_for_syn,
                            gpu_for_syn,
                        )
                    },
                )
            },
            || {
                run_optional_analysis(
                    true,
                    "analysis::analyze_all → neuron analysis starting",
                    "analysis::analyze_all → neuron analysis finished",
                    "analysis::analyze_all → neuron analysis skipped",
                    "neuron_analysis",
                    || {
                        neuron::analyze_neurons_with_cache_and_gpu_queue(
                            &neu_input,
                            cache_for_neu,
                            gpu_for_neu,
                        )
                    },
                )
            },
        );
        Ok((syn_result?, neu_result?))
    } else {
        // Only one (or neither) analysis is enabled — run sequentially.
        let synapse_result = run_optional_analysis(
            synapse_input.is_some(),
            "analysis::analyze_all → synapse analysis starting",
            "analysis::analyze_all → synapse analysis finished",
            "analysis::analyze_all → synapse analysis skipped",
            "synapse_analysis",
            || {
                let inner = synapse_input.expect("checked is_some");
                synapse::analyze_synapses_with_cache_and_gpu_queue(
                    &inner,
                    Arc::clone(shared_cache),
                    Arc::clone(shared_gpu_queue),
                )
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
                neuron::analyze_neurons_with_cache_and_gpu_queue(
                    &inner,
                    Arc::clone(shared_cache),
                    Arc::clone(shared_gpu_queue),
                )
            },
        )?;

        Ok((synapse_result, neuron_result))
    }
}

/// Combined analysis function that runs both synapse and neuron analysis.
///
/// # Concurrent execution (Issue #1002)
///
/// When both analyses are enabled, they run **concurrently** via `rayon::join`
/// sharing a single `GpuWorkQueue`. This reduces wall-clock time by overlapping
/// CPU-bound work across both analyses while the shared GPU thread processes
/// work from both submitters.
#[tracing::instrument(skip_all, fields(focus_neurons = input.focus_neurons.len()))]
pub fn analyze_all(input: &AnalyzeAllInput) -> Result<AnalyzeAllResult> {
    // Issue #1047: Clear any stale cancellation flag from a previous run
    // so that a prior SIGTERM does not immediately abort this invocation.
    crate::cancellation::reset_cancellation();

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
    let (effective_focus_neurons, fingerprint_cache_hits, fingerprint_cache_misses) =
        if let Some(prev_fp) = &input.previous_neuron_fingerprints {
            let filter_result = neuron_fingerprint::filter_changed_neurons(
                &input.focus_neurons,
                &input.creature,
                prev_fp,
            );

            if utils::verbose_enabled() {
                tracing::debug!(
                    cache_hits = filter_result.cache_hits,
                    total = filter_result.total_focus_neurons,
                    to_analyse = filter_result.cache_misses,
                    "incremental analysis: skipping unchanged focus neurons"
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
            memory_budget_exceeded: false,
            cancelled: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    // If all focus neurons were skipped by fingerprint filtering, return early.
    if effective_focus_neurons.is_empty() && (include_synapse || include_neuron) {
        if utils::verbose_enabled() {
            tracing::debug!("all focus neurons unchanged — skipping GPU analysis");
        }
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: false,
            cancelled: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    // Issue #1028: Check memory budget before expensive GPU work.
    if utils::is_memory_budget_exceeded(input.max_analysis_memory_mb) {
        tracing::warn!(
            budget_mb = input.max_analysis_memory_mb,
            allocated_bytes = crate::ALLOCATOR.allocated(),
            "memory budget exceeded before GPU analysis — returning early with no candidates"
        );
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: true,
            cancelled: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    // Early GPU availability check (Issue #988): fail fast with a structured
    // error before attempting expensive parquet I/O. The per-module checks in
    // synapse/neuron analysis are retained as a defence-in-depth measure.
    if !super::gpu::GpuAnalyzer::gpu_is_available() {
        return Err(crate::ffi_types::DiscoveryError::GpuUnavailable {
            reason: "No compatible GPU adapter found on this system".to_string(),
        }
        .into());
    }

    crate::watchdog::beat("analysis::analyze_all → loading parquet cache");

    // Pre-load ALL records from parquet in one pass. This is MUCH faster than
    // lazy-loading each neuron separately (1 scan vs ~2000 scans for large creatures).
    // Issue #648: Pass the analysis deadline so loading can abort early if time runs out.
    let loading_deadline = utils::build_deadline(input.analysis_deadline_ms);
    let parquet_loading_start = std::time::Instant::now();
    let cache_result =
        cache::RecordCache::new_adaptive_with_deadline(&input.parquet_file, loading_deadline);

    // Issue #1047: If parquet loading was cancelled, return a clean partial
    // result instead of propagating the error.
    let shared_cache = match cache_result {
        Ok(c) => Arc::new(c),
        Err(_e) if crate::cancellation::is_cancelled() => {
            tracing::info!("parquet loading cancelled by host — returning empty result");
            return Ok(AnalyzeAllResult {
                synapse: None,
                neuron: None,
                memory_budget_exceeded: false,
                cancelled: true,
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            });
        }
        Err(e) => return Err(e),
    };
    profile.record_phase(
        "parquet_loading",
        parquet_loading_start.elapsed().as_millis() as u64,
    );
    crate::watchdog::beat("analysis::analyze_all → parquet cache loaded");

    // Issue #1028: Check memory budget after parquet loading (often the largest
    // single allocation). If the cache already consumed most of the budget,
    // return early with partial results rather than proceeding to GPU analysis.
    if utils::is_memory_budget_exceeded(input.max_analysis_memory_mb) {
        tracing::warn!(
            budget_mb = input.max_analysis_memory_mb,
            allocated_bytes = crate::ALLOCATOR.allocated(),
            "memory budget exceeded after parquet loading — returning early"
        );
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: true,
            cancelled: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    let synapse_input = if include_synapse {
        Some(AnalyzeSynapsesInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: effective_focus_neurons.clone(),
            max_candidates: input.max_synapse_candidates,
            analysis_deadline_ms: input.analysis_deadline_ms,
            random_seed: input.random_seed,
            temperature: input.temperature,
        })
    } else {
        None
    };

    let neuron_input = if include_neuron {
        Some(AnalyzeNeuronsInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: effective_focus_neurons,
            max_candidates: input.max_neuron_candidates,
            analysis_deadline_ms: input.analysis_deadline_ms,
            random_seed: input.random_seed,
            temperature: input.temperature,
        })
    } else {
        None
    };

    // Issue #1002: Create a shared GPU work queue for both analyses.
    // The GpuWorkQueue is designed for concurrent submitters via crossbeam_channel,
    // so a single GPU thread serves both synapse and neuron analyses.
    let loading_deadline_for_gpu = utils::build_deadline(input.analysis_deadline_ms);
    let shared_gpu_queue =
        Arc::new(super::gpu::GpuWorkQueue::new()?.with_deadline(loading_deadline_for_gpu));

    // Issue #1002: Both analyses run concurrently via rayon::join when both are
    // enabled, so randomised ordering is no longer needed — both get the full
    // time budget.
    let dispatch_result = dispatch_analyses(
        synapse_input,
        neuron_input,
        &shared_cache,
        &shared_gpu_queue,
    );

    // Issue #1047: If analysis was cancelled during dispatch, return a clean
    // partial result rather than propagating the error.
    let (synapse_result, neuron_result) = match dispatch_result {
        Ok(results) => results,
        Err(_) if crate::cancellation::is_cancelled() => {
            tracing::info!("analysis dispatch cancelled by host — returning empty result");
            return Ok(AnalyzeAllResult {
                synapse: None,
                neuron: None,
                memory_budget_exceeded: false,
                cancelled: true,
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            });
        }
        Err(e) => return Err(e),
    };

    // Issue #1028: Check memory budget after GPU analysis. If exceeded, skip
    // post-processing and return the candidates we have so far.
    let memory_budget_exceeded = utils::is_memory_budget_exceeded(input.max_analysis_memory_mb);
    if memory_budget_exceeded {
        tracing::warn!(
            budget_mb = input.max_analysis_memory_mb,
            allocated_bytes = crate::ALLOCATOR.allocated(),
            "memory budget exceeded after GPU analysis — skipping post-processing"
        );
    }

    // Post-process: convert certain add-neuron candidates into coordinated-structural replacements.
    let mut synapse_result = synapse_result;
    let mut neuron_result = neuron_result;

    // Issue #792: Resolve the module outcome tracker from input or use a default.
    let tracker = input.module_outcome_tracker.clone().unwrap_or_default();

    // Issue #1057: Gate add-synapse candidates based on historical success rate
    // and synapse density. When the ModuleOutcomeTracker shows consistent failure
    // or the network is too dense, clear helpful_synapses to save compute.
    if let Some(syn) = synapse_result.as_mut() {
        synapse::add_synapse_gating::gate_add_synapse_candidates(
            &mut syn.helpful_synapses,
            &tracker,
            &input.creature,
        );
    }

    if !memory_budget_exceeded
        && let (Some(syn), Some(neuron)) = (synapse_result.as_mut(), neuron_result.as_mut())
    {
        candidate_aggregation::convert_neurons_to_coordinated_replacements(
            input,
            syn,
            neuron,
            &shared_cache,
        );
    }

    // Issue #1028: Skip post-processing when memory budget is exceeded.
    // The candidates from GPU analysis are still returned, but compression,
    // discovery module detection, and reranking are skipped to avoid further
    // memory growth.
    if !memory_budget_exceeded {
        // Issue #1004: Overlap candidate compression with discovery module detection.
        //
        // Both compression (reads `helpful_synapses` immutably) and discovery module
        // detection (reads creature/cache, runs ~48 detection closures) are independent
        // in their detection phases. We run them concurrently via `rayon::join`, then
        // merge results sequentially (compression first, then discovery modules) to
        // preserve deterministic ordering.
        if let Some(syn) = synapse_result.as_mut() {
            let max_candidates = input.max_synapse_candidates;
            let diversify = input.analysis_deadline_ms.is_some();

            // Snapshot data needed by compression (immutable reads).
            let helpful_synapses_snapshot = syn.helpful_synapses.clone();
            let creature_for_compression = input.creature.clone();

            // Data needed by discovery module detection.
            let creature = Arc::new(input.creature.clone());
            let hidden_neurons: Arc<Vec<(String, String, f32)>> = Arc::new(
                input
                    .creature
                    .neurons
                    .iter()
                    .filter(|n| n.neuron_type == "hidden")
                    .map(|n| {
                        // Issue #753: ensure squash is uppercase even for
                        // programmatically constructed NeuronJson (serde path
                        // normalises during deserialisation, this covers the rest).
                        let squash = if n.squash.bytes().all(|b| !b.is_ascii_lowercase()) {
                            n.squash.clone()
                        } else {
                            n.squash.to_ascii_uppercase()
                        };
                        (n.uuid.clone(), squash, n.bias)
                    })
                    .collect(),
            );

            // Issue #1004: Run compression detection and discovery module detection
            // concurrently. The heavier discovery dispatch (~48 parallel modules)
            // overlaps with the lighter compression work.
            let (all_compressed, discovery_results) = rayon::join(
                || {
                    // Issue #921 / #922: Compress IDENTITY and non-linear candidates.
                    let (identity_compressed, nonlinear_compressed) = rayon::join(
                        || {
                            candidate_compression::compress_identity_candidates(
                                &helpful_synapses_snapshot,
                                &creature_for_compression,
                            )
                        },
                        || {
                            candidate_compression::compress_nonlinear_candidates(
                                &helpful_synapses_snapshot,
                                &creature_for_compression,
                            )
                        },
                    );
                    let mut all = identity_compressed;
                    all.extend(nonlinear_compressed);
                    all
                },
                || {
                    // Issue #375 / #419: Discovery module detection phase only.
                    // Issue #1029: Pass the analysis deadline so detection modules
                    // are skipped when time runs out, preventing lockups.
                    let discovery_deadline = utils::build_deadline(input.analysis_deadline_ms);
                    module_dispatch_specs::prepare_and_detect_discovery_modules(
                        &creature,
                        &hidden_neurons,
                        &shared_cache,
                        &tracker,
                        discovery_deadline,
                    )
                },
            );

            // Sequential merge phase: compression results first, then discovery modules.
            // This preserves the same ordering as the previous sequential pipeline.
            if !all_compressed.is_empty() {
                candidate_aggregation::merge_coordinated_structural_replacements(
                    syn,
                    all_compressed,
                    max_candidates,
                    diversify,
                );
            }

            discovery_dispatch::merge_discovery_module_results(
                syn,
                discovery_results,
                max_candidates,
                diversify,
                &tracker,
            );
        }

        // Issue #963: Cross-detection candidate synthesis — synthesise combined
        // candidates when multiple detection modules flag the same neuron.
        if let Some(syn) = synapse_result.as_mut() {
            module_dispatch_specs::synthesise_cross_detection_candidates(syn);
        }

        // Issue #489: Cross-module candidate deduplication.
        if let Some(syn) = synapse_result.as_mut() {
            module_dispatch_specs::deduplicate_cross_module_candidates(syn);
        }

        // Issue #572: Ensemble candidate scoring — combine predictions across modules.
        if let Some(syn) = synapse_result.as_mut() {
            module_dispatch_specs::apply_ensemble_scoring(syn, &tracker);
        }

        // Issue #792: Apply per-module boost factors to candidate expected gains.
        if let Some(syn) = synapse_result.as_mut() {
            module_weights::apply_module_boost_to_candidates(
                &mut syn.coordinated_structural_candidates,
                &tracker,
            );
        }

        // Issue #610: Diversity-aware reranking — penalise structurally similar candidates.
        if let Some(syn) = synapse_result.as_mut() {
            module_dispatch_specs::apply_diversity_reranking(syn);
        }

        // Issue #224: Candidate clustering to reduce redundant ablation tests.
        if let Some(syn) = synapse_result.as_mut() {
            module_dispatch_specs::cluster_synapse_candidates(syn, &input.creature);
        }
    } // end if !memory_budget_exceeded (Issue #1028)

    // Collect final profile data (Issue #214)
    let synapse_candidates = synapse_result.as_ref().map_or(0, |s| {
        s.helpful_synapses.len()
            + s.harmful_synapses.len()
            + s.coordinated_structural_candidates.len()
    });
    let neuron_candidates = neuron_result
        .as_ref()
        .map_or(0, |n| n.helpful_neurons.len());
    let total_candidates = synapse_candidates + neuron_candidates;
    profile.set_candidates_found(total_candidates);
    profile.set_candidates_returned(total_candidates);

    // Set focus neurons completed from metadata
    let synapse_completed = synapse_result
        .as_ref()
        .map_or(0, |s| s.metadata.completed_focus_neurons);
    let neuron_completed = neuron_result
        .as_ref()
        .map_or(0, |n| n.metadata.completed_focus_neurons);
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
        memory_budget_exceeded,
        cancelled: crate::cancellation::is_cancelled(),
        neuron_fingerprints: Some(current_fingerprints),
        fingerprint_cache_hits,
        fingerprint_cache_misses,
        module_outcome_tracker: tracker,
    })
}
