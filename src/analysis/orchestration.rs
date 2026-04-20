//! Top-level analysis orchestration (Issue #562).
//!
//! Contains the `analyze_all` entry point and its supporting helpers:
//! - `run_optional_analysis` — guarded analysis phase execution
//! - `dispatch_analyses` — concurrent synapse/neuron dispatch (Issue #1002)

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use std::panic::AssertUnwindSafe;
use std::sync::Arc;

use anyhow::{Context, Result};

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

/// Aggregate rejection counts from synapse `no_candidate_reasons` into the
/// structured breakdown and compute the one-sentence `top_level_summary`
/// (Issue #1129).
fn aggregate_synapse_rejection_breakdown(syn: &mut AnalyzeSynapsesResult) {
    use super::diagnostics::rejection_reasons::{
        REJECTION_BELOW_THRESHOLD, REJECTION_NO_DIAGNOSTICS, REJECTION_NO_ELIGIBLE_SOURCES,
        REJECTION_NO_SAMPLES, REJECTION_NO_TARGET_RECORDS, REJECTION_ZERO_IMPROVEMENT,
        top_level_summary,
    };
    use super::shared::SynapseNoCandidateReason;

    for summary in &syn.no_candidate_reasons {
        let reason_name = match summary.reason {
            SynapseNoCandidateReason::NoEligibleSources => REJECTION_NO_ELIGIBLE_SOURCES,
            SynapseNoCandidateReason::NoDiagnostics => REJECTION_NO_DIAGNOSTICS,
            SynapseNoCandidateReason::NoSamples => REJECTION_NO_SAMPLES,
            SynapseNoCandidateReason::ZeroImprovement => REJECTION_ZERO_IMPROVEMENT,
            SynapseNoCandidateReason::BelowThreshold => REJECTION_BELOW_THRESHOLD,
            SynapseNoCandidateReason::NoTargetRecords => REJECTION_NO_TARGET_RECORDS,
        };
        syn.metadata.rejection_breakdown.record(reason_name);
    }
    // Denominator is rejections + returned: the total number of candidate
    // decisions the pipeline made. This lets "N of M" read naturally.
    let denom = syn.metadata.rejection_breakdown.total()
        + u32::try_from(syn.metadata.candidates_returned).unwrap_or(u32::MAX);
    syn.metadata.top_level_summary =
        top_level_summary(&syn.metadata.rejection_breakdown, Some(denom));
}

/// Aggregate rejection counts from neuron `no_candidate_reasons` into the
/// structured breakdown and compute the one-sentence `top_level_summary`
/// (Issue #1129).
fn aggregate_neuron_rejection_breakdown(neu: &mut AnalyzeNeuronsResult) {
    use super::diagnostics::rejection_reasons::{
        REJECTION_BELOW_THRESHOLD, REJECTION_CONSTANT_NEURON_FILTERED,
        REJECTION_HIDDEN_NEURON_FILTERED, REJECTION_INPUT_NEURON_FILTERED,
        REJECTION_NO_DIAGNOSTICS, REJECTION_NO_ELIGIBLE_SOURCES, REJECTION_NO_SAMPLES,
        top_level_summary,
    };
    use super::shared::NeuronNoCandidateReason;

    for summary in &neu.no_candidate_reasons {
        let reason_name = match summary.reason {
            NeuronNoCandidateReason::NoEligibleSources => REJECTION_NO_ELIGIBLE_SOURCES,
            NeuronNoCandidateReason::NoDiagnostics => REJECTION_NO_DIAGNOSTICS,
            NeuronNoCandidateReason::NoSamples | NeuronNoCandidateReason::NotEnoughActivations => {
                REJECTION_NO_SAMPLES
            }
            NeuronNoCandidateReason::WeightDegenerate | NeuronNoCandidateReason::BelowThreshold => {
                REJECTION_BELOW_THRESHOLD
            }
            NeuronNoCandidateReason::HiddenNeuronFiltered => REJECTION_HIDDEN_NEURON_FILTERED,
            NeuronNoCandidateReason::InputNeuronFiltered => REJECTION_INPUT_NEURON_FILTERED,
            NeuronNoCandidateReason::ConstantNeuronFiltered => REJECTION_CONSTANT_NEURON_FILTERED,
        };
        neu.metadata.rejection_breakdown.record(reason_name);
    }
    let denom = neu.metadata.rejection_breakdown.total()
        + u32::try_from(neu.metadata.candidates_returned).unwrap_or(u32::MAX);
    neu.metadata.top_level_summary =
        top_level_summary(&neu.metadata.rejection_breakdown, Some(denom));
}

/// Extract a human-readable message from a panic payload (Issue #1087).
fn format_panic_payload(payload: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        format!("{payload:?}")
    }
}

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
        let result = f().with_context(|| format!("failed during {phase_name} phase"))?;
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

        // Issue #1087: Wrap each rayon::join branch with catch_unwind so a
        // panic in one analysis does not corrupt results from the other.
        let (syn_result, neu_result) = rayon::join(
            || {
                std::panic::catch_unwind(AssertUnwindSafe(|| {
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
                }))
                .unwrap_or_else(|panic_payload| {
                    let msg = format_panic_payload(&panic_payload);
                    tracing::warn!(
                        phase = "synapse_analysis",
                        panic_message = %msg,
                        "Synapse analysis panicked — caught and converted to error (Issue #1087)"
                    );
                    Err(anyhow::anyhow!("synapse analysis module panicked: {msg}"))
                })
            },
            || {
                std::panic::catch_unwind(AssertUnwindSafe(|| {
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
                }))
                .unwrap_or_else(|panic_payload| {
                    let msg = format_panic_payload(&panic_payload);
                    tracing::warn!(
                        phase = "neuron_analysis",
                        panic_message = %msg,
                        "Neuron analysis panicked — caught and converted to error (Issue #1087)"
                    );
                    Err(anyhow::anyhow!("neuron analysis module panicked: {msg}"))
                })
            },
        );
        Ok((
            syn_result.context("failed during synapse analysis phase")?,
            neu_result.context("failed during neuron analysis phase")?,
        ))
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
            memory_pressure_cancelled: false,
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
            memory_pressure_cancelled: false,
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
            memory_pressure_cancelled: false,
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

    // Issue #1099: Check system memory pressure before expensive parquet I/O.
    // If the system is under CRITICAL pressure (< 5% available), cancel early
    // to prevent OOM. This self-monitoring complements the host-side
    // `cancel_analysis_memory_pressure()` FFI call.
    if utils::check_memory_pressure_and_cancel() {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: false,
            cancelled: true,
            memory_pressure_cancelled: true,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    crate::watchdog::beat("analysis::analyze_all → loading parquet cache");

    // Issue #1097: Build the overall deadline ONCE and convert to an absolute
    // timestamp. All sub-phases share this single deadline so that a relative
    // duration (e.g., 600_000ms = 10 minutes) is not re-interpreted as "10
    // minutes from now" by each phase independently. Without this, parquet
    // loading, synapse analysis, and neuron analysis each got a fresh 10-minute
    // window, allowing total analysis to exceed 24 minutes on a 10-minute budget.
    let analysis_deadline = utils::build_deadline(input.analysis_deadline_ms);

    // Issue #1098: Cap the analysis deadline to the overall wall-clock limit.
    // Discovery has two additive timeouts (recording + analysis) with no overall
    // cap. The wall-clock cap ensures total elapsed time never exceeds the
    // configured limit, even if recording consumed some of the budget.
    // If the caller does not pass a cap, fall back to the environment variable
    // NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES (default 20 min).
    let wall_clock_minutes = input
        .max_discovery_wall_clock_minutes
        .unwrap_or_else(crate::config::max_wall_clock_minutes);
    let discovery_start = std::time::SystemTime::now();
    let overall_deadline = utils::cap_deadline_to_wall_clock(
        analysis_deadline,
        discovery_start,
        Some(wall_clock_minutes),
    );
    if analysis_deadline != overall_deadline {
        tracing::info!(
            wall_clock_cap_minutes = wall_clock_minutes,
            "Issue #1098: analysis deadline capped by wall-clock limit"
        );
    }
    let shared_deadline_abs_ms = utils::deadline_to_absolute_ms(&overall_deadline);

    // Pre-load ALL records from parquet in one pass. This is MUCH faster than
    // lazy-loading each neuron separately (1 scan vs ~2000 scans for large creatures).
    // Issue #648: Pass the analysis deadline so loading can abort early if time runs out.
    let loading_deadline = overall_deadline;
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
                memory_pressure_cancelled: crate::cancellation::is_memory_pressure_cancelled(),
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            });
        }
        Err(e) => return Err(e).context("failed to load parquet record cache for analysis"),
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
            memory_pressure_cancelled: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    // Issue #1099: Re-check system memory pressure after parquet loading
    // (the single largest allocation). Parquet loading may have pushed the
    // system into CRITICAL pressure.
    if utils::check_memory_pressure_and_cancel() {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: false,
            cancelled: true,
            memory_pressure_cancelled: true,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
        });
    }

    // Issue #1097: Pass the shared absolute deadline to sub-phases so they
    // all count down from the same point in time.
    let synapse_input = if include_synapse {
        Some(AnalyzeSynapsesInput {
            parquet_file: input.parquet_file.clone(),
            creature: input.creature.clone(),
            focus_neurons: effective_focus_neurons.clone(),
            max_candidates: input.max_synapse_candidates,
            analysis_deadline_ms: shared_deadline_abs_ms,
            random_seed: input.random_seed,
            temperature: input.temperature,
            failure_cache: input.failure_cache.clone(),
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
            analysis_deadline_ms: shared_deadline_abs_ms,
            random_seed: input.random_seed,
            temperature: input.temperature,
            failure_cache: input.failure_cache.clone(),
        })
    } else {
        None
    };

    // Issue #1002: Create a shared GPU work queue for both analyses.
    // The GpuWorkQueue is designed for concurrent submitters via crossbeam_channel,
    // so a single GPU thread serves both synapse and neuron analyses.
    // Issue #1097: Reuse the already-computed overall deadline instead of
    // building a new one (which would reset a relative duration).
    let loading_deadline_for_gpu = overall_deadline;
    let shared_gpu_queue = Arc::new(
        super::gpu::GpuWorkQueue::new()
            .context("failed to create GPU work queue for analysis dispatch")?
            .with_deadline(loading_deadline_for_gpu),
    );

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
                memory_pressure_cancelled: crate::cancellation::is_memory_pressure_cancelled(),
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            });
        }
        Err(e) => return Err(e).context("failed during analysis dispatch"),
    };

    // Issue #1099: Check system memory pressure after GPU dispatch.
    // The GPU analysis phase accumulates large candidate buffers that may
    // push the system into CRITICAL pressure. If so, skip post-processing
    // and return partial results.
    utils::check_memory_pressure_and_cancel();

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
    let mut tracker = input.module_outcome_tracker.clone().unwrap_or_default();

    // Issue #1057: Gate add-synapse candidates based on historical success rate
    // and synapse density. When the ModuleOutcomeTracker shows consistent failure
    // or the network is too dense, clear helpful_synapses to save compute.
    if let Some(syn) = synapse_result.as_mut() {
        let removed = synapse::add_synapse_gating::gate_add_synapse_candidates(
            &mut syn.helpful_synapses,
            &tracker,
            &input.creature,
        );
        // Issue #1129: Record the gating drop in the structured rejection
        // breakdown so downstream callers can see why helpful synapses vanished.
        syn.metadata.rejection_breakdown.record_many_u32(
            super::diagnostics::rejection_reasons::REJECTION_ADD_SYNAPSE_GATED,
            u32::try_from(removed).unwrap_or(u32::MAX),
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

    // Issue #1097: Check overall deadline before post-processing. If the
    // analysis phases consumed the entire time budget, skip the heavyweight
    // post-processing to stay within the configured timeout.
    let post_processing_deadline_passed = utils::deadline_passed(&overall_deadline);
    if post_processing_deadline_passed {
        tracing::info!(
            "Analysis deadline reached before post-processing — skipping discovery \
             modules, compression, and reranking to stay within timeout (Issue #1097)"
        );
    }

    // Issue #1028: Skip post-processing when memory budget is exceeded.
    // Issue #1097: Also skip when the analysis deadline has passed.
    // The candidates from GPU analysis are still returned, but compression,
    // discovery module detection, and reranking are skipped to avoid further
    // memory growth or exceeding the timeout.
    if !memory_budget_exceeded && !post_processing_deadline_passed {
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
            // Issue #1087: Wrap both branches with catch_unwind to prevent panics
            // in one branch from corrupting the other's results.
            let (compressed_result, discovery_result) = rayon::join(
                || {
                    std::panic::catch_unwind(AssertUnwindSafe(|| {
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
                    }))
                    .unwrap_or_else(|panic_payload| {
                        let msg = format_panic_payload(&panic_payload);
                        tracing::warn!(
                            phase = "candidate_compression",
                            panic_message = %msg,
                            "Candidate compression panicked — returning empty results \
                             (Issue #1087)"
                        );
                        Vec::new()
                    })
                },
                || {
                    std::panic::catch_unwind(AssertUnwindSafe(|| {
                        // Issue #375 / #419: Discovery module detection phase only.
                        // Issue #1029: Pass the analysis deadline so detection modules
                        // are skipped when time runs out, preventing lockups.
                        // Issue #1097: Use the shared absolute deadline.
                        let discovery_deadline = utils::build_deadline(shared_deadline_abs_ms);
                        module_dispatch_specs::prepare_and_detect_discovery_modules(
                            &creature,
                            &hidden_neurons,
                            &shared_cache,
                            &tracker,
                            discovery_deadline,
                        )
                    }))
                    .unwrap_or_else(|panic_payload| {
                        let msg = format_panic_payload(&panic_payload);
                        tracing::warn!(
                            phase = "discovery_module_detection",
                            panic_message = %msg,
                            "Discovery module detection panicked — returning empty results \
                             (Issue #1087)"
                        );
                        discovery_dispatch::DiscoveryModuleDetectionResults {
                            entries: Vec::new(),
                        }
                    })
                },
            );
            let all_compressed = compressed_result;
            let discovery_results = discovery_result;

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
                &mut tracker,
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

        // Issue #1110, #1128: Final coordinated-structural gain floor.
        // Applied AFTER module boost and diversity reranking so that gains
        // which started above the floor but were discounted by those steps
        // cannot reach the FFI response. Production failure evidence
        // (GRQ-sampler failures cache) shows sub-1e-5 gains harm the network.
        // Metadata must be refreshed after the sweep since
        // `merge_coordinated_structural_replacements` wrote
        // `candidates_returned` before these downstream filters ran.
        if let Some(syn) = synapse_result.as_mut() {
            let removed = candidate_aggregation::apply_coordinated_gain_floor(
                &mut syn.coordinated_structural_candidates,
            );
            syn.metadata.rejection_breakdown.record_many_u32(
                super::diagnostics::rejection_reasons::REJECTION_BELOW_EXPECTED_GAIN_FLOOR,
                removed,
            );
            syn.metadata.candidates_returned = syn.helpful_synapses.len()
                + syn.harmful_synapses.len()
                + syn.coordinated_structural_candidates.len();
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

    // Issue #1129: Aggregate per-target rejection reasons into the structured
    // breakdown and compute a one-sentence top-level summary so downstream
    // tooling can root-cause "no candidates found" without re-running analysis.
    if let Some(syn) = synapse_result.as_mut() {
        aggregate_synapse_rejection_breakdown(syn);
    }
    if let Some(neu) = neuron_result.as_mut() {
        aggregate_neuron_rejection_breakdown(neu);
    }

    Ok(AnalyzeAllResult {
        synapse: synapse_result,
        neuron: neuron_result,
        memory_budget_exceeded,
        cancelled: crate::cancellation::is_cancelled(),
        memory_pressure_cancelled: crate::cancellation::is_memory_pressure_cancelled(),
        neuron_fingerprints: Some(current_fingerprints),
        fingerprint_cache_hits,
        fingerprint_cache_misses,
        module_outcome_tracker: tracker,
    })
}
