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

use super::cost_function_hint::CostFunctionHint;
use super::diagnostics::{RejectionBreakdown, rejection_reasons};
use super::shared::{AnalyzeAllResult, AnalyzeNeuronsResult, AnalyzeSynapsesResult};
use super::task_descriptor::TaskDescriptor;
use super::{
    cache, candidate_aggregation, candidate_compression, discovery_dispatch,
    fingerprint_skip_escape, module_dispatch_specs, module_weights, neuron, neuron_fingerprint,
    synapse, utils,
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

/// Total focus targets dropped by the per-target cooldown filter across the
/// whole pass (Issue #1797).
///
/// Each surface filters its own focus order, so the pass total is the sum of
/// the two phase counts; a surface that did not run contributes nothing. This
/// is what feeds the drought diagnostic's `target_cooldown_skipped` field,
/// replacing the hard-coded `0` the diagnostic used to report.
fn pass_target_cooldown_skipped(
    synapse: Option<&super::shared::SynapseAnalysisMetadata>,
    neuron: Option<&super::shared::NeuronAnalysisMetadata>,
) -> u32 {
    let synapse_skipped = synapse.map_or(0, |m| m.target_cooldown_skipped);
    let neuron_skipped = neuron.map_or(0, |m| m.target_cooldown_skipped);
    synapse_skipped.saturating_add(neuron_skipped)
}

/// Build the pass-level rejection breakdown, seeded with the focus neurons the
/// structural fingerprint cache skipped this pass (Issue #1781, #1801).
///
/// A skipped focus neuron was never analysed, so it could not produce a
/// proposal — its absence must not read as "the gate rejected it"
/// (`REJECTION_FINGERPRINT_UNCHANGED` is in
/// [`UPSTREAM_REJECTION_REASONS`](super::candidate_starvation::UPSTREAM_REJECTION_REASONS)).
/// #1781 counted only the whole-pass skip; the far more common partial skip
/// (some focus neurons unchanged, the rest analysed) stayed silent. Every
/// `analyze_all` return path builds its breakdown here, so both cases are
/// visible and — the returns being mutually exclusive — the same hits can never
/// be counted twice. Zero hits yields an empty breakdown, because
/// [`RejectionBreakdown::record_many_u32`] ignores a zero count.
fn pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits: usize) -> RejectionBreakdown {
    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many_u32(
        rejection_reasons::REJECTION_FINGERPRINT_UNCHANGED,
        u32::try_from(fingerprint_cache_hits).unwrap_or(u32::MAX),
    );
    breakdown
}

/// The pass breakdown for a run whose GPU analyses were skipped because the GPU
/// is wedged (Issue #1931).
///
/// One `gpu_wedged` count per focus neuron that survived the fingerprint cache
/// and was still never evaluated, so the pass reads as an environmental failure
/// rather than search exhaustion.
fn pass_breakdown_with_gpu_wedged(
    fingerprint_cache_hits: usize,
    skipped_focus_neurons: usize,
) -> RejectionBreakdown {
    let mut breakdown = pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits);
    breakdown.record_many_u32(
        rejection_reasons::REJECTION_GPU_WEDGED,
        u32::try_from(skipped_focus_neurons).unwrap_or(u32::MAX),
    );
    breakdown
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

/// Wall-clock milliseconds spent in each analysis phase (Issue #1409).
///
/// `None` when the corresponding analysis was disabled for the run. Feeds the
/// consolidated per-cycle deadline-consumption breakdown.
#[derive(Debug, Clone, Copy, Default)]
struct DispatchDurations {
    synapse_ms: Option<u64>,
    neuron_ms: Option<u64>,
}

/// Run synapse and neuron analyses concurrently when both are enabled (Issue #1002).
///
/// Uses `rayon::join` to run both analyses in parallel with a shared GPU queue,
/// reducing wall-clock time. When only one analysis is enabled, it runs alone.
///
/// Also returns the per-phase wall-clock durations (Issue #1409) so the
/// orchestrator can attribute deadline consumption.
fn dispatch_analyses(
    synapse_input: Option<AnalyzeSynapsesInput>,
    neuron_input: Option<AnalyzeNeuronsInput>,
    shared_cache: &Arc<cache::RecordCache>,
    shared_gpu_queue: &Arc<super::gpu::GpuWorkQueue>,
) -> Result<(
    Option<AnalyzeSynapsesResult>,
    Option<AnalyzeNeuronsResult>,
    DispatchDurations,
)> {
    use std::time::Instant;

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
        // Issue #1409: Each branch also returns its wall-clock duration.
        let ((syn_result, syn_ms), (neu_result, neu_ms)) = rayon::join(
            || {
                let started = Instant::now();
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
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
                });
                (result, started.elapsed().as_millis() as u64)
            },
            || {
                let started = Instant::now();
                let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
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
                });
                (result, started.elapsed().as_millis() as u64)
            },
        );
        Ok((
            syn_result.context("failed during synapse analysis phase")?,
            neu_result.context("failed during neuron analysis phase")?,
            DispatchDurations {
                synapse_ms: Some(syn_ms),
                neuron_ms: Some(neu_ms),
            },
        ))
    } else {
        // Only one (or neither) analysis is enabled — run sequentially.
        let synapse_enabled = synapse_input.is_some();
        let synapse_started = Instant::now();
        let synapse_result = run_optional_analysis(
            synapse_enabled,
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
        let synapse_ms = synapse_enabled.then(|| synapse_started.elapsed().as_millis() as u64);

        let neuron_enabled = neuron_input.is_some();
        let neuron_started = Instant::now();
        let neuron_result = run_optional_analysis(
            neuron_enabled,
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
        let neuron_ms = neuron_enabled.then(|| neuron_started.elapsed().as_millis() as u64);

        Ok((
            synapse_result,
            neuron_result,
            DispatchDurations {
                synapse_ms,
                neuron_ms,
            },
        ))
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

    // Issue #1790: advance the global target-cooldown epoch exactly once per
    // discovery pass, at the head of the pass. Both `apply_target_cooldown`
    // call sites (neuron preparation and synapse orchestration) read the epoch
    // off this tracker later in the pass, so advancing here — rather than
    // inside each — gives them one consistent value and moves the counter by
    // exactly one per pass. Without this the counter never left `0` and no
    // cooldown could ever expire.
    let pass_epoch = super::target_failure_tracker::advance_global_epoch();
    tracing::debug!(
        pass_epoch,
        "Issue #1790: advanced the global target-cooldown epoch for this discovery pass"
    );

    // Phase timer for total analysis (Issue #214). Also drives the consolidated
    // per-cycle deadline-consumption breakdown total (Issue #1409).
    let total_timer = PhaseTimer::new("total_analysis");

    // Profile data collection (when NEAT_AI_DISCOVERY_PROFILE=json)
    let mut profile = ProfileData::new();
    profile.set_focus_neurons_requested(input.focus_neurons.len());

    // Optional hang watchdog for unattended workers.
    // If enabled, this will emit a thread dump then abort the process if analysis stalls.
    let _watchdog = crate::watchdog::start_from_env("analysis::analyze_all");

    let include_synapse = input.include_synapse_analysis.unwrap_or(true);
    let include_neuron = input.include_neuron_analysis.unwrap_or(true);

    // Issue #1317: Derive the cost-function hint that gates the
    // implied-target reconstruction guard. When the caller supplies a known
    // linear-residual cost name (MSE / MAE / CE / BCE) the reconstruction-
    // dependent detectors run; for non-linear costs (MAPE / MSLE / HINGE /
    // CATEGORICAL_ERROR) they are skipped; absent / unrecognised / OTHER
    // collapses to `neutral()` which maps to a conservative skip.
    // Issue #1316: also keep the full descriptor — the output_bias_drift
    // module needs the topology (OneHot / Simplex) to weight up capacity-
    // starved output neurons.
    let task_descriptor: TaskDescriptor = input
        .cost_name
        .as_deref()
        .map_or_else(TaskDescriptor::neutral, |name| {
            TaskDescriptor::from_name(name, input.creature.output)
        });
    let cost_hint: CostFunctionHint = task_descriptor.cost_function_hint();

    // Issue #490: Compute current fingerprints and filter unchanged neurons.
    // Issue #1781: the structural fingerprint cache is released while the
    // creature is in a drought — the topology cannot change while nothing is
    // accepted, so the cache would otherwise skip every focus neuron pass after
    // pass, ignoring freshly recorded data.
    let current_fingerprints = neuron_fingerprint::compute_neuron_fingerprints(&input.creature);
    let fingerprint_cache_bypassed = fingerprint_skip_escape::should_bypass_fingerprint_cache(
        input.discovery_outcome_log.as_ref(),
        fingerprint_skip_escape::DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS,
    );
    if fingerprint_cache_bypassed && input.previous_neuron_fingerprints.is_some() {
        tracing::warn!(
            focus_neurons = input.focus_neurons.len(),
            drought_epochs = fingerprint_skip_escape::DEFAULT_FINGERPRINT_SKIP_DROUGHT_EPOCHS,
            "Issue #1781: drought escape hatch — ignoring previousNeuronFingerprints and \
             re-analysing the full focus set"
        );
    }
    let (effective_focus_neurons, fingerprint_cache_hits, fingerprint_cache_misses) =
        if let Some(prev_fp) = input
            .previous_neuron_fingerprints
            .as_ref()
            .filter(|_| !fingerprint_cache_bypassed)
        {
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
            gpu_wedged: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
        });
    }

    // If all focus neurons were skipped by fingerprint filtering, return early.
    if effective_focus_neurons.is_empty() && (include_synapse || include_neuron) {
        // Issue #1781: count the whole-pass drop so it is not silent. Without
        // this the pass returns nothing at all — no candidates, no breakdown,
        // no diagnostic — and reads as unexplained search exhaustion.
        tracing::warn!(
            skipped_focus_neurons = fingerprint_cache_hits,
            "Issue #1781: every focus neuron was unchanged — whole pass skipped, \
             recorded as fingerprint_unchanged"
        );
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: false,
            cancelled: false,
            memory_pressure_cancelled: false,
            gpu_wedged: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
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
            gpu_wedged: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
        });
    }

    // Issue #1931: the GPU circuit breaker (Issue #1930) has tripped, so the
    // GPU analyses cannot run for the rest of this process. Skip them and exit
    // normally with a signalled partial result rather than propagating an error
    // that would discard the CPU-side accounting below and read to the host as
    // one more failed attempt. Checked before `gpu_is_available()` because a
    // wedged device usually still enumerates as an adapter.
    if super::gpu::breaker::gpu_wedged_skip_reason().is_some() {
        return Ok(AnalyzeAllResult {
            synapse: None,
            neuron: None,
            memory_budget_exceeded: false,
            cancelled: false,
            memory_pressure_cancelled: false,
            gpu_wedged: true,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_gpu_wedged(
                fingerprint_cache_hits,
                effective_focus_neurons.len(),
            ),
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
            gpu_wedged: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
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

    // Issue #1408: Reserve a guaranteed minimum window for synapse/neuron
    // analysis so focus selection and parquet loading cannot starve it. If the
    // shared deadline has already been so consumed that less than the usable
    // hard floor remains for analysis, fail fast with an actionable error
    // instead of running analysis that completes 0 of N targets.
    let analysis_reserve_ms = crate::config::analysis_reserve_ms();
    let analysis_reserve_fraction = crate::config::analysis_reserve_fraction();
    let now_abs_ms = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64);
    if let Some(remaining_ms) = utils::analysis_reserve_shortfall_ms(
        shared_deadline_abs_ms,
        now_abs_ms,
        analysis_reserve_ms,
        analysis_reserve_fraction,
    ) {
        return Err(anyhow::anyhow!(
            "discovery budget exhausted: only {remaining_ms}ms remain for synapse/neuron \
             analysis after focus selection and parquet loading, below the {floor}ms minimum \
             (Issue #1408). Increase discoveryAnalysisTimeoutMinutes, or set \
             NEAT_AI_DISCOVERY_ANALYSIS_RESERVE_MS=0 to opt out of the reserve.",
            floor = utils::ANALYSIS_RESERVE_HARD_FLOOR_MS,
        ));
    }

    // Pre-load ALL records from parquet in one pass. This is MUCH faster than
    // lazy-loading each neuron separately (1 scan vs ~2000 scans for large creatures).
    // Issue #648: Pass the analysis deadline so loading can abort early if time runs out.
    // Issue #1408: Curtail loading at `overall_deadline - reserve` so the reserved
    // analysis window survives even if parquet loading is slow.
    let loading_deadline = utils::reserved_loading_deadline(
        overall_deadline,
        analysis_reserve_ms,
        analysis_reserve_fraction,
    );
    let parquet_loading_start = std::time::Instant::now();
    // Issue #3176: honour the supplied #1567 analysis memory budget so the cache
    // pre-load makes the same eager-vs-lazy decision as focus ranking (corrected
    // available-memory accounting + budget), rather than the divergent
    // 50%-of-total-RAM heuristic that forced eager-capable hosts onto lazy mode.
    let cache_result = cache::RecordCache::new_adaptive_with_deadline_and_budget(
        &input.parquet_file,
        loading_deadline,
        input.max_analysis_memory_mb,
    );

    // GRQ #4068: projected ≫ budget — skip analysis rather than entering an
    // unworkable lazy path that can sit silent past the logical deadline.
    let cache_result = match cache_result {
        Ok(None) => {
            tracing::warn!(
                "analysis phase skipped: projected pre-load unworkable relative to \
                 memory budget (GRQ #4068)"
            );
            return Ok(AnalyzeAllResult {
                synapse: None,
                neuron: None,
                memory_budget_exceeded: false,
                cancelled: true,
                memory_pressure_cancelled: false,
                gpu_wedged: false,
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
                pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(
                    fingerprint_cache_hits,
                ),
            });
        }
        other => other,
    };

    // Issue #1047: If parquet loading was cancelled, return a clean partial
    // result instead of propagating the error.
    let shared_cache = match cache_result {
        Ok(Some(c)) => Arc::new(c),
        Ok(None) => unreachable!("SkipUnworkable handled above"),
        Err(_e) if crate::cancellation::is_cancelled() => {
            tracing::info!("parquet loading cancelled by host — returning empty result");
            return Ok(AnalyzeAllResult {
                synapse: None,
                neuron: None,
                memory_budget_exceeded: false,
                cancelled: true,
                memory_pressure_cancelled: crate::cancellation::is_memory_pressure_cancelled(),
                gpu_wedged: false,
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
                pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(
                    fingerprint_cache_hits,
                ),
            });
        }
        Err(e) => return Err(e).context("failed to load parquet record cache for analysis"),
    };
    // Issue #1409: capture the parquet reload duration for the consolidated
    // per-cycle deadline-consumption breakdown.
    let parquet_reload_ms = parquet_loading_start.elapsed().as_millis() as u64;
    profile.record_phase("parquet_loading", parquet_reload_ms);
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
            gpu_wedged: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
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
            gpu_wedged: false,
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
            module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
            pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
        });
    }

    // Issue #1444: Fail-fast insufficient-recording gate. When the record phase
    // times out, the selected focus neurons can have zero Parquet rows, so the
    // full synapse/neuron analysis is guaranteed to return nothing yet still
    // burns the entire analysis budget. Detect that cheaply here (an in-memory
    // record-count scan of the just-loaded cache) and skip the wasted GPU work,
    // surfacing `insufficient_recording` as the dominant rejection reason in the
    // performance-summary metadata. Disabled via
    // NEAT_AI_DISCOVERY_INSUFFICIENT_RECORDING_FRACTION=0.
    if let Some(fraction) = crate::config::insufficient_recording_fraction() {
        let coverage = super::insufficient_recording::assess_focus_recording_coverage(
            &shared_cache,
            &effective_focus_neurons,
        );
        if super::insufficient_recording::is_insufficient_recording(&coverage, fraction) {
            let diagnostic = super::insufficient_recording::InsufficientRecordingDiagnostic {
                focus_neurons_total: coverage.focus_neurons_total,
                focus_neurons_with_zero_rows: coverage.focus_neurons_with_zero_rows,
                focus_neuron_records_total: coverage.focus_neuron_records_total,
                records_processed: shared_cache.loaded_record_count(),
                threshold_fraction: fraction,
            };
            tracing::warn!(
                focus_neurons_total = diagnostic.focus_neurons_total,
                focus_neurons_with_zero_rows = diagnostic.focus_neurons_with_zero_rows,
                records_processed = diagnostic.records_processed,
                threshold_fraction = fraction,
                "Issue #1444: insufficient Parquet recording for the selected focus neurons — \
                 skipping synapse/neuron analysis to avoid spending the full budget on a \
                 guaranteed-empty pass (record phase likely timed out)"
            );
            let synapse = include_synapse.then(|| {
                super::insufficient_recording::synapse_skip_result(
                    &diagnostic,
                    &effective_focus_neurons,
                )
            });
            let neuron = include_neuron.then(|| {
                super::insufficient_recording::neuron_skip_result(
                    &diagnostic,
                    &effective_focus_neurons,
                )
            });
            return Ok(AnalyzeAllResult {
                synapse,
                neuron,
                memory_budget_exceeded: false,
                cancelled: false,
                memory_pressure_cancelled: false,
                gpu_wedged: false,
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
                pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(
                    fingerprint_cache_hits,
                ),
            });
        }
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
            discovery_outcome_log: input.discovery_outcome_log.clone(),
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
            discovery_outcome_log: input.discovery_outcome_log.clone(),
            // Issue #1319: thread the task descriptor down so the neuron
            // post-processing can bias per-class allocation under OneHot.
            task_descriptor: Some(task_descriptor),
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
    let (synapse_result, neuron_result, dispatch_durations) = match dispatch_result {
        Ok(results) => results,
        Err(_) if crate::cancellation::is_cancelled() => {
            tracing::info!("analysis dispatch cancelled by host — returning empty result");
            return Ok(AnalyzeAllResult {
                synapse: None,
                neuron: None,
                memory_budget_exceeded: false,
                cancelled: true,
                memory_pressure_cancelled: crate::cancellation::is_memory_pressure_cancelled(),
                gpu_wedged: false,
                neuron_fingerprints: Some(current_fingerprints),
                fingerprint_cache_hits,
                fingerprint_cache_misses,
                module_outcome_tracker: input.module_outcome_tracker.clone().unwrap_or_default(),
                pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(
                    fingerprint_cache_hits,
                ),
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
    let base_tracker = input.module_outcome_tracker.clone().unwrap_or_default();

    // Issue #1132: Compute creature-level discovery mode from the caller-supplied
    // rolling outcome log. When the rolling success rate has fallen below the
    // configured threshold, bias the module tracker toward low-risk change types
    // before it is consumed by downstream allocation and boost passes.
    let outcome_log = input.discovery_outcome_log.clone().unwrap_or_default();
    let mode_decision = super::discovery_mode::decide_mode_with_escalation(
        &outcome_log,
        crate::config::low_success_rate_threshold(),
        crate::config::conservative_mode_max_epochs(),
    );
    let rolling_success_rate = mode_decision.rolling_success_rate;
    let discovery_mode = mode_decision.mode;
    if discovery_mode == super::discovery_mode::DiscoveryMode::Conservative
        && utils::verbose_enabled()
    {
        tracing::info!(
            rolling_success_rate,
            "Issue #1132: conservative discovery mode engaged — biasing module weights \
             toward low-risk change types"
        );
    }
    if mode_decision.is_extended_drought() {
        tracing::info!(
            rolling_success_rate,
            trailing_failure_streak = mode_decision.trailing_failure_streak,
            max_conservative_epochs = crate::config::conservative_mode_max_epochs(),
            "Issue #1803: extended drought — conservative risk bias reverted to normal, \
             expensive discovery modules kept escalated"
        );
    }
    let mut tracker = match discovery_mode {
        super::discovery_mode::DiscoveryMode::Normal => base_tracker,
        super::discovery_mode::DiscoveryMode::Conservative => {
            super::discovery_mode::biased_tracker_for_conservative_mode(&base_tracker)
        }
    };

    // Issue #1547: Creature-scale module tiering re-enables the full discovery
    // module set whenever the creature is in a drought / novelty-escalation pass.
    // The low rolling success rate that drives novelty escalation (#1423) and
    // drought escape (#1422) is the signal — available before dispatch — used to
    // keep every expensive module running while the creature is struggling.
    //
    // Issue #1803: this used to read `discovery_mode == Conservative`, so the
    // #1132 risk-bias cooldown also tiered out the expensive modules once the
    // failure streak passed `conservative_mode_max_epochs` — narrowing the module
    // set exactly when the drought was worst. `module_escalation_active` tracks
    // only the collapsed-rate condition, so breadth now outlives the bias.

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
                            cost_hint,
                            task_descriptor,
                            &mode_decision,
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

        // Issue #224: Candidate clustering to reduce redundant ablation tests.
        if let Some(syn) = synapse_result.as_mut() {
            module_dispatch_specs::cluster_synapse_candidates(syn, &input.creature);
        }
    } // end if !memory_budget_exceeded (Issue #1028)

    // Issue #1530: Make the propagation-aware remove-neuron estimator the source
    // of truth for the emitted gain. Milestone #1516 merged
    // `estimate_remove_neuron_gain` (PR #1523) but nothing in the live pipeline
    // invoked it, so the reported remove-neuron gain was still the fabricated
    // NEAT-AI #2483 placeholder (+0.17879 on creature 45a04ef1, versus a
    // measured -0.00032). Override every single-op RemoveNeuron candidate's gain
    // with the honest, propagation-aware estimate before the drought demotion
    // and final gain floor act on it, so downstream scoring, sorting, and
    // filtering all operate on the honest value rather than the placeholder.
    if let Some(syn) = synapse_result.as_mut() {
        let overridden = super::discovery_dispatch::apply_honest_remove_neuron_gain(
            &input.creature,
            &mut syn.coordinated_structural_candidates,
        );
        if overridden > 0 {
            tracing::debug!(
                overridden,
                "Issue #1530: replaced {overridden} remove-neuron candidate gain(s) with the \
                 propagation-aware estimate"
            );
        }
    }

    // Issue #1689: Wire the variance-aware weight-redistribution compensation
    // (#1559) into the live path. Milestone #1559 built the compensation API but
    // nothing invoked it, so emitted remove-neuron candidates carried no remedy
    // and the applier fell back to the mean-only bias fold, which regresses the
    // removed neuron's per-sample (variance) signal (the #1558/#1686 failure
    // class). For each sole-op RemoveNeuron candidate that removes a
    // variance-carrying neuron, attach counterfactual (d): the optimal weight
    // bump into a correlated survivor plus the covariance sufficient statistic.
    // Constant neurons route to the #1623 bias fold and are left untouched here.
    if let Some(syn) = synapse_result.as_mut() {
        // Gather the per-sample records for every neuron a bare remove-neuron
        // candidate needs — the candidate itself and each shared-target survivor
        // — from the record cache in one pass.
        let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
        for candidate in &syn.coordinated_structural_candidates {
            if let [crate::CoordinatedStructuralOpJson::RemoveNeuron { neuron_uuid }] =
                candidate.operations.as_slice()
            {
                needed.insert(neuron_uuid.clone());
                for shared in super::shared_downstream_targets(&input.creature, neuron_uuid) {
                    needed.insert(shared.survivor_uuid);
                }
            }
        }

        let mut records: Vec<crate::types::DiscoverRecord> = Vec::new();
        let mut missing = 0usize;
        for uuid in &needed {
            match shared_cache.get(uuid) {
                Ok(neuron_records) => records.extend(neuron_records.iter().cloned()),
                // No per-sample records for this neuron — counterfactual (d)
                // cannot be evaluated for candidates that need it, so the remedy
                // is legitimately absent (the applier flags such removals rather
                // than folding the mean). Count it for observability rather than
                // masking it.
                Err(_) => missing += 1,
            }
        }
        if missing > 0 {
            tracing::debug!(
                missing,
                "Issue #1689: {missing} neuron(s) had no cached per-sample records — \
                 compensation left absent for candidates that depend on them"
            );
        }

        let attached = super::discovery_dispatch::apply_remove_neuron_compensation(
            &input.creature,
            &records,
            &mut syn.coordinated_structural_candidates,
        );
        if attached > 0 {
            tracing::debug!(
                attached,
                "Issue #1689: attached variance-aware weight-redistribution compensation to \
                 {attached} remove-neuron candidate(s)"
            );
        }

        // Issue #1690: Wire the constant-neuron bias fold (#1623) into the live
        // path, alongside the #1559 redistribution above. A genuinely-constant
        // neuron carries no per-sample variance, so a plain bias fold is fully
        // compensable — no survivor redistribution is needed. For each sole-op
        // RemoveNeuron candidate whose neuron is *measured* functionally constant
        // (Issue #1779 — the declared `"constant"` class never reaches here, as
        // every producer emits hidden neurons), evaluate the fold behind the
        // evaluate-before-accept gate and emit the folded per-target bias deltas
        // so the applier folds the constant contribution into downstream biases
        // rather than folding a mean. Routing is mutually exclusive with the
        // redistribution path: a looks-constant or no-records candidate is
        // rejected fail-loud (no fold emitted, never deleted blind), and the same
        // records gathered above are reused.
        let folded = super::discovery_dispatch::apply_constant_neuron_bias_fold(
            &input.creature,
            &records,
            &mut syn.coordinated_structural_candidates,
        );
        if folded > 0 {
            tracing::debug!(
                folded,
                "Issue #1690: attached constant-neuron bias fold to {folded} \
                 remove-neuron candidate(s)"
            );
        }
    }

    // Issue #1448: Deprioritise destructive remove-neuron candidates during a
    // search-exhaustion drought. On a plateaued dense creature the remove-neuron
    // path dominates the failure cache (bucket `247b83ab`) with low-impact
    // proposals that never pass scoring, starving the constructive modules. When
    // the trailing-failure streak marks an active, search-exhausted drought
    // (the #1424/#1421 classification — environmental droughts are left alone),
    // demote single-op remove-neuron gains so they sort below add-synapse /
    // squash / multi-op coordinated candidates and the most over-confident ones
    // fall through the noise floor applied immediately below. Runs before the
    // final gain floor so the demoted gains are screened in the same pass.
    if let Some(syn) = synapse_result.as_mut() {
        let drought_threshold = super::drought_diagnostic::drought_threshold_for_task(
            crate::config::drought_log_threshold(),
            &task_descriptor,
        );
        let deprioritisation_inputs = super::remove_neuron_drought::DroughtDeprioritisationInputs {
            consecutive_failures: outcome_log.consecutive_trailing_failures(),
            environmentally_disabled_passes: outcome_log.environmentally_disabled_passes,
            drought_threshold,
        };
        let factor = super::remove_neuron_drought::remove_neuron_deprioritisation_factor(
            &deprioritisation_inputs,
            crate::config::remove_neuron_drought_factor(),
        );
        let demoted = super::remove_neuron_drought::deprioritise_remove_neuron_candidates(
            &mut syn.coordinated_structural_candidates,
            factor,
        );
        if demoted > 0 {
            syn.metadata.rejection_breakdown.record_many_u32(
                super::diagnostics::rejection_reasons::REJECTION_REMOVE_NEURON_DROUGHT_DEPRIORITISED,
                demoted,
            );
            tracing::warn!(
                demoted,
                factor,
                drought_threshold,
                consecutive_failures = deprioritisation_inputs.consecutive_failures,
                "Issue #1448: search-exhaustion drought — deprioritised {demoted} \
                 remove-neuron candidate(s) by {factor}× in favour of constructive \
                 change types"
            );
        }
    }

    // Issue #1622: Promote flagged functionally-constant hidden neurons to
    // priority remove-neuron candidates. A zero-variance neuron gets an honest
    // gain of ≈0 (#1518) and is further demoted during a drought (#1448), so it
    // is never selected even though removing it is harmless (its contribution
    // folds into its targets' biases). The constant-neuron detector (a sibling
    // sub-issue of the #1620 milestone) flags such neurons; this overrides the
    // flagged candidate's gain with the priority marker AFTER the honest-gain
    // override and drought demotion (bypassing both) and BEFORE the final gain
    // floor (so the promoted candidate survives). Two flag sources are unioned:
    // the structural detector (#1813 — a hidden neuron whose output cannot vary
    // given the topology, which needs no recorded activations) and the measured
    // #1779 source (every candidate that just received an accepted #1623 bias
    // fold above, i.e. the same evaluate-before-accept verification this module's
    // safety argument rests on). Without either, the honest gain (≈ −0.75 for a
    // harmless constant neuron) is below the floor and the fold never reaches the
    // consumer at all.
    if let Some(syn) = synapse_result.as_mut() {
        let mut flagged =
            super::remove_neuron_constant_promotion::functionally_constant_neuron_uuids(
                &input.creature,
            );
        let structural = flagged.len();
        let measured_flags =
            super::remove_neuron_constant_promotion::bias_folded_constant_neuron_uuids(
                &syn.coordinated_structural_candidates,
            );
        let measured = measured_flags.len();
        flagged.extend(measured_flags);
        let promoted =
            super::remove_neuron_constant_promotion::promote_constant_remove_neuron_candidates(
                &mut syn.coordinated_structural_candidates,
                &flagged,
            );
        if promoted > 0 {
            tracing::debug!(
                promoted,
                structural,
                measured,
                "Issue #1622: promoted {promoted} functionally-constant remove-neuron \
                 candidate(s) to priority ({structural} flagged structurally, #1813)"
            );
        }
    }

    // Issue #1110, #1128, #1139: Final coordinated-structural gain floor.
    //
    // MUST run unconditionally — outside the memory/deadline guard — because
    // `pair_coordinated_structural_with_weight_variants` (invoked earlier in
    // synapse post-processing) produces `Gentle Nudge`/`Micro Nudge` variants
    // whose `expected_creature_score_gain` is multiplied by 0.25× / 0.1× and
    // can fall below `COORDINATED_POST_DISCOUNT_NOISE_FLOOR` (5e-7). Running
    // this filter only in the fast-path guard leaked sub-floor variants to
    // the FFI response whenever the memory budget or deadline was exceeded
    // (the production discovery cache at discoveryVersion 0.74.16 captured
    // 1.3e-7 gains damaging creatures). Applying it here ensures the floor holds in both the
    // fast-path and the skipped-post-processing fallback, and refreshes
    // `candidates_returned` + the rejection breakdown in either path.
    if let Some(syn) = synapse_result.as_mut() {
        candidate_aggregation::apply_final_coordinated_gain_floor(
            syn,
            discovery_mode,
            crate::config::conservative_gain_multiplier(),
        );
    }

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

    // Issue #1194: Emit a structured zero-success batch summary so failure
    // clusters in the input failure cache are visible without manual
    // inspection. The aggregation only runs when accepted == 0, keeping the
    // hot path unchanged for successful batches.
    if let Some(cache) = input.failure_cache.as_deref() {
        let _ =
            crate::observability::maybe_emit_zero_success_batch_summary(total_candidates, cache);
    }

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

    // Issue #1132: Expose the creature-level discovery mode and rolling
    // success rate on the response metadata so callers can observe when the
    // pipeline has entered conservative mode.
    if let Some(syn) = synapse_result.as_mut() {
        syn.metadata.discovery_mode = discovery_mode;
        syn.metadata.rolling_success_rate = rolling_success_rate;
    }
    if let Some(neu) = neuron_result.as_mut() {
        neu.metadata.discovery_mode = discovery_mode;
        neu.metadata.rolling_success_rate = rolling_success_rate;
    }

    // Issue #1791: flush this pass's per-target outcomes to the global
    // target-failure tracker. The tracker was read by both preparation layers
    // but never written, so `is_empty()` short-circuited every cooldown filter
    // and the suppression was permanently inert.
    //
    // One flush per pass, under a single lock, merging both modules' verdicts:
    // the tracker is per-target-per-*pass*, and the epoch only advances once per
    // pass (Issue #1790), so recording each module separately would grow a
    // target's streak twice as fast as the cooldown window can expire it.
    let mut pass_target_outcomes: Vec<super::target_pass_outcomes::TargetPassOutcome> = Vec::new();
    let mut pass_candidates = 0usize;
    if let Some(syn) = synapse_result.as_ref() {
        pass_target_outcomes.extend(syn.metadata.target_pass_outcomes.iter().cloned());
        pass_candidates = pass_candidates.saturating_add(syn.metadata.candidates_returned);
    }
    if let Some(neu) = neuron_result.as_ref() {
        pass_target_outcomes.extend(neu.metadata.target_pass_outcomes.iter().cloned());
        pass_candidates = pass_candidates.saturating_add(neu.metadata.candidates_returned);
    }
    let target_cooldown_skipped = pass_target_cooldown_skipped(
        synapse_result.as_ref().map(|syn| &syn.metadata),
        neuron_result.as_ref().map(|neu| &neu.metadata),
    );
    let pass_analysis_outcome = super::AnalysisOutcome::from_pass_flags(
        memory_budget_exceeded,
        crate::cancellation::is_memory_pressure_cancelled(),
        pass_candidates,
    );
    super::target_pass_outcomes::flush_target_pass_outcomes(
        &pass_target_outcomes,
        &pass_analysis_outcome,
    );

    // Issue #1202: Drought diagnostic. When the trailing-failure streak in the
    // caller-supplied outcome log crosses the configured threshold, emit a
    // single structured warn log and attach the payload to both metadata
    // surfaces. The orchestrator owns the only call site, so the warn fires
    // at most once per `analyze_all` invocation.
    let consecutive_failures = outcome_log.consecutive_trailing_failures();
    // Issue #1320: calibrate the drought threshold to the task descriptor.
    // Classification topologies generate sparse per-sample improvement signal,
    // so the base threshold is scaled up to avoid premature drought fires.
    // OTHER / Unknown / Independent descriptors keep the base threshold (the
    // regression guard).
    let drought_threshold = super::drought_diagnostic::drought_threshold_for_task(
        crate::config::drought_log_threshold(),
        &task_descriptor,
    );
    if consecutive_failures >= drought_threshold {
        // Snapshot the global target-cooldown tracker. A poisoned lock is
        // recovered (Issue #1875) so the diagnostic always carries real tracker
        // state rather than silently degrading to "no tracker".
        let tracker_snapshot = super::target_failure_tracker::snapshot_tracker(
            super::target_failure_tracker::global_tracker(),
        );
        let current_epoch = tracker_snapshot.current_epoch();

        // Pick whichever metadata surface has the richer rejection breakdown
        // for the dominant-reason field. Synapse takes precedence when both
        // are present.
        let empty_breakdown = super::diagnostics::RejectionBreakdown::new();
        let rejection_breakdown = synapse_result
            .as_ref()
            .map(|s| &s.metadata.rejection_breakdown)
            .or_else(|| {
                neuron_result
                    .as_ref()
                    .map(|n| &n.metadata.rejection_breakdown)
            })
            .unwrap_or(&empty_breakdown);
        let candidates_returned = synapse_result
            .as_ref()
            .map_or(0, |s| {
                u32::try_from(s.metadata.candidates_returned).unwrap_or(u32::MAX)
            })
            .saturating_add(neuron_result.as_ref().map_or(0, |n| {
                u32::try_from(n.metadata.candidates_returned).unwrap_or(u32::MAX)
            }));

        let inputs = super::drought_diagnostic::DroughtInputs {
            consecutive_failures,
            rolling_success_rate,
            discovery_mode,
            target_tracker: Some(&tracker_snapshot),
            current_epoch,
            // Issue #1791: the real per-phase filter return value, no longer a
            // hard-coded `0`. A permanently-zero value across a fleet is the
            // production signature that tracker population has regressed.
            target_cooldown_skipped,
            rejection_breakdown,
            candidates_returned,
        };
        if let Some(diagnostic) =
            super::drought_diagnostic::emit_drought_diagnostic(&inputs, drought_threshold)
        {
            if let Some(syn) = synapse_result.as_mut() {
                syn.metadata.drought_diagnostic = Some(diagnostic.clone());
            }
            if let Some(neu) = neuron_result.as_mut() {
                neu.metadata.drought_diagnostic = Some(diagnostic);
            }
        }

        // Issue #1205: Operator escape hatch — after the drought diagnostic
        // has surfaced, optionally force a one-shot reset of the active target
        // cooldowns. Driven by
        // `NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS`; disabled when unset.
        if let Some(drought_reset_after) = crate::config::drought_reset_after_epochs() {
            // Re-lock the global tracker so we can mutate it. The earlier
            // snapshot for the diagnostic was a clone; the reset must land on
            // the actual global state. A poisoned lock is recovered rather than
            // skipped (Issue #1875) — the escape hatch must never no-op
            // silently.
            let _ = super::drought_reset::maybe_perform_drought_reset_locked(
                super::target_failure_tracker::global_tracker(),
                consecutive_failures,
                drought_reset_after,
            );
        }
    } else if let Some(drought_reset_after) = crate::config::drought_reset_after_epochs() {
        // Issue #1205: when consecutive_failures is below the diagnostic
        // threshold (or zero) the lever cannot fire, but we still need to
        // re-arm the tombstone after a successful pass so the next future
        // drought is not skipped.
        let _ = drought_reset_after; // configured value retained for diagnostics; no-op here
        if consecutive_failures == 0 {
            // Issue #1875: a poisoned lock must not silently skip the re-arm.
            super::drought_reset::rearm_drought_reset_locked(
                super::target_failure_tracker::global_tracker(),
            );
        }
    }

    // Issue #1424: Creature-level drought alarm. Independent of the per-pass
    // drought diagnostic above, this emits a single durable alarm when the
    // creature's epochs-since-last-acceptance crosses the configured "weeks"
    // threshold. `genuinely_empty` is the trailing search-exhausted streak;
    // `environmentally_disabled` is the host-gated passes the log accumulated.
    // Their sum is the epochs since the creature last accepted a candidate.
    // Disabled when `NEAT_AI_DISCOVERY_DROUGHT_ALARM_EPOCHS=0`.
    if let Some(alarm_threshold) = crate::config::drought_alarm_epochs() {
        let genuinely_empty = consecutive_failures;
        let environmentally_disabled = outcome_log.environmentally_disabled_passes;
        let epochs_since_last_accepted = genuinely_empty.saturating_add(environmentally_disabled);
        // The FFI creature carries no uuid; derive a stable identity from its
        // persistent output-neuron uuids (Issue #1424).
        let output_uuids: Vec<&str> = input
            .creature
            .neurons
            .iter()
            .filter(|n| n.neuron_type == "output")
            .map(|n| n.uuid.as_str())
            .collect();
        let creature_id = super::creature_drought_alarm::derive_creature_id(&output_uuids);
        let alarm_inputs = super::creature_drought_alarm::CreatureDroughtAlarmInputs {
            creature_uuid: &creature_id,
            epochs_since_last_accepted,
            genuinely_empty_passes: genuinely_empty,
            environmentally_disabled_passes: environmentally_disabled,
        };
        if let Some(alarm) = super::creature_drought_alarm::emit_creature_drought_alarm(
            &alarm_inputs,
            alarm_threshold,
        ) {
            if let Some(syn) = synapse_result.as_mut() {
                syn.metadata.creature_drought_alarm = Some(alarm.clone());
            }
            if let Some(neu) = neuron_result.as_mut() {
                neu.metadata.creature_drought_alarm = Some(alarm);
            }
        }
    }

    // Issue #1409: Emit one consolidated, greppable summary attributing
    // deadline consumption across the analysis phases, plus an explicit STARVED
    // warning when synapse/neuron analysis was curtailed by the deadline. The
    // focus phase (parquet load + focus ranking) runs in the separate
    // rank_focus_neurons call and is surfaced there (Issue #1377).
    use super::deadline_breakdown::{DeadlineConsumptionBreakdown, PhaseCompletion};
    let breakdown = DeadlineConsumptionBreakdown {
        parquet_reload_ms,
        synapse_analysis_ms: dispatch_durations.synapse_ms,
        neuron_analysis_ms: dispatch_durations.neuron_ms,
        total_analysis_ms: total_timer.elapsed_ms(),
        synapse: synapse_result.as_ref().map(|s| PhaseCompletion {
            timed_out: s.metadata.timed_out,
            completed_focus_neurons: s.metadata.completed_focus_neurons,
            total_focus_neurons: s.metadata.total_focus_neurons,
        }),
        neuron: neuron_result.as_ref().map(|n| PhaseCompletion {
            timed_out: n.metadata.timed_out,
            completed_focus_neurons: n.metadata.completed_focus_neurons,
            total_focus_neurons: n.metadata.total_focus_neurons,
        }),
    };
    breakdown.emit();

    Ok(AnalyzeAllResult {
        synapse: synapse_result,
        neuron: neuron_result,
        memory_budget_exceeded,
        cancelled: crate::cancellation::is_cancelled(),
        memory_pressure_cancelled: crate::cancellation::is_memory_pressure_cancelled(),
        gpu_wedged: false,
        neuron_fingerprints: Some(current_fingerprints),
        fingerprint_cache_hits,
        fingerprint_cache_misses,
        module_outcome_tracker: tracker,
        pass_rejection_breakdown: pass_breakdown_with_fingerprint_skips(fingerprint_cache_hits),
    })
}

#[cfg(test)]
mod target_cooldown_tests {
    use super::pass_target_cooldown_skipped;
    use crate::analysis::candidate_starvation::signals_from_breakdown;
    use crate::analysis::diagnostics::rejection_reasons::REJECTION_TARGET_COOLDOWN_SKIPPED;
    use crate::analysis::shared::{NeuronAnalysisMetadata, SynapseAnalysisMetadata};
    use crate::analysis::target_failure_tracker::fold_target_cooldown_skips;

    /// Build a phase metadata surface exactly as the orchestration layers do:
    /// the filter return value is stored on the metadata *and* folded into that
    /// surface's rejection breakdown.
    fn synapse_surface(skipped: u32) -> SynapseAnalysisMetadata {
        let mut metadata = SynapseAnalysisMetadata {
            target_cooldown_skipped: skipped,
            ..Default::default()
        };
        fold_target_cooldown_skips(skipped, &mut metadata.rejection_breakdown);
        metadata
    }

    fn neuron_surface(skipped: u32) -> NeuronAnalysisMetadata {
        let mut metadata = NeuronAnalysisMetadata {
            target_cooldown_skipped: skipped,
            ..Default::default()
        };
        fold_target_cooldown_skips(skipped, &mut metadata.rejection_breakdown);
        metadata
    }

    fn breakdown_count(counts: &crate::analysis::diagnostics::RejectionBreakdown) -> u32 {
        counts
            .counts()
            .get(REJECTION_TARGET_COOLDOWN_SKIPPED)
            .copied()
            .unwrap_or(0)
    }

    /// Issue #1797: a synapse-only pass with K cooldown-skipped targets reports
    /// K in the surfaced breakdown and K as the pass total.
    #[test]
    fn cooldown_skipped_counts_synapse_only() {
        const K: u32 = 3;
        let synapse = synapse_surface(K);

        assert_eq!(breakdown_count(&synapse.rejection_breakdown), K);
        assert_eq!(pass_target_cooldown_skipped(Some(&synapse), None), K);

        // The starvation classifier reads only the breakdown: skipped targets
        // were never analysed, so they are upstream evidence.
        let signals = signals_from_breakdown(&synapse.rejection_breakdown, 0);
        assert_eq!(signals.upstream_rejections, K);
        assert_eq!(signals.gate_side_rejections, 0);
    }

    /// A neuron-only pass reports its own count and nothing from the absent
    /// synapse surface.
    #[test]
    fn cooldown_skipped_counts_neuron_only() {
        const K: u32 = 4;
        let neuron = neuron_surface(K);

        assert_eq!(breakdown_count(&neuron.rejection_breakdown), K);
        assert_eq!(pass_target_cooldown_skipped(None, Some(&neuron)), K);

        let signals = signals_from_breakdown(&neuron.rejection_breakdown, 0);
        assert_eq!(signals.upstream_rejections, K);
    }

    /// Both surfaces filter their own focus order, so the pass total is the sum
    /// — and each surface's breakdown carries only its own skips.
    #[test]
    fn cooldown_skipped_counts_both_no_double_count() {
        const SYNAPSE_SKIPPED: u32 = 3;
        const NEURON_SKIPPED: u32 = 2;
        let synapse = synapse_surface(SYNAPSE_SKIPPED);
        let neuron = neuron_surface(NEURON_SKIPPED);

        assert_eq!(
            breakdown_count(&synapse.rejection_breakdown),
            SYNAPSE_SKIPPED
        );
        assert_eq!(breakdown_count(&neuron.rejection_breakdown), NEURON_SKIPPED);
        assert_eq!(
            pass_target_cooldown_skipped(Some(&synapse), Some(&neuron)),
            SYNAPSE_SKIPPED + NEURON_SKIPPED
        );
        // Neither surface absorbed the other's skips.
        assert_eq!(synapse.rejection_breakdown.total(), SYNAPSE_SKIPPED);
        assert_eq!(neuron.rejection_breakdown.total(), NEURON_SKIPPED);
    }

    /// A pass with no cooldown skips leaves the reason key absent rather than
    /// present-and-zero, and totals zero.
    #[test]
    fn cooldown_skipped_absent_when_no_targets_dropped() {
        let synapse = synapse_surface(0);
        let neuron = neuron_surface(0);
        assert!(synapse.rejection_breakdown.is_empty());
        assert!(neuron.rejection_breakdown.is_empty());
        assert_eq!(
            pass_target_cooldown_skipped(Some(&synapse), Some(&neuron)),
            0
        );
        assert_eq!(pass_target_cooldown_skipped(None, None), 0);
    }
}
