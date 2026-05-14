//! Synapse analysis orchestration
//!
//! Contains the core `analyze_synapses_with_cache_impl` function that coordinates
//! the multi-phase synapse analysis pipeline: building creature lookups, setting
//! up focus targets, running parallel per-target analysis, and assembling results.

use crate::AnalyzeSynapsesInput;
use anyhow::{Context, Result};

use crate::analysis::diagnostics::{TargetDiagnostics, require_unique_focus};
use crate::analysis::gpu::{GpuAnalyzer, GpuWorkQueue};
use crate::analysis::shared::AnalyzeSynapsesResult;
use crate::analysis::utils::{
    build_deadline, deadline_passed, log_analysis_start, order_focus_targets,
};

use super::metadata::AtomicMetadata;
use super::preparation;
use super::results::{FinaliseParams, finalise_synapse_results};
use super::target_analysis;
use crate::analysis::cache::RecordCache;

use super::metadata::MergedResults;

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Internal implementation of synapse analysis with cache.
/// This is called from the public API functions below.
///
/// The `gpu_queue` parameter is mandatory - callers should create it once and reuse it
/// across multiple calls for better performance (avoids ~100ms initialisation overhead).
pub(crate) fn analyze_synapses_with_cache_impl(
    input: &AnalyzeSynapsesInput,
    cache: Arc<RecordCache>,
    gpu_queue: Arc<GpuWorkQueue>,
) -> Result<AnalyzeSynapsesResult> {
    // Phase 1: Build creature lookup maps
    let lookups = preparation::build_creature_lookups(input);

    // Phase 2: Focus target and diagnostics setup
    let unique_focus = require_unique_focus(&input.focus_neurons, "analyse_synapses")
        .context("synapse analysis input validation failed")?;
    let diagnostics = Arc::new(TargetDiagnostics::new(&unique_focus));
    let timing_collector = Arc::new(crate::analysis::shared::TimingCollector::new(
        crate::analysis::utils::gpu_timing_enabled(),
    ));
    let deadline = build_deadline(input.analysis_deadline_ms);

    let mut focus_order: Vec<String> = unique_focus.iter().map(|s| (*s).clone()).collect();
    let focus_neuron_type_map: HashMap<&str, &str> = input
        .creature
        .neurons
        .iter()
        .map(|n| (n.uuid.as_str(), n.neuron_type.as_str()))
        .collect();
    order_focus_targets(&mut focus_order, input.random_seed, &focus_neuron_type_map);

    // Issue #1130: drop targets currently in cooldown after focus filtering,
    // before we incur any per-target analysis cost.
    // Issue #1204: thread the discovery outcome log so cooldown relaxes during
    // a drought when mode/drought signals are available.
    let _cooldown_skipped =
        apply_target_cooldown(&mut focus_order, input.discovery_outcome_log.as_ref());

    log_analysis_start(
        "synapse",
        input.analysis_deadline_ms,
        focus_order.len(),
        &focus_order,
    );

    let total_focus_count = focus_order.len();
    let completed_count = Arc::new(AtomicUsize::new(0));

    // GPU is required for analysis. Return a structured error instead of panicking
    // so the TypeScript layer receives a JSON response it can handle gracefully.
    if !GpuAnalyzer::gpu_is_available() {
        return Err(crate::ffi_types::DiscoveryError::GpuUnavailable {
            reason: "No compatible GPU adapter found on this system".to_string(),
        }
        .into());
    }

    // Phase 3: Shared atomic state for parallel processing (lock-free)
    let metadata = Arc::new(AtomicMetadata::new());
    let analysis_timed_out = Arc::new(AtomicBool::new(false));

    // Phase 4: Compute constant-source threshold
    let constant_source_effect_threshold =
        preparation::compute_constant_source_threshold_from_cache(input, cache.as_ref());

    // Phase 5: Build shared context for per-target analysis
    let acceptance_tracker = Arc::new(std::sync::Mutex::new(
        crate::analysis::synapse::adaptive_proposal::AcceptanceTracker::new(),
    ));
    // Issue #1021: MCMC diagnostics tracker for acceptance rate and diversity metrics
    let mcmc_tracker =
        Arc::new(crate::analysis::diagnostics::mcmc_diagnostics::McmcDiagnosticsTracker::new());
    // Issue #1164: within-batch target-failure short-circuit. Lives for the
    // duration of this orchestration call only.
    let within_batch_failures =
        Arc::new(crate::analysis::within_batch_failures::WithinBatchFailureTracker::new());
    let ctx = Arc::new(target_analysis::TargetAnalysisContext {
        ordered_neurons: Arc::new(lookups.ordered_neurons),
        order_map: Arc::new(lookups.order_map),
        neuron_index: Arc::new(lookups.neuron_index),
        existing_synapses: Arc::new(lookups.existing_synapses),
        existing_synapse_weights: Arc::new(lookups.existing_synapse_weights),
        synapses_by_target: Arc::new(lookups.synapses_by_target),
        neuron_squash_map: Arc::new(lookups.neuron_squash_map),
        neuron_type_map: Arc::new(lookups.neuron_type_map),
        input_neuron_uuids: Arc::new(lookups.input_neuron_uuids),
        used_inputs: Arc::new(lookups.used_inputs),
        neuron_bias_map: Arc::new(lookups.neuron_bias_map),
        constant_source_effect_threshold,
        diagnostics: diagnostics.clone(),
        timing_collector: timing_collector.clone(),
        deadline,
        threshold: 0.0,
        acceptance_tracker,
        temperature: input.temperature,
        mcmc_tracker: mcmc_tracker.clone(),
        within_batch_failures: within_batch_failures.clone(),
    });

    // Phase 6: Process each focus neuron in parallel — thread-local collection (Issue #744)
    //
    // Each thread returns its own TargetAnalysisResults, avoiding mutex contention.
    // Results are merged in a single-threaded pass after the parallel section.
    let per_target_results: Vec<Option<target_analysis::TargetAnalysisResults>> = focus_order
        .par_iter()
        .map(|target_uuid| -> Result<Option<target_analysis::TargetAnalysisResults>> {
            if analysis_timed_out.load(Ordering::Relaxed) || deadline_passed(&deadline) {
                analysis_timed_out.store(true, Ordering::Relaxed);
                return Ok(None);
            }

            let target_results = target_analysis::analyse_single_target(
                target_uuid,
                input,
                cache.as_ref(),
                &gpu_queue,
                &ctx,
            )
            .with_context(|| {
                format!("failed during synapse analysis for target neuron {target_uuid}")
            })?;

            // Update lock-free atomic metadata
            metadata.merge_atomic(&target_results);

            let completed =
                completed_count.fetch_add(1, Ordering::Relaxed) + 1;
            crate::watchdog::beat(format!(
                "synapse analysis → completed {completed}/{total_focus_count} (last target {target_uuid})"
            ));
            Ok(Some(target_results))
        })
        .collect::<Result<Vec<_>>>()?;

    // Issue #1164: surface the within-batch short-circuit savings.
    let within_batch_skips = within_batch_failures.skip_count();
    if within_batch_skips > 0 {
        tracing::info!(
            phase = "synapse",
            within_batch_skipped = within_batch_skips,
            failed_targets = within_batch_failures.failed_target_count(),
            failure_limit = within_batch_failures.failure_limit(),
            "Short-circuited same-target candidates after within-batch failure (Issue #1164)"
        );
    }

    // Phase 7: Single-threaded merge (fast, no contention)
    let collectors = MergedResults::from_per_target(
        per_target_results,
        analysis_timed_out.load(Ordering::Relaxed),
        metadata,
    );

    // Phase 8: Collect results and build final output
    // Issue #1165: scan the failure cache for prediction-vs-actual calibration
    // mismatches and emit structured diagnostic logs + per-entry records on
    // the MCMC tracker before the summary is built.
    if let Some(cache) = input.failure_cache.as_deref() {
        mcmc_tracker.record_calibration_misses_from_cache(
            cache,
            crate::config::calibration_miss_threshold(),
        );
    }
    let mcmc_summary = mcmc_tracker.build_summary();
    finalise_synapse_results(FinaliseParams {
        collectors,
        completed_count,
        total_focus_count,
        diagnostics,
        timing_collector,
        input,
        cache,
        order_map: &ctx.order_map,
        mcmc_summary,
    })
}

/// Drop focus targets in cooldown via the global target-failure tracker
/// (Issue #1130). Returns the number of targets removed so callers can include
/// it in diagnostics alongside the `cooldown_skipped` reason-name convention
/// from Issue #1129.
fn apply_target_cooldown(
    focus_order: &mut Vec<String>,
    discovery_outcome_log: Option<&crate::analysis::discovery_mode::DiscoveryOutcomeLog>,
) -> u32 {
    use crate::analysis::target_failure_tracker::{
        filter_cooldown_targets, filter_cooldown_targets_adaptive, global_tracker,
    };

    let tracker_lock = match global_tracker().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    if tracker_lock.is_empty() {
        return 0;
    }
    let current_epoch = tracker_lock.current_epoch();
    match discovery_outcome_log {
        Some(log) if !log.is_empty() => {
            let mode = crate::analysis::discovery_mode::decide_mode(
                log,
                crate::config::low_success_rate_threshold(),
                crate::config::conservative_mode_max_epochs(),
            );
            let drought_failures = log.consecutive_trailing_failures();
            filter_cooldown_targets_adaptive(
                focus_order,
                &tracker_lock,
                current_epoch,
                mode,
                drought_failures,
            )
        }
        _ => filter_cooldown_targets(focus_order, &tracker_lock, current_epoch),
    }
}
