//! Top-level analysis orchestration (Issue #562).
//!
//! Contains the `analyze_all` entry point and its supporting helpers:
//! - `choose_deadline_order_synapse_first` — randomised analysis ordering
//! - `run_optional_analysis` — guarded analysis phase execution

use std::sync::Arc;
use std::time::SystemTime;

use anyhow::Result;

use crate::observability::{
    PhaseTimer, ProfileData, ProfileMode, global_gpu_metrics, profile_mode,
    report_global_gpu_metrics,
};
use crate::{AnalyzeAllInput, AnalyzeNeuronsInput, AnalyzeSynapsesInput};

use super::{
    AnalyzeAllResult, cache, candidate_aggregation, module_dispatch_specs, neuron,
    neuron_fingerprint, synapse, utils,
};

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
pub(crate) fn choose_deadline_order_synapse_first(random_seed: Option<u64>, now_ms: u64) -> bool {
    // A tiny, deterministic "coin flip": parity of (seed XOR time).
    //
    // This is good enough for long-run fairness (50/50 over time) and is easy to test.
    ((random_seed.unwrap_or(0) ^ now_ms) & 1) == 0
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
/// When `analysis_deadline_ms` is set and both analyses are enabled, the library **randomises
/// the run order** on each invocation. This means one run may return only synapse candidates
/// (neuron starved) and the next may return only neuron candidates (synapse starved).
///
/// When no deadline is set, the original "neuron-first" ordering is preserved for
/// backwards compatibility (neuron discovery creates new network structure and may be
/// considered higher value when time is not constrained).
#[tracing::instrument(skip_all, fields(focus_neurons = input.focus_neurons.len()))]
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
            neuron_fingerprints: Some(current_fingerprints),
            fingerprint_cache_hits,
            fingerprint_cache_misses,
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
            tracing::debug!(
                deadline_ms = input.analysis_deadline_ms.unwrap_or(0),
                first = if synapse_first { "synapse" } else { "neuron" },
                "deadline set — randomised analysis ordering"
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
    let mut synapse_result = synapse_result;
    let mut neuron_result = neuron_result;

    if let (Some(syn), Some(neuron)) = (synapse_result.as_mut(), neuron_result.as_mut()) {
        candidate_aggregation::convert_neurons_to_coordinated_replacements(
            input,
            syn,
            neuron,
            &shared_cache,
        );
    }

    // Issue #375 / Issue #419: Discovery module dispatch using parallel pattern.
    if let Some(syn) = synapse_result.as_mut() {
        let max_candidates = input.max_synapse_candidates;
        let diversify = input.analysis_deadline_ms.is_some();

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

        module_dispatch_specs::dispatch_and_merge_discovery_modules(
            syn,
            &creature,
            &hidden_neurons,
            &shared_cache,
            max_candidates,
            diversify,
        );
    }

    // Issue #489: Cross-module candidate deduplication.
    if let Some(syn) = synapse_result.as_mut() {
        module_dispatch_specs::deduplicate_cross_module_candidates(syn);
    }

    // Issue #224: Candidate clustering to reduce redundant ablation tests.
    if let Some(syn) = synapse_result.as_mut() {
        module_dispatch_specs::cluster_synapse_candidates(syn, &input.creature);
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
