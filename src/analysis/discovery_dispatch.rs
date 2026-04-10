//! Generic discovery module dispatch pattern (Issue #375).
//!
//! Extracts the repeated detect → convert → log → merge boilerplate from
//! `analyze_all()` into a single reusable function. Each discovery module
//! supplies its detection and conversion logic via closures, while this
//! module handles watchdog beats, phase timing, verbose logging, and
//! merging into the synapse result.
//!
//! ## Parallel dispatch (Issue #419)
//!
//! `run_discovery_modules_parallel()` runs all detection phases concurrently
//! via `rayon::into_par_iter()`, then merges results sequentially. This
//! preserves deterministic ordering while utilising multiple CPU cores.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use crate::CoordinatedStructuralCandidateJson;
use crate::observability::PhaseTimer;
use rayon::prelude::*;

use super::constants::{
    MODULE_GATE_THRESHOLD, QUALITY_SKIP_GAIN_THRESHOLD, QUALITY_SKIP_MIN_CANDIDATES,
    SOFT_FAILURE_WEIGHT,
};
use super::module_weights::{DiscoveryModuleStatsJson, ModuleOutcomeTracker};
use super::shared;
use super::utils;

/// Result of a discovery module's detection phase.
///
/// The `detected_count` is used for verbose logging (how many raw detections),
/// while `candidates` are the coordinated structural candidates to merge.
pub struct DiscoveryDetectionResult {
    pub detected_count: usize,
    pub candidates: Vec<CoordinatedStructuralCandidateJson>,
}

/// Run a single discovery detection module with the standard dispatch pattern:
///
/// 1. Watchdog beat (starting)
/// 2. `PhaseTimer` creation
/// 3. Call `detect_fn` which runs module-specific detection and conversion
/// 4. Verbose logging if results are non-empty
/// 5. Merge into synapse result via `merge_coordinated_structural_replacements`
/// 6. Watchdog beat (finished)
///
/// The `detect_fn` closure encapsulates all module-specific logic (record
/// collection, detection, and conversion to coordinated candidates).
#[tracing::instrument(skip_all, fields(module = module_name, phase = phase_name))]
pub fn run_discovery_module(
    syn: &mut shared::AnalyzeSynapsesResult,
    module_name: &str,
    phase_name: &'static str,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    detect_fn: impl FnOnce() -> Option<DiscoveryDetectionResult>,
) {
    let starting = format!("analysis::analyze_all → {module_name} starting");
    let finished = format!("analysis::analyze_all → {module_name} finished");

    crate::watchdog::beat(&starting);
    let _timer = PhaseTimer::new(phase_name);

    if let Some(result) = detect_fn()
        && !result.candidates.is_empty()
    {
        if utils::verbose_enabled() {
            tracing::debug!(
                module = module_name,
                detections = result.detected_count,
                candidates = result.candidates.len(),
                "discovery module results"
            );
        }

        super::merge_coordinated_structural_replacements(
            syn,
            result.candidates,
            max_synapse_candidates,
            diversify,
        );
    }

    crate::watchdog::beat(&finished);
}

/// Specification for a single discovery module to be dispatched in parallel.
///
/// Each module provides a name (for logging/watchdog), a phase name (for timing),
/// a maximum candidate budget (Issue #967), and a detection closure that returns
/// candidates independently.
pub struct DiscoveryModuleSpec {
    pub module_name: String,
    pub phase_name: &'static str,
    /// Maximum number of candidates this module should generate (Issue #967).
    /// Allocated by `allocate_candidate_budgets` based on historical success rate.
    pub max_candidates: usize,
    pub detect_fn: Box<dyn FnOnce() -> Option<DiscoveryDetectionResult> + Send>,
}

/// A single module's detection output, paired with its metadata for the merge phase.
pub struct DiscoveryModuleDetectionEntry {
    pub module_name: String,
    pub phase_name: &'static str,
    pub max_candidates: usize,
    pub result: Option<DiscoveryDetectionResult>,
}

/// Collected detection results from all discovery modules (Issue #1004).
///
/// Returned by [`detect_discovery_modules_parallel`] for deferred merging,
/// allowing the detection phase to overlap with other concurrent work
/// (e.g., candidate compression).
pub struct DiscoveryModuleDetectionResults {
    pub entries: Vec<DiscoveryModuleDetectionEntry>,
}

/// Run all discovery module detection phases in parallel, returning results
/// without merging into the synapse result (Issue #1004).
///
/// Detection closures execute concurrently via `rayon::into_par_iter()`. Results
/// are collected in original module order (rayon preserves indexed iterator order),
/// ensuring deterministic output regardless of thread scheduling.
///
/// ## Module gating (Issue #1060)
///
/// When a tracker is provided, modules whose historical success rate is below
/// [`MODULE_GATE_THRESHOLD`] are skipped entirely, saving compute on consistently
/// failing modules.
///
/// ## Deadline enforcement (Issue #1029)
///
/// When a deadline is provided, each module checks `deadline_passed()` before
/// executing its detection closure. Modules that start after the deadline are
/// skipped (returning `None`), preventing the parallel detection phase from
/// running indefinitely when the analysis time budget is exhausted.
///
/// Call [`merge_discovery_module_results`] afterwards to merge into `syn`.
#[tracing::instrument(skip_all, fields(module_count = modules.len()))]
pub fn detect_discovery_modules_parallel(
    modules: Vec<DiscoveryModuleSpec>,
    deadline: Option<SystemTime>,
    tracker: Option<&ModuleOutcomeTracker>,
) -> DiscoveryModuleDetectionResults {
    if modules.is_empty() {
        return DiscoveryModuleDetectionResults {
            entries: Vec::new(),
        };
    }

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection starting");
    let _timer = PhaseTimer::new("parallel_discovery_detection");

    // Issue #1029: Shared flag so that once the deadline is observed by any thread,
    // all remaining modules skip execution promptly without repeated syscalls.
    let timed_out = AtomicBool::new(false);

    // Parallel detection phase: run all closures concurrently.
    // `into_par_iter().map().collect()` preserves input order for indexed iterators.
    let entries: Vec<DiscoveryModuleDetectionEntry> = modules
        .into_par_iter()
        .map(|spec| {
            // Issue #1060: Skip detection if the module is gated (low success rate).
            if let Some(t) = tracker
                && t.is_gated(&spec.module_name, MODULE_GATE_THRESHOLD)
            {
                tracing::debug!(
                    module = %spec.module_name,
                    threshold = MODULE_GATE_THRESHOLD,
                    "Skipping discovery module — gated by low success rate (Issue #1060)"
                );
                return DiscoveryModuleDetectionEntry {
                    module_name: spec.module_name,
                    phase_name: spec.phase_name,
                    max_candidates: spec.max_candidates,
                    result: None,
                };
            }

            // Issue #1029: Skip detection if the analysis deadline has passed.
            if timed_out.load(Ordering::Relaxed) || utils::deadline_passed(&deadline) {
                timed_out.store(true, Ordering::Relaxed);
                tracing::debug!(
                    module = %spec.module_name,
                    "Skipping discovery module — deadline passed (Issue #1029)"
                );
                return DiscoveryModuleDetectionEntry {
                    module_name: spec.module_name,
                    phase_name: spec.phase_name,
                    max_candidates: spec.max_candidates,
                    result: None,
                };
            }

            let result = (spec.detect_fn)();
            DiscoveryModuleDetectionEntry {
                module_name: spec.module_name,
                phase_name: spec.phase_name,
                max_candidates: spec.max_candidates,
                result,
            }
        })
        .collect();

    let skipped = timed_out.load(Ordering::Relaxed);
    if skipped {
        let skipped_count = entries.iter().filter(|e| e.result.is_none()).count();
        tracing::info!(
            skipped_count,
            total = entries.len(),
            "Discovery detection reached deadline — some modules were skipped (Issue #1029)"
        );
    }

    // Issue #1060: Log gated modules for observability.
    if let Some(t) = tracker {
        let gated_count = entries
            .iter()
            .filter(|e| t.is_gated(&e.module_name, MODULE_GATE_THRESHOLD))
            .count();
        if gated_count > 0 {
            tracing::info!(
                gated_count,
                total = entries.len(),
                threshold = MODULE_GATE_THRESHOLD,
                "Discovery detection: module(s) gated by low success rate (Issue #1060)"
            );
        }
    }

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection finished");

    DiscoveryModuleDetectionResults { entries }
}

/// Count how many accumulated coordinated structural candidates exceed the
/// quality gain threshold (Issue #1074).
fn count_high_quality_candidates(syn: &shared::AnalyzeSynapsesResult, threshold: f32) -> usize {
    syn.coordinated_structural_candidates
        .iter()
        .filter(|c| c.expected_creature_score_gain >= threshold)
        .count()
}

/// Merge previously-detected discovery module results into the synapse result (Issue #1004).
///
/// Iterates entries in original order and merges non-empty results sequentially.
/// Also collects per-module stats for metadata (Issue #485, #792) and records
/// pre-filtering soft failures in the tracker (Issue #1060).
///
/// ## Quality-based module skipping (Issue #1074)
///
/// After merging each module's results, checks whether the accumulated
/// candidates already contain enough high-quality entries (at least
/// [`QUALITY_SKIP_MIN_CANDIDATES`] candidates with gain above
/// [`QUALITY_SKIP_GAIN_THRESHOLD`]). If so, remaining modules are skipped
/// during the merge phase, saving post-processing time.
pub fn merge_discovery_module_results(
    syn: &mut shared::AnalyzeSynapsesResult,
    detection_results: DiscoveryModuleDetectionResults,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    tracker: &mut ModuleOutcomeTracker,
) {
    let mut quality_skip_active = false;
    let mut modules_skipped_by_quality: usize = 0;

    for entry in detection_results.entries {
        let candidates_produced = entry.result.as_ref().map_or(0, |r| r.candidates.len());

        // Record per-module stats in metadata from historical tracker (Issue #792).
        let historical = tracker.stats(&entry.module_name);
        let gated = tracker.is_gated_default(&entry.module_name);
        syn.metadata
            .discovery_module_stats
            .push(DiscoveryModuleStatsJson {
                module_name: entry.module_name.clone(),
                candidates_produced,
                attempts: historical.attempts,
                successes: historical.successes,
                success_rate: historical.success_rate(),
                soft_failures: historical.soft_failures,
                gated,
            });

        // Issue #1074: Quality-based module skipping — once enough high-quality
        // candidates have been accumulated, skip merging results from remaining
        // modules. Stats are still recorded for observability.
        if quality_skip_active {
            modules_skipped_by_quality += 1;
            let finished = format!("analysis::analyze_all → {} finished", entry.module_name);
            crate::watchdog::beat(&finished);
            continue;
        }

        if let Some(mut result) = entry.result
            && !result.candidates.is_empty()
        {
            let initial_count = result.candidates.len();

            // Issue #967: Truncate to per-module candidate budget if set.
            if entry.max_candidates > 0 && result.candidates.len() > entry.max_candidates {
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %entry.module_name,
                        before = result.candidates.len(),
                        budget = entry.max_candidates,
                        "Truncating candidates to module budget (Issue #967)"
                    );
                }
                result.candidates.truncate(entry.max_candidates);
            }

            // Issue #1060: Count candidates filtered by positive-gain check before merge.
            let pre_filter_count = result.candidates.len();
            result
                .candidates
                .retain(|c| c.expected_creature_score_gain > 0.0);
            let post_filter_count = result.candidates.len();

            // Issue #1060: Record pre-filtering soft failures for candidates that
            // were truncated (budget exceeded) or filtered (non-positive gain).
            let filtered_count = initial_count - post_filter_count;
            if filtered_count > 0 {
                tracker.record_soft_failures(
                    &entry.module_name,
                    filtered_count,
                    SOFT_FAILURE_WEIGHT,
                );
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %entry.module_name,
                        filtered = filtered_count,
                        truncated = initial_count - pre_filter_count,
                        negative_gain = pre_filter_count - post_filter_count,
                        weight = SOFT_FAILURE_WEIGHT,
                        "Recorded pre-filtering soft failures (Issue #1060)"
                    );
                }
            }

            if !result.candidates.is_empty() {
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %entry.module_name,
                        detections = result.detected_count,
                        candidates = result.candidates.len(),
                        budget = entry.max_candidates,
                        "discovery module results"
                    );
                }

                super::merge_coordinated_structural_replacements(
                    syn,
                    result.candidates,
                    max_synapse_candidates,
                    diversify,
                );
            }
        }

        // Issue #1074: After merging this module's results, check if we have
        // enough high-quality candidates to skip remaining modules.
        let high_quality_count = count_high_quality_candidates(syn, QUALITY_SKIP_GAIN_THRESHOLD);
        if high_quality_count >= QUALITY_SKIP_MIN_CANDIDATES {
            quality_skip_active = true;
            tracing::debug!(
                high_quality_count,
                threshold = QUALITY_SKIP_GAIN_THRESHOLD,
                min_required = QUALITY_SKIP_MIN_CANDIDATES,
                "Quality-based module skipping activated (Issue #1074)"
            );
        }

        let finished = format!("analysis::analyze_all → {} finished", entry.module_name);
        crate::watchdog::beat(&finished);
    }

    if modules_skipped_by_quality > 0 {
        tracing::info!(
            skipped = modules_skipped_by_quality,
            "Discovery merge: skipped module(s) due to quality-based early exit (Issue #1074)"
        );
    }
}

/// Run all discovery module detection phases in parallel, then merge results
/// sequentially into the synapse result (Issue #419).
///
/// Detection closures execute concurrently via `rayon::into_par_iter()`. Results
/// are collected in original module order (rayon preserves indexed iterator order),
/// ensuring deterministic output regardless of thread scheduling.
///
/// The merge phase runs sequentially because `merge_coordinated_structural_replacements`
/// mutates the synapse result and may re-sort/truncate the combined candidate set.
#[tracing::instrument(skip_all, fields(module_count = modules.len()))]
pub fn run_discovery_modules_parallel(
    syn: &mut shared::AnalyzeSynapsesResult,
    modules: Vec<DiscoveryModuleSpec>,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    tracker: &mut ModuleOutcomeTracker,
) {
    let detection_results = detect_discovery_modules_parallel(modules, None, Some(tracker));
    merge_discovery_module_results(
        syn,
        detection_results,
        max_synapse_candidates,
        diversify,
        tracker,
    );
}

#[cfg(test)]
#[path = "discovery_dispatch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "discovery_dispatch_parallel_tests.rs"]
mod parallel_tests;
