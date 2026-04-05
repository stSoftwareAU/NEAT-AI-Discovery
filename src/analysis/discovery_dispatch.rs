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

use crate::CoordinatedStructuralCandidateJson;
use crate::observability::PhaseTimer;
use rayon::prelude::*;

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
/// Call [`merge_discovery_module_results`] afterwards to merge into `syn`.
#[tracing::instrument(skip_all, fields(module_count = modules.len()))]
pub fn detect_discovery_modules_parallel(
    modules: Vec<DiscoveryModuleSpec>,
) -> DiscoveryModuleDetectionResults {
    if modules.is_empty() {
        return DiscoveryModuleDetectionResults {
            entries: Vec::new(),
        };
    }

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection starting");
    let _timer = PhaseTimer::new("parallel_discovery_detection");

    // Parallel detection phase: run all closures concurrently.
    // `into_par_iter().map().collect()` preserves input order for indexed iterators.
    let entries: Vec<DiscoveryModuleDetectionEntry> = modules
        .into_par_iter()
        .map(|spec| {
            let result = (spec.detect_fn)();
            DiscoveryModuleDetectionEntry {
                module_name: spec.module_name,
                phase_name: spec.phase_name,
                max_candidates: spec.max_candidates,
                result,
            }
        })
        .collect();

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection finished");

    DiscoveryModuleDetectionResults { entries }
}

/// Merge previously-detected discovery module results into the synapse result (Issue #1004).
///
/// Iterates entries in original order and merges non-empty results sequentially.
/// Also collects per-module stats for metadata (Issue #485, #792).
pub fn merge_discovery_module_results(
    syn: &mut shared::AnalyzeSynapsesResult,
    detection_results: DiscoveryModuleDetectionResults,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    tracker: &ModuleOutcomeTracker,
) {
    for entry in detection_results.entries {
        let candidates_produced = entry.result.as_ref().map_or(0, |r| r.candidates.len());

        // Record per-module stats in metadata from historical tracker (Issue #792).
        let historical = tracker.stats(&entry.module_name);
        syn.metadata
            .discovery_module_stats
            .push(DiscoveryModuleStatsJson {
                module_name: entry.module_name.clone(),
                candidates_produced,
                attempts: historical.attempts,
                successes: historical.successes,
                success_rate: historical.success_rate(),
            });

        if let Some(mut result) = entry.result
            && !result.candidates.is_empty()
        {
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

        let finished = format!("analysis::analyze_all → {} finished", entry.module_name);
        crate::watchdog::beat(&finished);
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
    tracker: &ModuleOutcomeTracker,
) {
    let detection_results = detect_discovery_modules_parallel(modules);
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
