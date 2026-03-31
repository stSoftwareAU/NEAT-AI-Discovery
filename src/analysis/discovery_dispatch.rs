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
    if modules.is_empty() {
        return;
    }

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection starting");
    let _timer = PhaseTimer::new("parallel_discovery_detection");

    // Parallel detection phase: run all closures concurrently.
    // `into_par_iter().map().collect()` preserves input order for indexed iterators.
    let results: Vec<(
        String,
        &'static str,
        usize,
        Option<DiscoveryDetectionResult>,
    )> = modules
        .into_par_iter()
        .map(|spec| {
            let result = (spec.detect_fn)();
            (
                spec.module_name,
                spec.phase_name,
                spec.max_candidates,
                result,
            )
        })
        .collect();

    crate::watchdog::beat("analysis::analyze_all → parallel discovery detection finished");

    // Sequential merge phase: iterate in original order and merge non-empty results.
    // Also collect per-module stats for metadata (Issue #485, #792).
    for (module_name, _phase_name, max_candidates, result) in results {
        let candidates_produced = result.as_ref().map_or(0, |r| r.candidates.len());

        // Record per-module stats in metadata from historical tracker (Issue #792).
        let historical = tracker.stats(&module_name);
        syn.metadata
            .discovery_module_stats
            .push(DiscoveryModuleStatsJson {
                module_name: module_name.clone(),
                candidates_produced,
                attempts: historical.attempts,
                successes: historical.successes,
                success_rate: historical.success_rate(),
            });

        if let Some(mut result) = result
            && !result.candidates.is_empty()
        {
            // Issue #967: Truncate to per-module candidate budget if set.
            if max_candidates > 0 && result.candidates.len() > max_candidates {
                if utils::verbose_enabled() {
                    tracing::debug!(
                        module = %module_name,
                        before = result.candidates.len(),
                        budget = max_candidates,
                        "Truncating candidates to module budget (Issue #967)"
                    );
                }
                result.candidates.truncate(max_candidates);
            }

            if utils::verbose_enabled() {
                tracing::debug!(
                    module = %module_name,
                    detections = result.detected_count,
                    candidates = result.candidates.len(),
                    budget = max_candidates,
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

        let finished = format!("analysis::analyze_all → {module_name} finished");
        crate::watchdog::beat(&finished);
    }
}

#[cfg(test)]
#[path = "discovery_dispatch_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "discovery_dispatch_parallel_tests.rs"]
mod parallel_tests;
