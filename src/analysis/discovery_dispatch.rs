//! Generic discovery module dispatch pattern (Issue #375).
//!
//! Extracts the repeated detect → convert → log → merge boilerplate from
//! `analyze_all()` into a single reusable function. Each discovery module
//! supplies its detection and conversion logic via closures, while this
//! module handles watchdog beats, phase timing, verbose logging, and
//! merging into the synapse result.
//!
//! Issue #429: Extended with `run_discovery_module_filtered()` for
//! budget-aware dispatch with hierarchical pre-filtering and
//! cross-module deduplication.

use crate::observability::PhaseTimer;
use crate::CoordinatedStructuralCandidateJson;

use super::candidate_prefilter::CandidatePreFilter;
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

    if let Some(result) = detect_fn() {
        if !result.candidates.is_empty() {
            if utils::verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] {module_name}: found {} detection(s), {} candidate(s)",
                    result.detected_count,
                    result.candidates.len()
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

    crate::watchdog::beat(&finished);
}

/// Budget-aware variant of [`run_discovery_module`] with pre-filtering (Issue #429).
///
/// This function wraps the standard dispatch pattern with:
/// 1. **Budget check** — if the candidate budget is exhausted, the module is
///    skipped entirely (no detection closure is called).
/// 2. **Pre-filtering** — candidates produced by the module are filtered
///    through the shared `CandidatePreFilter` before merge, removing
///    low-gain and duplicate candidates.
///
/// The `pre_filter` accumulates state across calls so later modules
/// benefit from earlier filtering decisions.
pub fn run_discovery_module_filtered(
    syn: &mut shared::AnalyzeSynapsesResult,
    module_name: &str,
    phase_name: &'static str,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    pre_filter: &mut CandidatePreFilter,
    detect_fn: impl FnOnce() -> Option<DiscoveryDetectionResult>,
) {
    let starting = format!("analysis::analyze_all → {module_name} starting");
    let finished = format!("analysis::analyze_all → {module_name} finished");

    // Budget-aware skip: if the candidate budget is exhausted, skip the module.
    if pre_filter.budget_exhausted() {
        pre_filter.record_module_skipped();
        if utils::verbose_enabled() {
            eprintln!(
                "[NEAT-AI-Discovery][verbose] {module_name}: skipped (candidate budget exhausted)"
            );
        }
        crate::watchdog::beat(format!(
            "analysis::analyze_all → {module_name} skipped (budget)"
        ));
        return;
    }

    crate::watchdog::beat(&starting);
    let _timer = PhaseTimer::new(phase_name);

    if let Some(result) = detect_fn() {
        if !result.candidates.is_empty() {
            let raw_count = result.candidates.len();

            // Apply pre-filter: gain threshold + budget + deduplication
            let filtered = pre_filter.filter(result.candidates);

            if utils::verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] {module_name}: found {} detection(s), {} raw candidate(s), {} after pre-filter",
                    result.detected_count,
                    raw_count,
                    filtered.len()
                );
            }

            if !filtered.is_empty() {
                super::merge_coordinated_structural_replacements(
                    syn,
                    filtered,
                    max_synapse_candidates,
                    diversify,
                );
            }
        }
    }

    crate::watchdog::beat(&finished);
}

#[cfg(test)]
#[path = "discovery_dispatch_tests.rs"]
mod tests;
