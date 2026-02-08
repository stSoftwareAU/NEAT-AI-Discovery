//! Generic discovery module dispatch pattern (Issue #375).
//!
//! Extracts the repeated detect → convert → log → merge boilerplate from
//! `analyze_all()` into a single reusable function. Each discovery module
//! supplies its detection and conversion logic via closures, while this
//! module handles watchdog beats, phase timing, verbose logging, and
//! merging into the synapse result.

use crate::observability::PhaseTimer;
use crate::CoordinatedStructuralCandidateJson;

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

/// Run a discovery module with cross-module deduplication (Issue #429).
///
/// Same as [`run_discovery_module`] but filters out near-duplicate candidates
/// before merging, using the shared deduplicator across all discovery modules.
pub fn run_discovery_module_dedup(
    syn: &mut shared::AnalyzeSynapsesResult,
    module_name: &str,
    phase_name: &'static str,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    dedup: &mut super::early_termination::CrossModuleDeduplicator,
    detect_fn: impl FnOnce() -> Option<DiscoveryDetectionResult>,
) {
    let starting = format!("analysis::analyze_all → {module_name} starting");
    let finished = format!("analysis::analyze_all → {module_name} finished");

    crate::watchdog::beat(&starting);
    let _timer = PhaseTimer::new(phase_name);

    if let Some(result) = detect_fn() {
        if !result.candidates.is_empty() {
            // Deduplicate against previously seen candidates
            let unique_candidates: Vec<CoordinatedStructuralCandidateJson> = result
                .candidates
                .into_iter()
                .filter(|c| {
                    let key = c.comment.as_deref().unwrap_or("unknown");
                    dedup.register_if_unique(key, module_name, c.expected_creature_score_gain)
                })
                .collect();

            if !unique_candidates.is_empty() {
                if utils::verbose_enabled() {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] {module_name}: found {} detection(s), {} unique candidate(s) (after dedup)",
                        result.detected_count,
                        unique_candidates.len()
                    );
                }

                super::merge_coordinated_structural_replacements(
                    syn,
                    unique_candidates,
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
