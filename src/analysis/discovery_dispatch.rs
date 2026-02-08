//! Generic discovery module dispatch pattern (Issue #375).
//!
//! Extracts the repeated detect → convert → log → merge boilerplate from
//! `analyze_all()` into a single reusable function. Each discovery module
//! supplies its detection and conversion logic via closures, while this
//! module handles watchdog beats, phase timing, verbose logging, and
//! merging into the synapse result.
//!
//! Issue #429: Per-module pre-filtering removes low-value candidates before
//! merging, reducing downstream processing for clearly poor candidates.

use crate::observability::PhaseTimer;
use crate::CoordinatedStructuralCandidateJson;

use super::candidate_prefilter::{filter_low_value_candidates, CandidatePrefilterConfig};
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
/// 4. Pre-filter: remove low-value candidates (Issue #429)
/// 5. Verbose logging if results are non-empty
/// 6. Merge into synapse result via `merge_coordinated_structural_replacements`
/// 7. Watchdog beat (finished)
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
            // Issue #429: Pre-filter low-value candidates before merging.
            let prefilter_config = CandidatePrefilterConfig::default();
            let candidates = filter_low_value_candidates(&result.candidates, &prefilter_config);

            let filtered_count = result.candidates.len() - candidates.len();

            if utils::verbose_enabled() {
                if filtered_count > 0 {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] {module_name}: found {} detection(s), {} candidate(s), {} pre-filtered",
                        result.detected_count,
                        candidates.len(),
                        filtered_count
                    );
                } else {
                    eprintln!(
                        "[NEAT-AI-Discovery][verbose] {module_name}: found {} detection(s), {} candidate(s)",
                        result.detected_count,
                        candidates.len()
                    );
                }
            }

            if !candidates.is_empty() {
                super::merge_coordinated_structural_replacements(
                    syn,
                    candidates,
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
