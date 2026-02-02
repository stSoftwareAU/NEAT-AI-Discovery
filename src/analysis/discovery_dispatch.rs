//! Generic discovery module dispatch pattern (Issue #375).
//!
//! Each discovery module follows the same pipeline:
//! 1. Watchdog beat (starting)
//! 2. `PhaseTimer` creation
//! 3. UUID/record collection from the shared cache
//! 4. Detection function call
//! 5. Conversion to coordinated candidates
//! 6. Verbose logging
//! 7. Merge into synapse results
//! 8. Watchdog beat (finished)
//!
//! This module extracts that boilerplate into a single generic function so that
//! `analyze_all()` can dispatch each module with minimal repetition.

use crate::analysis::cache::RecordCache;
use crate::analysis::shared;
use crate::analysis::utils;
use crate::observability::PhaseTimer;
use crate::types::DiscoverRecord;
use crate::{CoordinatedStructuralCandidateJson, CreatureJson};
use std::sync::Arc;

/// Collects records for a set of UUIDs from the shared cache.
///
/// Returns a `Vec<(String, Vec<DiscoverRecord>)>` — one entry per UUID that
/// was found in the cache.
pub fn collect_records_for_uuids(
    uuids: &[String],
    shared_cache: &Arc<RecordCache>,
) -> Vec<(String, Vec<DiscoverRecord>)> {
    uuids
        .iter()
        .filter_map(|uuid| {
            shared_cache
                .get(uuid)
                .ok()
                .map(|records: Arc<Vec<DiscoverRecord>>| (uuid.clone(), records.as_ref().to_vec()))
        })
        .collect()
}

/// Configuration for a single discovery module dispatch.
pub struct DiscoveryDispatchConfig<'a> {
    /// Human-readable name for watchdog and logging (e.g. "saturation detection").
    pub name: &'a str,
    /// Phase timer name (e.g. "saturation_detection") — must be `'static` for `PhaseTimer`.
    pub phase_name: &'static str,
}

/// Result from a detection function, pairing the detected items count with
/// the coordinated candidates produced from them.
pub struct DetectionResult {
    /// Number of raw detections (for verbose logging).
    pub detected_count: usize,
    /// Coordinated candidates produced from the detections.
    pub candidates: Vec<CoordinatedStructuralCandidateJson>,
}

/// Run a single discovery module through the standard dispatch pipeline.
///
/// The `detect_and_convert` closure receives nothing and is expected to perform
/// the detection and conversion steps, returning a `DetectionResult` (or `None`
/// if the preconditions for running the module are not met).
///
/// This function handles:
/// - Watchdog beats (starting / finished)
/// - `PhaseTimer` creation
/// - Verbose logging with consistent formatting
/// - Merging candidates via `merge_coordinated_structural_replacements`
pub fn dispatch_discovery_module(
    config: &DiscoveryDispatchConfig,
    syn: &mut shared::AnalyzeSynapsesResult,
    max_synapse_candidates: Option<usize>,
    diversify: bool,
    detect_and_convert: impl FnOnce() -> Option<DetectionResult>,
) {
    crate::watchdog::beat(format!(
        "analysis::analyze_all → {name} starting",
        name = config.name
    ));
    let _timer = PhaseTimer::new(config.phase_name);

    if let Some(result) = detect_and_convert() {
        if !result.candidates.is_empty() {
            if utils::verbose_enabled() {
                eprintln!(
                    "[NEAT-AI-Discovery][verbose] {name}: found {detected} detection(s), {candidates} candidate(s)",
                    name = config.name,
                    detected = result.detected_count,
                    candidates = result.candidates.len()
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

    crate::watchdog::beat(format!(
        "analysis::analyze_all → {name} finished",
        name = config.name
    ));
}

/// Collect UUIDs for hidden neurons from the creature and return the `(uuid, squash, bias)` tuples.
pub fn collect_hidden_neurons(creature: &CreatureJson) -> Vec<(String, String, f32)> {
    creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "hidden")
        .map(|n| (n.uuid.clone(), n.squash.clone(), n.bias))
        .collect()
}

/// Collect records for hidden neurons from the shared cache.
pub fn collect_hidden_neuron_records(
    hidden_neurons: &[(String, String, f32)],
    shared_cache: &Arc<RecordCache>,
) -> Vec<(String, Vec<DiscoverRecord>)> {
    let uuids: Vec<String> = hidden_neurons
        .iter()
        .map(|(uuid, _, _)| uuid.clone())
        .collect();
    collect_records_for_uuids(&uuids, shared_cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::shared::AnalyzeSynapsesResult;

    /// Helper to create a minimal empty synapse result for testing.
    fn empty_synapse_result() -> AnalyzeSynapsesResult {
        AnalyzeSynapsesResult {
            helpful_synapses: vec![],
            harmful_synapses: vec![],
            synapse_weight_updates: vec![],
            coordinated_structural_candidates: vec![],
            candidate_clusters: vec![],
            gpu_used: false,
            no_candidate_reasons: vec![],
            metadata: crate::analysis::shared::SynapseAnalysisMetadata::default(),
        }
    }

    #[test]
    fn test_dispatch_merges_candidates_into_synapse_result() {
        let mut syn = empty_synapse_result();
        let config = DiscoveryDispatchConfig {
            name: "test module",
            phase_name: "test_module",
        };

        dispatch_discovery_module(&config, &mut syn, None, false, || {
            Some(DetectionResult {
                detected_count: 2,
                candidates: vec![
                    CoordinatedStructuralCandidateJson {
                        operations: vec![],
                        expected_creature_score_gain: 0.5,
                        comment: Some("test candidate 1".to_string()),
                    },
                    CoordinatedStructuralCandidateJson {
                        operations: vec![],
                        expected_creature_score_gain: 0.3,
                        comment: Some("test candidate 2".to_string()),
                    },
                ],
            })
        });

        assert_eq!(syn.coordinated_structural_candidates.len(), 2);
        assert_eq!(
            syn.coordinated_structural_candidates[0].expected_creature_score_gain,
            0.5
        );
    }

    #[test]
    fn test_dispatch_skips_when_closure_returns_none() {
        let mut syn = empty_synapse_result();
        let config = DiscoveryDispatchConfig {
            name: "skipped module",
            phase_name: "skipped_module",
        };

        dispatch_discovery_module(&config, &mut syn, None, false, || None);

        assert!(syn.coordinated_structural_candidates.is_empty());
    }

    #[test]
    fn test_dispatch_skips_when_candidates_empty() {
        let mut syn = empty_synapse_result();
        let config = DiscoveryDispatchConfig {
            name: "empty module",
            phase_name: "empty_module",
        };

        dispatch_discovery_module(&config, &mut syn, None, false, || {
            Some(DetectionResult {
                detected_count: 0,
                candidates: vec![],
            })
        });

        assert!(syn.coordinated_structural_candidates.is_empty());
    }

    #[test]
    fn test_dispatch_respects_max_synapse_candidates_limit() {
        let mut syn = empty_synapse_result();
        let config = DiscoveryDispatchConfig {
            name: "limited module",
            phase_name: "limited_module",
        };

        // Add 5 candidates with a limit of 2
        let candidates: Vec<CoordinatedStructuralCandidateJson> = (0..5)
            .map(|i| CoordinatedStructuralCandidateJson {
                operations: vec![],
                expected_creature_score_gain: 1.0 - (i as f32 * 0.1),
                comment: Some(format!("candidate {i}")),
            })
            .collect();

        dispatch_discovery_module(&config, &mut syn, Some(2), false, || {
            Some(DetectionResult {
                detected_count: 5,
                candidates,
            })
        });

        // merge_coordinated_structural_replacements truncates to the limit
        assert!(syn.coordinated_structural_candidates.len() <= 2);
    }

    #[test]
    fn test_multiple_dispatches_accumulate_candidates() {
        let mut syn = empty_synapse_result();

        // First dispatch
        dispatch_discovery_module(
            &DiscoveryDispatchConfig {
                name: "module A",
                phase_name: "module_a",
            },
            &mut syn,
            None,
            false,
            || {
                Some(DetectionResult {
                    detected_count: 1,
                    candidates: vec![CoordinatedStructuralCandidateJson {
                        operations: vec![],
                        expected_creature_score_gain: 0.8,
                        comment: Some("from A".to_string()),
                    }],
                })
            },
        );

        // Second dispatch
        dispatch_discovery_module(
            &DiscoveryDispatchConfig {
                name: "module B",
                phase_name: "module_b",
            },
            &mut syn,
            None,
            false,
            || {
                Some(DetectionResult {
                    detected_count: 1,
                    candidates: vec![CoordinatedStructuralCandidateJson {
                        operations: vec![],
                        expected_creature_score_gain: 0.6,
                        comment: Some("from B".to_string()),
                    }],
                })
            },
        );

        assert_eq!(syn.coordinated_structural_candidates.len(), 2);
        // Should be sorted by expected gain (highest first) since diversify=false
        assert_eq!(
            syn.coordinated_structural_candidates[0].expected_creature_score_gain,
            0.8
        );
        assert_eq!(
            syn.coordinated_structural_candidates[1].expected_creature_score_gain,
            0.6
        );
    }
}
