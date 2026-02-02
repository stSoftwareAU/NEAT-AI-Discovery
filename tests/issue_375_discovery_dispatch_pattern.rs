//! Integration tests for the generic discovery module dispatch pattern (Issue #375).
//!
//! These tests verify that the dispatch infrastructure correctly:
//! - Merges candidates into synapse results
//! - Skips modules when the closure returns `None`
//! - Skips when candidates are empty
//! - Accumulates candidates from multiple dispatches
//! - Respects `max_synapse_candidates` limits

use neat_ai_discovery::analysis::discovery_dispatch::{
    dispatch_discovery_module, DetectionResult, DiscoveryDispatchConfig,
};
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::CoordinatedStructuralCandidateJson;

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
        metadata: SynapseAnalysisMetadata::default(),
    }
}

fn make_candidate(gain: f32, comment: &str) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![],
        expected_creature_score_gain: gain,
        comment: Some(comment.to_string()),
    }
}

#[test]
fn test_dispatch_merges_candidates() {
    let mut syn = empty_synapse_result();
    dispatch_discovery_module(
        &DiscoveryDispatchConfig {
            name: "test module",
            phase_name: "test_module",
        },
        &mut syn,
        None,
        false,
        || {
            Some(DetectionResult {
                detected_count: 2,
                candidates: vec![make_candidate(0.5, "a"), make_candidate(0.3, "b")],
            })
        },
    );

    assert_eq!(syn.coordinated_structural_candidates.len(), 2);
}

#[test]
fn test_dispatch_skips_on_none() {
    let mut syn = empty_synapse_result();
    dispatch_discovery_module(
        &DiscoveryDispatchConfig {
            name: "skipped",
            phase_name: "skipped",
        },
        &mut syn,
        None,
        false,
        || None,
    );

    assert!(syn.coordinated_structural_candidates.is_empty());
}

#[test]
fn test_dispatch_skips_on_empty_candidates() {
    let mut syn = empty_synapse_result();
    dispatch_discovery_module(
        &DiscoveryDispatchConfig {
            name: "empty",
            phase_name: "empty",
        },
        &mut syn,
        None,
        false,
        || {
            Some(DetectionResult {
                detected_count: 0,
                candidates: vec![],
            })
        },
    );

    assert!(syn.coordinated_structural_candidates.is_empty());
}

#[test]
fn test_multiple_dispatches_accumulate() {
    let mut syn = empty_synapse_result();

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
                candidates: vec![make_candidate(0.8, "from A")],
            })
        },
    );

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
                candidates: vec![make_candidate(0.6, "from B")],
            })
        },
    );

    assert_eq!(syn.coordinated_structural_candidates.len(), 2);
    // Sorted by expected gain (highest first) since diversify=false
    assert_eq!(
        syn.coordinated_structural_candidates[0].expected_creature_score_gain,
        0.8
    );
    assert_eq!(
        syn.coordinated_structural_candidates[1].expected_creature_score_gain,
        0.6
    );
}

#[test]
fn test_dispatch_respects_max_candidates_limit() {
    let mut syn = empty_synapse_result();
    let candidates: Vec<CoordinatedStructuralCandidateJson> = (0..5)
        .map(|i| make_candidate(1.0 - (i as f32 * 0.1), &format!("c{i}")))
        .collect();

    dispatch_discovery_module(
        &DiscoveryDispatchConfig {
            name: "limited",
            phase_name: "limited",
        },
        &mut syn,
        Some(2),
        false,
        || {
            Some(DetectionResult {
                detected_count: 5,
                candidates,
            })
        },
    );

    assert!(syn.coordinated_structural_candidates.len() <= 2);
}

#[test]
fn test_dispatch_preserves_candidate_details() {
    let mut syn = empty_synapse_result();
    dispatch_discovery_module(
        &DiscoveryDispatchConfig {
            name: "detail check",
            phase_name: "detail_check",
        },
        &mut syn,
        None,
        false,
        || {
            Some(DetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(0.42, "specific comment")],
            })
        },
    );

    assert_eq!(syn.coordinated_structural_candidates.len(), 1);
    let c = &syn.coordinated_structural_candidates[0];
    assert!((c.expected_creature_score_gain - 0.42).abs() < f32::EPSILON);
    assert_eq!(c.comment.as_deref(), Some("specific comment"));
}
