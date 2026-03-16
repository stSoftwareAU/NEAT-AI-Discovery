//! Tests for Issue #224: Candidate clustering to reduce redundant ablation tests.
//!
//! When discovery returns many similar candidates (e.g., multiple synapses from the same
//! source region targeting the same neuron), clustering groups them so the controller can
//! test a representative first and skip redundant tests if it fails.
//!
//! ## TDD Plan
//! 1. Candidates targeting the same neuron with similar improvements are clustered
//! 2. Representative is the highest-improvement candidate in each cluster
//! 3. Independent candidates (different targets) form separate clusters
//! 4. Internal correlation is computed correctly
//! 5. Single candidate does not form a cluster (no redundancy to reduce)
//! 6. Empty input produces no clusters
//! 7. Cluster JSON output has the expected structure

use neat_ai_discovery::analysis::candidate_clustering::{ClusterableCandidate, cluster_candidates};

/// Helper: create a ClusterableCandidate for testing.
fn candidate(from: &str, to: &str, improvement: f32) -> ClusterableCandidate {
    ClusterableCandidate {
        from_neuron_uuid: from.to_string(),
        to_neuron_uuid: to.to_string(),
        expected_improvement: improvement,
        neuron_type: "input".to_string(),
    }
}

/// Test 1: Candidates targeting the same neuron are clustered together.
#[test]
fn test_same_target_candidates_clustered() {
    let candidates = vec![
        candidate("input-42", "hidden-5", 0.10),
        candidate("input-43", "hidden-5", 0.098),
        candidate("input-44", "hidden-5", 0.101),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(
        clusters.len(),
        1,
        "All candidates targeting hidden-5 should form one cluster"
    );
    assert_eq!(
        clusters[0].member_count, 3,
        "Cluster should contain all 3 candidates"
    );
}

/// Test 2: Representative is the highest-improvement candidate.
#[test]
fn test_representative_is_best_candidate() {
    let candidates = vec![
        candidate("input-42", "hidden-5", 0.10),
        candidate("input-43", "hidden-5", 0.098),
        candidate("input-44", "hidden-5", 0.15), // highest
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(clusters.len(), 1);
    assert_eq!(
        clusters[0].representative_from_uuid, "input-44",
        "Representative should be the candidate with highest improvement"
    );
    assert_eq!(clusters[0].representative_to_uuid, "hidden-5");
}

/// Test 3: Candidates with different targets form separate clusters.
#[test]
fn test_different_targets_separate_clusters() {
    let candidates = vec![
        candidate("input-1", "hidden-5", 0.10),
        candidate("input-2", "hidden-5", 0.09),
        candidate("input-3", "hidden-10", 0.08),
        candidate("input-4", "hidden-10", 0.07),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(
        clusters.len(),
        2,
        "Two different targets should form two clusters"
    );

    // Each cluster should have 2 members
    let mut counts: Vec<usize> = clusters.iter().map(|c| c.member_count).collect();
    counts.sort();
    assert_eq!(counts, vec![2, 2]);
}

/// Test 4: Internal correlation is between 0.0 and 1.0.
#[test]
fn test_internal_correlation_range() {
    let candidates = vec![
        candidate("input-42", "hidden-5", 0.10),
        candidate("input-43", "hidden-5", 0.098),
        candidate("input-44", "hidden-5", 0.101),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(clusters.len(), 1);
    assert!(
        clusters[0].internal_correlation >= 0.0 && clusters[0].internal_correlation <= 1.0,
        "Internal correlation should be between 0.0 and 1.0, got {}",
        clusters[0].internal_correlation
    );
}

/// Test 5: Single candidate does not form a cluster.
/// A single candidate targeting a neuron has no redundancy to reduce.
#[test]
fn test_single_candidate_no_cluster() {
    let candidates = vec![candidate("input-1", "hidden-5", 0.10)];

    let clusters = cluster_candidates(&candidates);

    assert!(
        clusters.is_empty(),
        "Single candidate should not form a cluster (no redundancy)"
    );
}

/// Test 6: Empty input produces no clusters.
#[test]
fn test_empty_candidates_no_clusters() {
    let clusters = cluster_candidates(&[]);

    assert!(
        clusters.is_empty(),
        "Empty input should produce no clusters"
    );
}

/// Test 7: Cluster JSON structure has expected fields.
#[test]
fn test_cluster_json_serialisation() {
    let candidates = vec![
        candidate("input-42", "hidden-5", 0.10),
        candidate("input-43", "hidden-5", 0.098),
        candidate("input-44", "hidden-5", 0.101),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(clusters.len(), 1);
    let cluster = &clusters[0];

    // Verify JSON serialisation
    let json = serde_json::to_value(cluster).unwrap();
    assert!(json.get("representativeFromUuid").is_some());
    assert!(json.get("representativeToUuid").is_some());
    assert!(json.get("memberCount").is_some());
    assert!(json.get("memberFromUuids").is_some());
    assert!(json.get("internalCorrelation").is_some());

    // Verify member UUIDs
    let member_uuids = json["memberFromUuids"].as_array().unwrap();
    assert_eq!(member_uuids.len(), 3);
}

/// Test 8: Candidates with very different improvements targeting the same neuron
/// but from different neuron types form separate sub-clusters.
#[test]
fn test_mixed_types_sub_clustering() {
    let candidates = vec![
        // Input neurons targeting hidden-5 (similar improvements)
        ClusterableCandidate {
            from_neuron_uuid: "input-1".to_string(),
            to_neuron_uuid: "hidden-5".to_string(),
            expected_improvement: 0.10,
            neuron_type: "input".to_string(),
        },
        ClusterableCandidate {
            from_neuron_uuid: "input-2".to_string(),
            to_neuron_uuid: "hidden-5".to_string(),
            expected_improvement: 0.098,
            neuron_type: "input".to_string(),
        },
        // Hidden neurons targeting hidden-5 (similar improvements but different type)
        ClusterableCandidate {
            from_neuron_uuid: "hidden-1".to_string(),
            to_neuron_uuid: "hidden-5".to_string(),
            expected_improvement: 0.05,
            neuron_type: "hidden".to_string(),
        },
        ClusterableCandidate {
            from_neuron_uuid: "hidden-2".to_string(),
            to_neuron_uuid: "hidden-5".to_string(),
            expected_improvement: 0.048,
            neuron_type: "hidden".to_string(),
        },
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(
        clusters.len(),
        2,
        "Input and hidden sources should form separate sub-clusters"
    );
}

/// Test 9: Candidates with widely different improvements targeting the same neuron
/// and same type form separate sub-clusters due to improvement dissimilarity.
#[test]
fn test_dissimilar_improvements_separate_clusters() {
    let candidates = vec![
        candidate("input-1", "hidden-5", 0.001), // very small
        candidate("input-2", "hidden-5", 0.002), // very small
        candidate("input-3", "hidden-5", 0.50),  // very large
        candidate("input-4", "hidden-5", 0.55),  // very large
    ];

    let clusters = cluster_candidates(&candidates);

    assert!(
        clusters.len() >= 2,
        "Very different improvements should form separate sub-clusters, got {}",
        clusters.len()
    );
}

/// Test 10: Cluster members are sorted by improvement (best first).
#[test]
fn test_cluster_members_sorted_by_improvement() {
    let candidates = vec![
        candidate("input-1", "hidden-5", 0.05),
        candidate("input-2", "hidden-5", 0.15),
        candidate("input-3", "hidden-5", 0.10),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(clusters.len(), 1);
    let cluster = &clusters[0];

    // Representative should be the best one
    assert_eq!(cluster.representative_from_uuid, "input-2");

    // member_from_uuids should be ordered by improvement (best first)
    assert_eq!(cluster.member_from_uuids[0], "input-2");
}

/// Test 11: Multiple clusters are sorted by representative improvement (best first).
#[test]
fn test_clusters_sorted_by_best_improvement() {
    let candidates = vec![
        candidate("input-1", "hidden-5", 0.05),
        candidate("input-2", "hidden-5", 0.06),
        candidate("input-3", "hidden-10", 0.20),
        candidate("input-4", "hidden-10", 0.18),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(clusters.len(), 2);

    // First cluster should be the one with the best representative
    assert!(
        clusters[0].representative_improvement >= clusters[1].representative_improvement,
        "Clusters should be sorted by representative improvement (best first)"
    );
}

/// Test 12: High similarity within cluster yields high internal correlation.
#[test]
fn test_similar_improvements_yield_high_correlation() {
    // All very similar improvements
    let candidates = vec![
        candidate("input-1", "hidden-5", 0.100),
        candidate("input-2", "hidden-5", 0.101),
        candidate("input-3", "hidden-5", 0.099),
        candidate("input-4", "hidden-5", 0.100),
    ];

    let clusters = cluster_candidates(&candidates);

    assert_eq!(clusters.len(), 1);
    assert!(
        clusters[0].internal_correlation >= 0.8,
        "Very similar improvements should yield high internal correlation, got {}",
        clusters[0].internal_correlation
    );
}

/// Test 13: Large number of candidates is correctly clustered.
#[test]
fn test_large_candidate_set() {
    let candidates: Vec<ClusterableCandidate> = (0..100)
        .map(|i| {
            let target = if i < 50 { "hidden-5" } else { "hidden-10" };
            let improvement = 0.1 + (i as f32 * 0.001);
            candidate(&format!("input-{i}"), target, improvement)
        })
        .collect();

    let clusters = cluster_candidates(&candidates);

    // Should form at least 2 clusters (one per target)
    assert!(
        clusters.len() >= 2,
        "100 candidates across 2 targets should form at least 2 clusters, got {}",
        clusters.len()
    );

    // Total members across all clusters should equal the input count
    let total_members: usize = clusters.iter().map(|c| c.member_count).sum();
    assert_eq!(
        total_members, 100,
        "All candidates should be assigned to clusters"
    );
}
