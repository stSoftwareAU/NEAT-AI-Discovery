//! Issue #509: Coordinated-structural: deduplicate candidates sharing a dominant neuron.
//!
//! When generating epistatic/synergistic coordinated-structural candidates, many pairs
//! can share the same "dominant" neuron (the source with the larger individual improvement).
//! This wastes candidate budget with essentially the same experiment repeated with
//! trivially different partners.
//!
//! ## Scenario from the issue
//!
//! In one production discovery-cache creature, 10 out of 20 candidates were
//! coordinated-structural pairs all sharing the same dominant neuron (e8480883 → output-0).
//! All 10 failed with nearly identical results (−0.042 ± 0.001). 50% of the discovery
//! budget was consumed by essentially the same experiment repeated 10 times.
//!
//! ## What this fixes
//!
//! After detecting epistatic pairs, group by dominant neuron and emit at most
//! `MAX_PAIRS_PER_DOMINANT_NEURON` diverse pairs per group (varying the partner,
//! complementarity, etc.) rather than all N×M combinations.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use neat_ai_discovery::analysis::recommendation::epistatic::{
    EpistaticPairCandidate, SynergisticCandidate, deduplicate_by_dominant_neuron,
    deduplicate_synergistic_by_dominant_neuron,
};

/// Helper: create an epistatic pair candidate with specified parameters.
fn make_pair(
    source_a: &str,
    source_b: &str,
    improvement_a: f32,
    improvement_b: f32,
    combined: f32,
    complementarity: f32,
) -> EpistaticPairCandidate {
    EpistaticPairCandidate {
        source_a_uuid: source_a.to_string(),
        source_b_uuid: source_b.to_string(),
        target_uuid: "output-0".to_string(),
        weight_a: 0.1,
        weight_b: 0.1,
        combined_improvement: combined,
        individual_improvement_a: improvement_a,
        individual_improvement_b: improvement_b,
        complementarity_score: complementarity,
        reason: "Test pair".to_string(),
    }
}

/// Helper: create a synergistic candidate with specified parameters.
fn make_synergistic(
    primary: &str,
    complement: &str,
    primary_improvement: f32,
    complement_improvement: f32,
    combined: f32,
) -> SynergisticCandidate {
    SynergisticCandidate {
        primary_source_uuid: primary.to_string(),
        complement_source_uuid: complement.to_string(),
        target_uuid: "output-0".to_string(),
        primary_weight: 0.1,
        complement_weight: 0.1,
        combined_improvement: combined,
        primary_improvement,
        complement_improvement,
        residual_reduction: 0.3,
        synergy_ratio: 1.5,
        reason: "Test synergistic".to_string(),
    }
}

/// Test: Many pairs sharing one dominant neuron are reduced to at most 3.
///
/// Reproduces the core issue scenario: 10 pairs all share the same dominant
/// neuron. After deduplication, at most 3 should remain.
#[test]
fn epistatic_pairs_with_shared_dominant_neuron_are_capped() {
    // Create 10 pairs where "dominant" always has the higher individual improvement
    let pairs: Vec<EpistaticPairCandidate> = (0..10)
        .map(|i| {
            make_pair(
                "dominant",
                &format!("partner-{i}"),
                0.05,                    // dominant has higher improvement
                0.01 + i as f32 * 0.001, // partners vary slightly
                0.08 + i as f32 * 0.001, // combined varies slightly
                0.85 + i as f32 * 0.005, // complementarity varies
            )
        })
        .collect();

    let deduplicated = deduplicate_by_dominant_neuron(pairs);

    assert!(
        deduplicated.len() <= 3,
        "Expected at most 3 pairs for the same dominant neuron, got {}",
        deduplicated.len()
    );
    assert!(
        !deduplicated.is_empty(),
        "Deduplication should keep at least one pair"
    );
}

/// Test: Pairs with different dominant neurons are not affected.
///
/// When each pair has a distinct dominant neuron, no deduplication should occur.
#[test]
fn epistatic_pairs_with_distinct_dominant_neurons_unchanged() {
    let pairs = vec![
        make_pair("dominant-a", "partner-a", 0.05, 0.01, 0.08, 0.9),
        make_pair("dominant-b", "partner-b", 0.05, 0.01, 0.08, 0.9),
        make_pair("dominant-c", "partner-c", 0.05, 0.01, 0.08, 0.9),
    ];

    let deduplicated = deduplicate_by_dominant_neuron(pairs);

    assert_eq!(
        deduplicated.len(),
        3,
        "All pairs have distinct dominant neurons — none should be removed"
    );
}

/// Test: Deduplication selects the best pairs by combined improvement.
///
/// The kept pairs should be the ones with the highest combined improvement
/// from each dominant group.
#[test]
fn epistatic_deduplication_keeps_best_combined_improvement() {
    let pairs = vec![
        make_pair("dominant", "partner-worst", 0.05, 0.01, 0.06, 0.85),
        make_pair("dominant", "partner-middle", 0.05, 0.01, 0.08, 0.85),
        make_pair("dominant", "partner-best", 0.05, 0.01, 0.10, 0.85),
        make_pair("dominant", "partner-low", 0.05, 0.01, 0.04, 0.85),
        make_pair("dominant", "partner-medium", 0.05, 0.01, 0.07, 0.85),
    ];

    let deduplicated = deduplicate_by_dominant_neuron(pairs);

    assert!(
        deduplicated.len() <= 3,
        "Expected at most 3, got {}",
        deduplicated.len()
    );

    // The best combined improvement (0.10) should be present
    assert!(
        deduplicated
            .iter()
            .any(|p| (p.combined_improvement - 0.10).abs() < 0.001),
        "Best pair (combined 0.10) should be kept"
    );
}

/// Test: Empty input returns empty output.
#[test]
fn epistatic_deduplication_empty_input() {
    let deduplicated = deduplicate_by_dominant_neuron(Vec::new());
    assert!(deduplicated.is_empty());
}

/// Test: Single pair passes through unchanged.
#[test]
fn epistatic_deduplication_single_pair() {
    let pairs = vec![make_pair("a", "b", 0.05, 0.01, 0.08, 0.9)];
    let deduplicated = deduplicate_by_dominant_neuron(pairs);
    assert_eq!(deduplicated.len(), 1);
}

/// Test: Mixed groups — some dominant neurons are shared, some are unique.
#[test]
fn epistatic_deduplication_mixed_groups() {
    let pairs = vec![
        // Group 1: "dominant-x" has 5 pairs — should be capped to 3
        make_pair("dominant-x", "partner-0", 0.05, 0.01, 0.08, 0.85),
        make_pair("dominant-x", "partner-1", 0.05, 0.01, 0.09, 0.85),
        make_pair("dominant-x", "partner-2", 0.05, 0.01, 0.07, 0.85),
        make_pair("dominant-x", "partner-3", 0.05, 0.01, 0.10, 0.85),
        make_pair("dominant-x", "partner-4", 0.05, 0.01, 0.06, 0.85),
        // Group 2: "dominant-y" has 2 pairs — both should survive
        make_pair("dominant-y", "partner-5", 0.04, 0.01, 0.07, 0.85),
        make_pair("dominant-y", "partner-6", 0.04, 0.01, 0.06, 0.85),
        // Group 3: "dominant-z" has 1 pair — passes through
        make_pair("dominant-z", "partner-7", 0.03, 0.01, 0.05, 0.85),
    ];

    let deduplicated = deduplicate_by_dominant_neuron(pairs);

    // Group 1: capped to 3, Group 2: keeps 2, Group 3: keeps 1 → total ≤ 6
    assert!(
        deduplicated.len() <= 6,
        "Expected at most 6, got {}",
        deduplicated.len()
    );
    // Group 3 should still have its unique pair
    assert!(
        deduplicated
            .iter()
            .any(|p| p.source_a_uuid == "dominant-z" || p.source_b_uuid == "dominant-z"),
        "Unique group (dominant-z) should survive"
    );
}

/// Test: Synergistic candidates with shared dominant (primary) are capped.
#[test]
fn synergistic_with_shared_primary_are_capped() {
    let candidates: Vec<SynergisticCandidate> = (0..8)
        .map(|i| {
            make_synergistic(
                "primary-dominant",
                &format!("complement-{i}"),
                0.05,
                0.01 + i as f32 * 0.001,
                0.08 + i as f32 * 0.001,
            )
        })
        .collect();

    let deduplicated = deduplicate_synergistic_by_dominant_neuron(candidates);

    assert!(
        deduplicated.len() <= 3,
        "Expected at most 3 synergistic candidates for same primary, got {}",
        deduplicated.len()
    );
    assert!(
        !deduplicated.is_empty(),
        "Should keep at least one synergistic candidate"
    );
}

/// Test: Synergistic candidates with distinct primaries are not affected.
#[test]
fn synergistic_with_distinct_primaries_unchanged() {
    let candidates = vec![
        make_synergistic("primary-a", "complement-a", 0.05, 0.01, 0.08),
        make_synergistic("primary-b", "complement-b", 0.05, 0.01, 0.08),
        make_synergistic("primary-c", "complement-c", 0.05, 0.01, 0.08),
    ];

    let deduplicated = deduplicate_synergistic_by_dominant_neuron(candidates);

    assert_eq!(
        deduplicated.len(),
        3,
        "All synergistic candidates have distinct primaries — none should be removed"
    );
}

/// Test: Dominant neuron is determined by higher individual improvement.
///
/// When `source_a` has lower improvement than `source_b`, `source_b` is the dominant.
/// Pairs should be grouped by actual dominant neuron, not always by `source_a`.
#[test]
fn epistatic_dominant_is_higher_individual_improvement() {
    // Here "partner-x" is actually the dominant because it has higher improvement
    let pairs = vec![
        make_pair("weak", "partner-x", 0.01, 0.06, 0.08, 0.85),
        make_pair("also-weak", "partner-x", 0.01, 0.06, 0.09, 0.85),
        make_pair("another-weak", "partner-x", 0.01, 0.06, 0.07, 0.85),
        make_pair("yet-another", "partner-x", 0.01, 0.06, 0.10, 0.85),
        make_pair("one-more", "partner-x", 0.01, 0.06, 0.06, 0.85),
    ];

    let deduplicated = deduplicate_by_dominant_neuron(pairs);

    assert!(
        deduplicated.len() <= 3,
        "partner-x is the dominant neuron in all pairs — should be capped to 3, got {}",
        deduplicated.len()
    );
}
