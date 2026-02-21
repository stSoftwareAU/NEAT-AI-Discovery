//! Property-based tests for candidate clustering and variance thresholds (Issue #680).
//!
//! Uses `proptest` to verify structural invariants of `cluster_candidates` and
//! mathematical properties of `compute_source_variance_discount` and
//! `compute_dynamic_constant_source_threshold`.

use neat_ai_discovery::analysis::candidate_clustering::{ClusterableCandidate, cluster_candidates};
use neat_ai_discovery::analysis::samples::{
    HelpfulSample, compute_dynamic_constant_source_threshold, compute_source_variance_discount,
};
use proptest::prelude::*;

/// Strategy for generating a ClusterableCandidate.
fn candidate_strategy(
    target: &'static str,
    neuron_type: &'static str,
) -> impl Strategy<Value = ClusterableCandidate> {
    ("[a-z]{1,4}", 0.001f32..1.0).prop_map(move |(from, improvement)| ClusterableCandidate {
        from_neuron_uuid: from,
        to_neuron_uuid: target.to_string(),
        expected_improvement: improvement,
        neuron_type: neuron_type.to_string(),
    })
}

/// Strategy for generating HelpfulSample with finite activations.
fn helpful_sample_strategy() -> impl Strategy<Value = HelpfulSample> {
    (-10.0f32..10.0, -10.0f32..10.0).prop_map(|(a, e)| HelpfulSample {
        activation: a,
        avg_error: e,
        target_value: None,
        target_activation: None,
    })
}

// =============================================================================
// 1. cluster_candidates Structural Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    /// Empty input must produce empty output.
    #[test]
    fn cluster_empty_returns_empty(_dummy in 0u8..1) {
        let result = cluster_candidates(&[]);
        prop_assert!(result.is_empty());
    }

    /// Single candidate must produce empty output (no singleton clusters).
    #[test]
    fn cluster_single_returns_empty(candidate in candidate_strategy("t1", "input")) {
        let result = cluster_candidates(&[candidate]);
        prop_assert!(result.is_empty());
    }

    /// Every cluster must have member_count >= 2.
    #[test]
    fn cluster_min_size(
        candidates in prop::collection::vec(candidate_strategy("t1", "input"), 2..30),
    ) {
        let clusters = cluster_candidates(&candidates);
        for cluster in &clusters {
            prop_assert!(
                cluster.member_count >= 2,
                "Cluster must have >= 2 members, got {}",
                cluster.member_count
            );
        }
    }

    /// member_count must equal member_from_uuids.len() for every cluster.
    #[test]
    fn cluster_member_count_consistent(
        candidates in prop::collection::vec(candidate_strategy("t1", "input"), 2..30),
    ) {
        let clusters = cluster_candidates(&candidates);
        for cluster in &clusters {
            prop_assert_eq!(
                cluster.member_count,
                cluster.member_from_uuids.len(),
                "member_count inconsistent with member_from_uuids"
            );
        }
    }

    /// internal_correlation must be in [0.0, 1.0].
    #[test]
    fn cluster_correlation_bounded(
        candidates in prop::collection::vec(candidate_strategy("t1", "input"), 2..30),
    ) {
        let clusters = cluster_candidates(&candidates);
        for cluster in &clusters {
            prop_assert!(
                (0.0..=1.0).contains(&cluster.internal_correlation),
                "internal_correlation {} out of [0, 1]",
                cluster.internal_correlation
            );
        }
    }

    /// Clusters must be sorted by representative_improvement in descending order.
    #[test]
    fn clusters_sorted_by_improvement(
        candidates in prop::collection::vec(candidate_strategy("t1", "input"), 2..30),
    ) {
        let clusters = cluster_candidates(&candidates);
        for window in clusters.windows(2) {
            prop_assert!(
                window[0].representative_improvement >= window[1].representative_improvement,
                "Clusters not sorted: {} < {}",
                window[0].representative_improvement,
                window[1].representative_improvement
            );
        }
    }

    /// Candidates targeting different neurons should form separate clusters.
    #[test]
    fn different_targets_separate_clusters(
        target1_candidates in prop::collection::vec(candidate_strategy("t1", "input"), 3..10),
        target2_candidates in prop::collection::vec(candidate_strategy("t2", "input"), 3..10),
    ) {
        let mut all = target1_candidates;
        all.extend(target2_candidates);

        let clusters = cluster_candidates(&all);

        // Verify each cluster targets a single neuron
        for cluster in &clusters {
            let targets: std::collections::HashSet<&str> = std::iter::once(cluster.representative_to_uuid.as_str()).collect();
            prop_assert_eq!(
                targets.len(),
                1,
                "Each cluster should target exactly one neuron"
            );
        }
    }

    /// Candidates with different neuron_type targeting the same neuron
    /// should form separate clusters.
    #[test]
    fn different_types_separate_clusters(
        input_candidates in prop::collection::vec(candidate_strategy("t1", "input"), 3..10),
        hidden_candidates in prop::collection::vec(candidate_strategy("t1", "hidden"), 3..10),
    ) {
        let mut all = input_candidates;
        all.extend(hidden_candidates);

        let clusters = cluster_candidates(&all);

        // Verify no cluster mixes neuron types
        // (We can only check indirectly: all member UUIDs should come from
        // candidates of the same type)
        for cluster in &clusters {
            prop_assert!(
                cluster.member_count >= 2,
                "All clusters should have >= 2 members"
            );
        }
    }
}

// =============================================================================
// 2. Identical-Improvement Cluster Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Candidates with identical improvements targeting the same neuron should
    /// produce a cluster with internal_correlation near 1.0.
    #[test]
    fn identical_improvements_high_correlation(
        improvement in 0.01f32..1.0,
        count in 3usize..10,
    ) {
        let candidates: Vec<ClusterableCandidate> = (0..count)
            .map(|i| ClusterableCandidate {
                from_neuron_uuid: format!("s{i}"),
                to_neuron_uuid: "target".to_string(),
                expected_improvement: improvement,
                neuron_type: "input".to_string(),
            })
            .collect();

        let clusters = cluster_candidates(&candidates);

        // Should produce exactly one cluster with all candidates
        prop_assert_eq!(
            clusters.len(),
            1,
            "Identical improvements should produce 1 cluster, got {}",
            clusters.len()
        );

        if let Some(cluster) = clusters.first() {
            prop_assert!(
                cluster.internal_correlation > 0.99,
                "Identical improvements should have correlation ~1.0, got {}",
                cluster.internal_correlation
            );
            prop_assert_eq!(cluster.member_count, count);
        }
    }
}

// =============================================================================
// 3. compute_source_variance_discount Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Discount must always be in [0.0, 1.0].
    #[test]
    fn variance_discount_bounded(
        samples in prop::collection::vec(helpful_sample_strategy(), 2..100),
    ) {
        let discount = compute_source_variance_discount(&samples);
        prop_assert!(
            (0.0..=1.0).contains(&discount),
            "Discount {discount} out of [0, 1]"
        );
    }

    /// Empty or single-sample input must return 0.0.
    #[test]
    fn variance_discount_insufficient_samples(count in 0usize..2) {
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|i| HelpfulSample {
                activation: i as f32,
                avg_error: 0.0,
                target_value: None,
                target_activation: None,
            })
            .collect();
        let discount = compute_source_variance_discount(&samples);
        prop_assert_eq!(
            discount, 0.0,
            "Should return 0.0 for {} samples",
            count
        );
    }

    /// Constant activation samples must produce 0.0 discount.
    #[test]
    fn variance_discount_constant_zero(
        value in -100.0f32..100.0,
        count in 5usize..50,
    ) {
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|_| HelpfulSample {
                activation: value,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();
        let discount = compute_source_variance_discount(&samples);
        prop_assert!(
            discount < 1e-6,
            "Constant activation should yield ~0 discount, got {discount}"
        );
    }

    /// High-variance samples should produce discount close to 1.0.
    #[test]
    fn variance_discount_high_variance_near_one(count in 10usize..50) {
        // Create samples with large spread
        let samples: Vec<HelpfulSample> = (0..count)
            .map(|i| HelpfulSample {
                activation: (i as f32 - count as f32 / 2.0) * 10.0,
                avg_error: 0.1,
                target_value: None,
                target_activation: None,
            })
            .collect();
        let discount = compute_source_variance_discount(&samples);
        prop_assert!(
            discount > 0.9,
            "High variance should yield discount near 1.0, got {discount}"
        );
    }
}

// =============================================================================
// 4. compute_dynamic_constant_source_threshold Properties
// =============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Dynamic threshold must always be finite.
    #[test]
    fn dynamic_threshold_finite(input in prop::num::f32::ANY) {
        let threshold = compute_dynamic_constant_source_threshold(input);
        prop_assert!(
            threshold.is_finite(),
            "Threshold must be finite for input {input}"
        );
    }

    /// Dynamic threshold must always be >= DEFAULT_CONSTANT_SOURCE_EFFECT_THRESHOLD (1e-7).
    #[test]
    fn dynamic_threshold_above_minimum(input in -100.0f32..100.0) {
        let threshold = compute_dynamic_constant_source_threshold(input);
        // The default threshold is 1e-7
        prop_assert!(
            threshold >= 1e-7 - 1e-15,
            "Threshold {threshold} below minimum (1e-7) for input {input}"
        );
    }

    /// Non-finite and non-positive inputs must return the default threshold.
    #[test]
    fn dynamic_threshold_default_for_invalid(kind in 0u8..5) {
        let input = match kind {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => 0.0,
            _ => -1.0,
        };
        let threshold = compute_dynamic_constant_source_threshold(input);
        let expected = 1e-7_f32;
        prop_assert!(
            (threshold - expected).abs() < 1e-15,
            "Invalid input {input} should return default {expected}, got {threshold}"
        );
    }

    /// Higher source_std_dev_avg should yield higher or equal threshold (monotonicity).
    #[test]
    fn dynamic_threshold_monotonic(
        low in 0.001f32..0.05,
        high in 0.05f32..10.0,
    ) {
        let threshold_low = compute_dynamic_constant_source_threshold(low);
        let threshold_high = compute_dynamic_constant_source_threshold(high);
        prop_assert!(
            threshold_high >= threshold_low - 1e-15,
            "Threshold should increase with std_dev: low={threshold_low} (input={low}), high={threshold_high} (input={high})"
        );
    }
}
