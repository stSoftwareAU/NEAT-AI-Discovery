//! Regression tests for Issue #1272 — operation-count-aware post-discount
//! noise floor for coordinated-structural candidates.
//!
//! Production failure-cache analysis captured two 4-op
//! coordinated-structural failures that **just barely** cleared the previous
//! single-value `COORDINATED_POST_DISCOUNT_NOISE_FLOOR` (5e-7) but harmed the
//! network by 1000× to 6000× the predicted magnitude:
//!
//! | Failure entry         | ops | predicted | actual    |
//! |-----------------------|-----|-----------|-----------|
//! | `cdff1014…`           | 4   | 5.06e-7   | -3.29e-3  |
//! | `113ec32a…`           | 4   | 5.21e-7   | -8.28e-4  |
//!
//! The `coordinated_empirical_discount()` lookup (#1058) already encodes that
//! 4-op candidates have near-zero production success. The same evidence
//! supports a stricter noise floor at the per-tier level — 1-op stays at
//! 5e-7, 2-op rises to 1e-6, 3-op to 2e-6, and 4+-op to 5e-6.

use neat_ai_discovery::analysis::candidate_aggregation::apply_coordinated_gain_floor;
use neat_ai_discovery::analysis::constants::{
    COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP, COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS,
    COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS, COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS,
    coordinated_post_discount_noise_floor,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

fn candidate_with_ops(
    op_count: usize,
    gain: f32,
    label: &str,
) -> CoordinatedStructuralCandidateJson {
    let mut operations = Vec::with_capacity(op_count.max(1));
    // First op is always a RemoveSynapse; subsequent ops alternate to keep
    // the candidate structurally distinct without exercising the rest of the
    // pipeline.
    operations.push(CoordinatedStructuralOpJson::RemoveSynapse {
        from_neuron_uuid: "a".to_string(),
        to_neuron_uuid: "b".to_string(),
    });
    for i in 1..op_count {
        operations.push(CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: format!("src{i}"),
            to_neuron_uuid: format!("dst{i}"),
            weight: 0.5,
        });
    }
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations,
        expected_creature_score_gain: gain,
        comment: Some(label.to_string()),
    }
}

#[test]
fn per_tier_constants_have_expected_values() {
    assert!((COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP - 5e-7).abs() < f32::EPSILON);
    assert!((COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS - 1e-6).abs() < 1e-12);
    assert!((COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS - 2e-6).abs() < 1e-12);
    assert!((COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS - 5e-6).abs() < 1e-12);
}

#[test]
fn helper_returns_per_tier_values_for_each_op_count() {
    // 0-op is treated as the single-op tier — the discount helper has the
    // same convention (`0 | 1 => 1.0`).
    assert_eq!(
        coordinated_post_discount_noise_floor(0),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP
    );
    assert_eq!(
        coordinated_post_discount_noise_floor(1),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP
    );
    assert_eq!(
        coordinated_post_discount_noise_floor(2),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS
    );
    assert_eq!(
        coordinated_post_discount_noise_floor(3),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS
    );
    assert_eq!(
        coordinated_post_discount_noise_floor(4),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS
    );
    assert_eq!(
        coordinated_post_discount_noise_floor(5),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS
    );
    assert_eq!(
        coordinated_post_discount_noise_floor(128),
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS
    );
}

/// The dispatch filter applies the per-tier floor, not a single global value.
/// A 2-op candidate that would have cleared the legacy 5e-7 floor is now
/// rejected against the 1e-6 2-op floor.
#[test]
fn dispatch_filter_applies_per_tier_floor() {
    let mut cands = vec![
        // 1-op at exactly its tier floor — retained.
        candidate_with_ops(1, COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP, "1op-at-floor"),
        // 2-op at 6e-7 — above the legacy 5e-7 but below the new 1e-6 2-op floor.
        candidate_with_ops(2, 6e-7, "2op-just-above-legacy"),
        // 3-op at 1.5e-6 — above the 1e-6 2-op floor but below the 2e-6 3-op floor.
        candidate_with_ops(3, 1.5e-6, "3op-below-tier"),
        // 4-op at 3e-6 — above the 3-op floor but below the 5e-6 4+-op floor.
        candidate_with_ops(4, 3e-6, "4op-below-tier"),
        // 4-op well above the 4+-op floor — retained.
        candidate_with_ops(4, 1.0, "4op-well-above"),
    ];
    apply_coordinated_gain_floor(&mut cands);

    let labels: Vec<&str> = cands
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect();
    assert!(labels.contains(&"1op-at-floor"), "labels={labels:?}");
    assert!(labels.contains(&"4op-well-above"), "labels={labels:?}");
    assert!(!labels.contains(&"2op-just-above-legacy"));
    assert!(!labels.contains(&"3op-below-tier"));
    assert!(!labels.contains(&"4op-below-tier"));
    assert_eq!(cands.len(), 2);
}

/// Regression: the two 4-op `bcbca347` entries from Issue #1272 evidence
/// — 5.06e-7 and 5.21e-7, which both **cleared** the legacy 5e-7 floor —
/// are now rejected under the new 5e-6 4+-op floor.
#[test]
fn bcbca347_4op_entries_are_filtered() {
    let mut cands = vec![
        candidate_with_ops(4, 5.06e-7, "cdff1014"),
        candidate_with_ops(4, 5.21e-7, "113ec32a"),
    ];
    apply_coordinated_gain_floor(&mut cands);
    assert!(
        cands.is_empty(),
        "bcbca347 4-op entries must be rejected under the new 4+-op floor, \
         survivors: {cands:?}"
    );
}

/// 1-op candidates retain their existing 5e-7 floor — no behavioural change
/// for the most common candidate tier.
#[test]
fn one_op_floor_unchanged_at_5e_minus_7() {
    let mut cands = vec![
        candidate_with_ops(1, 5e-7, "at-floor"),
        candidate_with_ops(1, 4.99e-7, "below-floor"),
        candidate_with_ops(1, 1.0, "well-above"),
    ];
    apply_coordinated_gain_floor(&mut cands);
    let labels: Vec<&str> = cands
        .iter()
        .map(|c| c.comment.as_deref().unwrap_or(""))
        .collect();
    assert!(labels.contains(&"at-floor"));
    assert!(labels.contains(&"well-above"));
    assert!(!labels.contains(&"below-floor"));
}

/// The deprecated `COORDINATED_POST_DISCOUNT_NOISE_FLOOR` alias still maps
/// to the 1-op value so external callers continue to build.
#[test]
#[allow(deprecated)]
fn legacy_alias_maps_to_one_op_floor() {
    use neat_ai_discovery::analysis::constants::COORDINATED_POST_DISCOUNT_NOISE_FLOOR;
    assert_eq!(
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR,
        COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP
    );
}
