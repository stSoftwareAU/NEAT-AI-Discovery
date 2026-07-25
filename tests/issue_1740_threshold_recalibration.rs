//! Threshold review and false-positive guards for the production network
//! (Issue #1740).
//!
//! Issue #1740 asked whether the coordinated expected-gain floors
//! (`COORDINATED_MIN_EXPECTED_GAIN`, `MIN_COORDINATED_MULTI_OP_GAIN`) and the
//! per-op post-discount noise floors are over-rejecting genuine improvements on
//! the large converged production network.
//!
//! **Review conclusion (see `docs/analysis/threshold-review-1740.md`): the
//! floors are correctly scaled and must NOT be lowered.** The diagnosis
//! (#1737) showed the blocker is the *expected-gain estimator* collapsing to
//! `~1e-10` / `0` — uncorrelated with the realised delta and frequently the
//! wrong sign — not the floor scale. Lowering the floors to admit the
//! achievable `~1e-7` band would necessarily admit noise-level candidates such
//! as the production coordinated `change-squash` whose `4e-10` estimate masked
//! a realised `-8.65e-4` (a harmful change).
//!
//! These tests are the false-positive guard the issue's Failure Detection
//! section mandates: they pin the reviewed floor values and prove that a
//! candidate at the production achievable-gain magnitude is still rejected at the gain
//! gate, so a future "just lower the floor" change fails here before merge.

use neat_ai_discovery::analysis::candidate_aggregation::{
    apply_coordinated_gain_floor, apply_operation_count_discount,
    validate_coordinated_candidate_gain,
};
use neat_ai_discovery::analysis::constants::{
    COORDINATED_MIN_EXPECTED_GAIN, COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP,
    COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS, COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS,
    COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS, MIN_COORDINATED_MULTI_OP_GAIN,
    coordinated_post_discount_noise_floor,
};
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};
use std::hint::black_box;

/// Read a constant through an optimisation barrier so the comparisons below are
/// genuine runtime assertions, not compile-time-constant ones (which
/// `clippy::assertions_on_constants` rejects under `-D warnings`).
fn rt(value: f32) -> f32 {
    black_box(value)
}

/// Magnitude of the only accepted changes on the converged production creature
/// (diagnosis #1737: the accepted `remove-neuron` realised `+1.95e-7`). This is
/// the "achievable-improvement band" the issue asks whether the floors reject.
const PRODUCTION_ACHIEVABLE_GAIN: f32 = 1.95e-7;

/// The production coordinated `change-squash` estimate captured in the diagnosis
/// (#1737): a `4.17e-10` expected gain that masked a realised `-8.65e-4`.
const PRODUCTION_REJECTED_MULTI_OP_GAIN: f32 = 4.165_739e-10;

fn single_op_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some("issue-1740".to_string()),
    }
}

fn two_op_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "b".to_string(),
            },
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid: "a".to_string(),
                to_neuron_uuid: "c".to_string(),
                weight: 0.5,
            },
        ],
        expected_creature_score_gain: gain,
        comment: Some("issue-1740".to_string()),
    }
}

// =============================================================================
// Floor values reviewed under #1740 are unchanged (recalibration pin)
// =============================================================================

/// The review concluded no floor change is warranted. Pin the reviewed values
/// so an accidental (or naive "just lower it") loosening fails here — the
/// issue's Failure Detection contract requires a test asserting the floor
/// values that survive the review.
#[test]
fn reviewed_floor_values_are_unchanged() {
    assert!(
        (rt(COORDINATED_MIN_EXPECTED_GAIN) - 1e-5).abs() < f32::EPSILON,
        "COORDINATED_MIN_EXPECTED_GAIN reviewed value is 1e-5, got {COORDINATED_MIN_EXPECTED_GAIN}",
    );
    assert!(
        (rt(MIN_COORDINATED_MULTI_OP_GAIN) - 1e-5).abs() < f32::EPSILON,
        "MIN_COORDINATED_MULTI_OP_GAIN reviewed value is 1e-5, got {MIN_COORDINATED_MULTI_OP_GAIN}",
    );
    // Per-op post-discount noise floors (Issue #1272) — the tiering reviewed
    // under #1740 must stay strictly increasing with op-count.
    assert!(
        (rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP) - 5e-7).abs() < f32::EPSILON,
        "1-op noise floor reviewed value is 5e-7",
    );
    assert!(
        (rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS) - 1e-6).abs() < f32::EPSILON,
        "2-op noise floor reviewed value is 1e-6",
    );
    assert!(
        (rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS) - 2e-6).abs() < f32::EPSILON,
        "3-op noise floor reviewed value is 2e-6",
    );
    assert!(
        (rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS) - 5e-6).abs() < f32::EPSILON,
        "4+-op noise floor reviewed value is 5e-6",
    );
    assert!(
        rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP) < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS
            && rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_2OPS)
                < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS
            && rt(COORDINATED_POST_DISCOUNT_NOISE_FLOOR_3OPS)
                < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_4PLUS_OPS,
        "per-op noise floors must increase with implementation risk (op-count)",
    );
}

// =============================================================================
// False-positive guard — the production achievable band sits in the noise floor
// =============================================================================

/// The core review finding: the only accepted improvements on the converged
/// production creature realise `~1.95e-7`, which is **below** the 1-op
/// post-discount noise floor (`5e-7`). A candidate whose *estimated* gain sits
/// in that band is therefore rejected at the gain gate — it can only reach
/// acceptance via the separate error-magnitude ranking path (which bypasses the
/// gain floor), exactly as the diagnosis observed for the accepted
/// `remove-neuron`.
///
/// This proves lowering the floor to "admit the achievable band" would drag the
/// floor into the noise band and admit false positives — so the floor is
/// correctly scaled and must not be lowered while the estimator is uncalibrated.
#[test]
fn achievable_band_estimate_is_below_the_1op_noise_floor() {
    assert!(
        rt(PRODUCTION_ACHIEVABLE_GAIN) < COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP,
        "the production achievable band ({PRODUCTION_ACHIEVABLE_GAIN}) must sit below the 1-op noise floor \
         ({COORDINATED_POST_DISCOUNT_NOISE_FLOOR_1OP}) — otherwise the review premise is wrong",
    );

    let mut candidates = vec![single_op_candidate(PRODUCTION_ACHIEVABLE_GAIN)];
    let dropped = apply_coordinated_gain_floor(&mut candidates);

    assert_eq!(
        dropped, 1,
        "achievable-band estimate must be dropped by the floor"
    );
    assert!(
        candidates.is_empty(),
        "a candidate estimated at the production achievable magnitude must not survive the gain gate",
    );
}

/// The production coordinated `change-squash` failure: a multi-op estimate of
/// `4.17e-10` that masked a realised `-8.65e-4`. It must be rejected by the
/// multi-op gain validation — lowering `MIN_COORDINATED_MULTI_OP_GAIN` to admit
/// this magnitude would admit a documented *harmful* candidate.
#[test]
fn production_multi_op_change_squash_noise_is_rejected() {
    let candidate = two_op_candidate(PRODUCTION_REJECTED_MULTI_OP_GAIN);

    // Discount is applied to multi-op candidates before the floor compare.
    let discounted = apply_operation_count_discount(&candidate);
    assert!(
        discounted <= MIN_COORDINATED_MULTI_OP_GAIN,
        "precondition: discounted multi-op gain ({discounted}) must be below the multi-op floor",
    );

    assert!(
        !validate_coordinated_candidate_gain(&candidate),
        "a 4e-10 multi-op change-squash (realised -8.65e-4 in production) must be rejected",
    );
}

/// The multi-op noise floor also rejects a candidate that is only a little below
/// the floor, guarding the boundary rather than just the extreme case.
#[test]
fn multi_op_candidate_just_below_floor_is_rejected() {
    // 2-op discount is 0.5×, so a raw gain of 1.5e-5 discounts to 7.5e-6 —
    // below the 1e-5 multi-op floor.
    let candidate = two_op_candidate(1.5e-5);
    assert!(
        !validate_coordinated_candidate_gain(&candidate),
        "a 2-op candidate discounted just below the floor must be rejected",
    );
}

// =============================================================================
// Regression guard — floors do NOT over-reject genuine improvements
// =============================================================================

/// A genuinely promising above-floor candidate must survive both the multi-op
/// validation and the post-discount noise sweep. This guards the *other*
/// direction: the review must not have tightened the floors into rejecting
/// legitimate large-gain candidates.
#[test]
fn above_floor_candidate_survives_all_gates() {
    // Single-op, well above the 1-op noise floor.
    let mut single = vec![single_op_candidate(1e-3)];
    let dropped_single = apply_coordinated_gain_floor(&mut single);
    assert_eq!(
        dropped_single, 0,
        "an above-floor single-op candidate must survive"
    );
    assert_eq!(single.len(), 1);

    // Multi-op: raw 1e-2 discounts to 5e-3 (2-op) — comfortably above 1e-5.
    let multi = two_op_candidate(1e-2);
    assert!(
        validate_coordinated_candidate_gain(&multi),
        "an above-floor multi-op candidate must pass gain validation",
    );
}

/// The per-op noise floor helper must return the reviewed tier for each
/// op-count, including the `0`-op alias to the single-op tier.
#[test]
fn per_op_noise_floor_lookup_matches_reviewed_tiers() {
    assert!((coordinated_post_discount_noise_floor(0) - 5e-7).abs() < f32::EPSILON);
    assert!((coordinated_post_discount_noise_floor(1) - 5e-7).abs() < f32::EPSILON);
    assert!((coordinated_post_discount_noise_floor(2) - 1e-6).abs() < f32::EPSILON);
    assert!((coordinated_post_discount_noise_floor(3) - 2e-6).abs() < f32::EPSILON);
    assert!((coordinated_post_discount_noise_floor(4) - 5e-6).abs() < f32::EPSILON);
    assert!((coordinated_post_discount_noise_floor(9) - 5e-6).abs() < f32::EPSILON);
}
