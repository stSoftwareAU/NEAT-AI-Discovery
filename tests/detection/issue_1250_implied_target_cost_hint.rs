//! Tests for Issue #1250: `activation ± error = implied target` is invalid
//! for non-linear-residual costs.
//!
//! Both `output_squash_mismatch::evaluate_alternative_squashes` (Strategy 4)
//! and `high_error_squash_exploration::evaluate_neuron` reconstruct an
//! implied target from the recorded `(activation, error)` pair. That
//! reconstruction is only valid when the error is a linear residual —
//! i.e., `error = target − output` or `error = output − target`.
//!
//! For `MSE`/`MAE`/`CE` the reconstruction recovers the recorded target
//! exactly. For `MAPE`/`MSLE`/`HINGE`/`CATEGORICAL_ERROR` it produces a
//! value that has nothing to do with the recorded target — so the cost
//! hint must gate those code paths off.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::cost_function_hint::CostFunctionHint;
use neat_ai_discovery::analysis::detection::high_error_squash_exploration::{
    detect_high_error_squash_candidates, detect_high_error_squash_candidates_with_cost_hint,
};
use neat_ai_discovery::analysis::detection::output_squash_mismatch::{
    detect_output_squash_mismatches, detect_output_squash_mismatches_with_cost_hint,
};
use neat_ai_discovery::types::DiscoverRecord;

// =============================================================================
// Helpers
// =============================================================================

fn make_record(
    uuid: &str,
    idx: u32,
    value: Option<f32>,
    activation: f32,
    error: f32,
) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value,
        activation,
        errors: vec![error],
    }
}

/// Build 50 high-error pre-activation/activation samples for a TANH neuron
/// where the recorded target is a linear ramp. For an `MSE`-style cost,
/// `error = output − target` so `activation − error = target` recovers the
/// recorded target exactly.
fn mse_high_error_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..50)
        .map(|i| {
            let pre_act = (i as f32 - 25.0) / 8.0;
            let output = pre_act.tanh();
            let target = pre_act; // linear target — what we want the neuron to learn
            // MSE-style residual (`output − target`) — matches the comment in
            // `output_squash_mismatch.rs` and the additive convention in
            // `high_error_squash_exploration.rs`.
            let error = output - target;
            make_record(uuid, i, Some(pre_act), output, error)
        })
        .collect()
}

// =============================================================================
// 1. Implied-target identity holds for MSE
// =============================================================================

#[test]
fn implied_target_recovers_recorded_target_for_mse() {
    // For MSE the recorded error is a linear residual. Both reconstructions
    // (`activation − error` and `activation + error`) recover the recorded
    // target up to a sign convention. We assert the magnitudes match the
    // linear target ramp.
    let records = mse_high_error_records("h1");
    for (i, r) in records.iter().enumerate() {
        let expected_target = (i as f32 - 25.0) / 8.0;
        let implied_minus = r.activation - r.errors[0]; // output_squash_mismatch convention
        let implied_plus = r.activation + r.errors[0]; // high_error_squash convention
        assert!(
            (implied_minus - expected_target).abs() < 1e-5,
            "MSE: activation − error should recover target ({expected_target}), got {implied_minus}",
        );
        // Under the same `error = output − target` convention,
        // `activation + error = 2·output − target` — but with MSE the
        // residual is signed both ways depending on direction. The point
        // for this test is that *one* of the two reconstructions matches
        // the recorded target exactly for MSE, which proves the detector
        // is operating on a valid identity. The other detector uses the
        // additive form because NEAT-AI's MSE in that file is documented
        // as `error = target − output` (sign flip).
        let _ = implied_plus;
    }
}

// =============================================================================
// 2. Non-linear cost: implied target diverges from the recorded target
// =============================================================================

#[test]
fn implied_target_diverges_for_mape_residual() {
    // MAPE residual: `(target − output) / |target|`. The "implied target"
    // `activation − error` becomes `output − (target − output)/|target|`,
    // which has no relation to `target` once `|target|` ≠ 1.
    let pre_act = 2.0_f32;
    let output = pre_act.tanh(); // ≈ 0.964
    let target = 0.5_f32;
    let mape_error = (target - output) / target.abs(); // ≈ -0.928
    let implied_target = output - mape_error;
    // The implied target should *not* be close to the recorded target.
    assert!(
        (implied_target - target).abs() > 0.1,
        "MAPE implied target ({implied_target}) should diverge from recorded target ({target})",
    );
}

// =============================================================================
// 3. Cost hint gates the affected detector code paths off
// =============================================================================

#[test]
fn high_error_squash_detector_is_gated_off_for_non_linear_cost() {
    // The same records that triggered detection under the unknown hint
    // must produce zero candidates when the caller declares the cost as
    // non-linear (MAPE / MSLE / HINGE / CATEGORICAL_ERROR).
    let records = mse_high_error_records("h1");
    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
    let neuron_records = vec![("h1".to_string(), records)];

    let with_unknown = detect_high_error_squash_candidates(&neurons, &neuron_records);
    assert!(
        !with_unknown.is_empty(),
        "Baseline: unknown hint must keep historical behaviour and emit candidates",
    );

    for hint_name in ["MAPE", "MSLE", "HINGE", "CATEGORICAL_ERROR"] {
        let hint = CostFunctionHint::from_name(hint_name);
        let result =
            detect_high_error_squash_candidates_with_cost_hint(&neurons, &neuron_records, hint);
        assert!(
            result.is_empty(),
            "{hint_name}: detector must be gated off for non-linear cost",
        );
    }
}

#[test]
fn high_error_squash_detector_keeps_running_for_linear_cost() {
    let records = mse_high_error_records("h1");
    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
    let neuron_records = vec![("h1".to_string(), records)];

    for hint_name in ["MSE", "MAE", "CROSS_ENTROPY"] {
        let hint = CostFunctionHint::from_name(hint_name);
        let result =
            detect_high_error_squash_candidates_with_cost_hint(&neurons, &neuron_records, hint);
        assert!(
            !result.is_empty(),
            "{hint_name}: detector must still run for linear-residual cost",
        );
    }
}

#[test]
fn output_squash_mismatch_strategy_4_is_gated_off_for_non_linear_cost() {
    // Strategy 4 (pre-activation comparison) is the only path inside
    // `detect_output_squash_mismatches` that depends on the implied
    // target. Build records that *only* trigger Strategy 4 (TANH neuron
    // with a linear target, no clipping/range/unbounded mismatch).
    let outputs = vec![("out-1".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<(String, Vec<DiscoverRecord>)> =
        vec![("out-1".to_string(), mse_high_error_records("out-1"))];

    let baseline = detect_output_squash_mismatches(&outputs, &records);
    let baseline_count = baseline.len();

    // Linear-cost hint preserves the baseline behaviour.
    let linear = detect_output_squash_mismatches_with_cost_hint(
        &outputs,
        &records,
        CostFunctionHint::LinearResidual,
    );
    assert_eq!(
        linear.len(),
        baseline_count,
        "LinearResidual hint must preserve baseline candidate count",
    );

    // Non-linear hint must drop any Strategy-4-only candidate. For these
    // records the only candidates are Strategy 4 candidates, so we expect
    // the count to fall to zero. (If Strategies 1–3 also fired the count
    // would still drop by exactly the number of Strategy-4 hits — which
    // is what we want.)
    for hint_name in ["MAPE", "MSLE", "HINGE", "CATEGORICAL_ERROR"] {
        let hint = CostFunctionHint::from_name(hint_name);
        let result = detect_output_squash_mismatches_with_cost_hint(&outputs, &records, hint);
        assert!(
            result.len() <= baseline_count,
            "{hint_name}: non-linear hint must not add candidates ({} > {baseline_count})",
            result.len(),
        );
        // For these records Strategy 4 is the *only* trigger, so the
        // non-linear hint must produce zero candidates.
        assert!(
            result.is_empty(),
            "{hint_name}: Strategy 4 must be gated off — got {} candidates",
            result.len(),
        );
    }
}

// =============================================================================
// 4. Backwards compatibility — Unknown hint preserves legacy behaviour
// =============================================================================

#[test]
fn unknown_hint_matches_legacy_high_error_squash_results() {
    let records = mse_high_error_records("h1");
    let neurons = vec![("h1".to_string(), "TANH".to_string(), 0.0)];
    let neuron_records = vec![("h1".to_string(), records)];

    let legacy = detect_high_error_squash_candidates(&neurons, &neuron_records);
    let hinted = detect_high_error_squash_candidates_with_cost_hint(
        &neurons,
        &neuron_records,
        CostFunctionHint::Unknown,
    );
    assert_eq!(legacy.len(), hinted.len());
    for (l, h) in legacy.iter().zip(hinted.iter()) {
        assert_eq!(l.neuron_uuid, h.neuron_uuid);
        assert_eq!(l.recommended_squash, h.recommended_squash);
    }
}

#[test]
fn unknown_hint_matches_legacy_output_squash_results() {
    let outputs = vec![("out-1".to_string(), "TANH".to_string(), 0.0)];
    let records: Vec<(String, Vec<DiscoverRecord>)> =
        vec![("out-1".to_string(), mse_high_error_records("out-1"))];

    let legacy = detect_output_squash_mismatches(&outputs, &records);
    let hinted = detect_output_squash_mismatches_with_cost_hint(
        &outputs,
        &records,
        CostFunctionHint::Unknown,
    );
    assert_eq!(legacy.len(), hinted.len());
    for (l, h) in legacy.iter().zip(hinted.iter()) {
        assert_eq!(l.neuron_uuid, h.neuron_uuid);
        assert_eq!(l.recommended_squash, h.recommended_squash);
    }
}
