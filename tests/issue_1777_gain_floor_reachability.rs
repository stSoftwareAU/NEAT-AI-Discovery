//! Issue #1777 — is the acceptance gain floor reachable at all?
//!
//! Diagnostic characterisation of the **expected-gain discount stack** against
//! the acceptance floor it must clear. The #1737 diagnosis found production
//! candidates carrying `expectedCreatureScoreGain` of `~1e-10` or exactly `0`
//! against a `1e-5` floor, and named the estimator as the primary root cause.
//! The #1740 threshold review then confirmed the floor value itself is sound.
//!
//! Neither audit measured the two together. This suite does: it drives the
//! **shipped** discount functions end to end, in the same order and with the
//! same constants as `neuron/post_processing.rs` and `synapse/scoring`, and
//! reports the *break-even raw gain* — the raw creature error reduction a
//! candidate must carry for its post-discount gain to survive the floor.
//!
//! The tests are characterisation-only: they pin today's measured numbers so a
//! later change to the calibration constants or the floor is visible as a test
//! diff. They deliberately assert **no** fix — the remediation is tracked
//! separately (see `docs/analysis/candidate-rate-diagnosis-1777.md`).

use neat_ai_discovery::analysis::constants::{
    MIN_EXPECTED_CREATURE_SCORE_GAIN, NEURON_PREDICTION_CALIBRATION, SYNAPSE_PREDICTION_CALIBRATION,
};
use neat_ai_discovery::analysis::scoring::calibration_correction::MIN_CALIBRATION_CORRECTION;
use neat_ai_discovery::analysis::synapse::scoring::{
    apply_logistic_prediction_calibration, apply_neuron_pessimism_discount,
    apply_saturation_prediction_discount, apply_synapse_pessimism_discount,
};

/// Largest realised score delta observed in the production candidate cache
/// (#1737): accepted changes land around `1e-7`, rejected ones up to `~1e-3`.
/// A candidate whose break-even raw gain sits above this band can never be
/// accepted on a converged network, however good the change actually is.
const OBSERVED_REALISED_DELTA_CEILING: f32 = 1e-3;

/// Run the shipped add-neuron discount stack over a raw creature error
/// reduction, in the order `apply_creature_level_metrics` applies it.
///
/// `impact` is the target's structural impact, `correction` the per-creature
/// calibration correction read from the failure cache (`[0.001, 1.0]`).
fn neuron_post_discount_gain(
    raw_error_reduction: f32,
    impact: f32,
    improved: u32,
    total: u32,
    magnitude_ratio: Option<f32>,
    saturation: Option<f32>,
    correction: f32,
) -> f32 {
    let gain = raw_error_reduction * impact;
    let gain = apply_neuron_pessimism_discount(gain, improved, total, magnitude_ratio);
    let gain = apply_saturation_prediction_discount(gain, saturation);
    apply_logistic_prediction_calibration(
        gain,
        improved,
        total,
        NEURON_PREDICTION_CALIBRATION * correction,
    )
}

/// Run the shipped add-synapse discount stack over a raw creature error
/// reduction.
fn synapse_post_discount_gain(
    raw_error_reduction: f32,
    improved: u32,
    total: u32,
    magnitude_ratio: Option<f32>,
    correction: f32,
) -> f32 {
    let gain = apply_synapse_pessimism_discount(raw_error_reduction, improved, total, magnitude_ratio);
    apply_logistic_prediction_calibration(
        gain,
        improved,
        total,
        SYNAPSE_PREDICTION_CALIBRATION * correction,
    )
}

/// Raw creature error reduction needed for the post-discount gain to reach the
/// acceptance floor, given a multiplier measured from a unit raw gain.
fn break_even_raw_gain(unit_multiplier: f32) -> f32 {
    MIN_EXPECTED_CREATURE_SCORE_GAIN / unit_multiplier
}

#[test]
fn perfect_add_neuron_candidate_needs_a_third_of_a_percent_raw_gain() {
    // The most favourable inputs the pipeline admits: full structural impact,
    // every sample improved, full improvement magnitude, no saturation, and a
    // neutral (never-discounting) calibration correction.
    let multiplier = neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), None, 1.0);

    // Every factor upstream of the calibration constant is <= 1.0, so the
    // constant caps the achievable multiplier. Only the logistic modulator
    // (0.96 at a perfect improved ratio) keeps it fractionally below the cap.
    assert!(
        multiplier <= NEURON_PREDICTION_CALIBRATION,
        "best-case neuron multiplier {multiplier:e} must not exceed \
         NEURON_PREDICTION_CALIBRATION {NEURON_PREDICTION_CALIBRATION:e}"
    );
    assert!(
        multiplier >= NEURON_PREDICTION_CALIBRATION * 0.9,
        "best-case neuron multiplier {multiplier:e} should sit within 10% of \
         the calibration cap {NEURON_PREDICTION_CALIBRATION:e}"
    );

    let break_even = break_even_raw_gain(multiplier);
    // 1e-5 / 3e-3 == 3.33e-3, i.e. a 0.33% creature error reduction — from a
    // single structural change, under ideal conditions.
    assert!(
        (3.0e-3..4.0e-3).contains(&break_even),
        "break-even raw gain {break_even:e} outside the characterised band"
    );
    assert!(
        break_even > OBSERVED_REALISED_DELTA_CEILING,
        "even a perfect add-neuron candidate needs {break_even:e} raw gain, \
         above the {OBSERVED_REALISED_DELTA_CEILING:e} realised ceiling"
    );
}

#[test]
fn perfect_add_synapse_candidate_needs_over_three_percent_raw_gain() {
    let multiplier = synapse_post_discount_gain(1.0, 100, 100, Some(1.0), 1.0);

    assert!(
        multiplier <= SYNAPSE_PREDICTION_CALIBRATION,
        "best-case synapse multiplier {multiplier:e} must not exceed \
         SYNAPSE_PREDICTION_CALIBRATION {SYNAPSE_PREDICTION_CALIBRATION:e}"
    );
    assert!(
        multiplier >= SYNAPSE_PREDICTION_CALIBRATION * 0.9,
        "best-case synapse multiplier {multiplier:e} should sit within 10% of \
         the calibration cap {SYNAPSE_PREDICTION_CALIBRATION:e}"
    );

    let break_even = break_even_raw_gain(multiplier);
    // 1e-5 / 3e-4 == 3.33e-2 — a 3.3% creature error reduction from one synapse.
    assert!(
        (3.0e-2..4.0e-2).contains(&break_even),
        "break-even raw gain {break_even:e} outside the characterised band"
    );
    assert!(
        break_even > OBSERVED_REALISED_DELTA_CEILING * 10.0,
        "even a perfect add-synapse candidate needs {break_even:e} raw gain"
    );
}

#[test]
fn typical_add_neuron_candidate_break_even_is_two_orders_above_the_realised_band() {
    // A plainly good candidate on a converged network: most samples improve,
    // improvements are a third of the available magnitude, target is not
    // saturated, and the failure cache is neutral.
    let multiplier = neuron_post_discount_gain(1.0, 0.8, 70, 100, Some(0.33), None, 1.0);
    let break_even = break_even_raw_gain(multiplier);

    assert!(
        multiplier < NEURON_PREDICTION_CALIBRATION,
        "typical multiplier {multiplier:e} must be below the best case"
    );
    // Measured: multiplier ~9.8e-4, break-even ~1.0e-2 (a 1% creature error
    // reduction from a single added neuron).
    assert!(
        (5.0e-3..5.0e-2).contains(&break_even),
        "typical break-even raw gain {break_even:e} outside the characterised band"
    );
    assert!(
        break_even > OBSERVED_REALISED_DELTA_CEILING,
        "typical add-neuron break-even {break_even:e} should sit above the \
         {OBSERVED_REALISED_DELTA_CEILING:e} realised ceiling"
    );
}

#[test]
fn saturated_target_break_even_is_another_order_of_magnitude_worse() {
    let unsaturated = neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), None, 1.0);
    let saturated = neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), Some(1.0), 1.0);

    assert!(
        saturated < unsaturated,
        "saturation must discount: {saturated:e} vs {unsaturated:e}"
    );
    // SATURATION_DISCOUNT_AGGRESSIVE == 0.15 at full saturation.
    let ratio = saturated / unsaturated;
    assert!(
        (0.14..0.16).contains(&ratio),
        "saturation ratio {ratio} outside the characterised band"
    );
}

/// The failure-cache calibration correction only ever discounts (clamped to
/// `[0.001, 1.0]`), and it is *fed by realised outcomes of accepted changes*.
/// A creature whose recent history is all over-estimates drives it to the
/// floor, which multiplies the already-small calibration constant by another
/// 1000×. This test measures how far that pushes the break-even gain.
#[test]
fn failure_cache_correction_at_its_floor_makes_the_gain_floor_unreachable() {
    let neutral = neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), None, 1.0);
    let clamped = neuron_post_discount_gain(
        1.0,
        1.0,
        100,
        100,
        Some(1.0),
        None,
        MIN_CALIBRATION_CORRECTION,
    );

    let ratio = neutral / clamped;
    assert!(
        (900.0..1100.0).contains(&ratio),
        "correction floor should shrink the multiplier ~1000x, got {ratio}"
    );

    let break_even = break_even_raw_gain(clamped);
    assert!(
        break_even > 1.0,
        "with the correction clamped to {MIN_CALIBRATION_CORRECTION}, a candidate \
         needs a raw gain of {break_even:e} (>100% creature error reduction) to \
         clear the {MIN_EXPECTED_CREATURE_SCORE_GAIN:e} floor — the floor is \
         unreachable by construction"
    );
}

/// The multiplier is monotonic in every input, so no combination of inputs can
/// recover the lost orders of magnitude: the calibration constant caps it.
#[test]
fn no_input_combination_lifts_the_multiplier_above_the_calibration_constant() {
    let cases = [
        (1.0f32, 100u32, 100u32, Some(1.0f32), None, 1.0f32),
        (1.0, 100, 100, None, None, 1.0),
        (1.0, 99, 100, Some(0.99), None, 1.0),
        (0.5, 100, 100, Some(1.0), Some(0.5), 1.0),
        (1.0, 100, 100, Some(1.0), None, 0.5),
    ];

    for (impact, improved, total, magnitude, saturation, correction) in cases {
        let multiplier =
            neuron_post_discount_gain(1.0, impact, improved, total, magnitude, saturation, correction);
        assert!(
            multiplier <= NEURON_PREDICTION_CALIBRATION * 1.001,
            "multiplier {multiplier:e} exceeded the calibration cap for \
             impact={impact} improved={improved}/{total} correction={correction}"
        );
    }
}
