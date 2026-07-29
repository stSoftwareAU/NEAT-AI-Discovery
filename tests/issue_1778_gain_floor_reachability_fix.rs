//! Issue #1778 — the expected-gain floor is reachable against the
//! post-calibration scale.
//!
//! The #1777 characterisation suite measured the add-neuron / add-synapse
//! discount stack against the acceptance floor and found the floor unreachable
//! by construction: every factor in the stack is `<= 1.0`, so the fixed
//! calibration constant (`0.003` neuron, `0.0003` synapse) caps the achievable
//! multiplier, and a raw `1e-5` floor compared against that capped value
//! demanded a raw creature error reduction of 0.35 %–3.5 % from a *single*
//! structural change — an order of magnitude above the entire realised band
//! observed in production (#1737).
//!
//! The fix settles the scale question the issue poses. `expectedCreatureScoreGain`
//! at the filter site is denominated in the **calibrated (realised)** scale;
//! `MIN_EXPECTED_CREATURE_SCORE_GAIN` is denominated in the **pre-calibration
//! (prediction)** scale, where the round-off argument that motivated it (#1191)
//! actually holds. The noise screen therefore belongs *upstream* of the
//! calibration constant, and the filter converts it into the calibrated scale
//! via `calibrated_gain_floor` rather than comparing across scales.
//!
//! These tests assert the fix from both directions:
//! - the floor is now **reachable** — break-even raw gains sit below the
//!   realised band, for every case #1777 pinned; and
//! - the floor still **rejects noise** — the exact production estimate that
//!   realised a harmful `-8.65e-4` is still dropped, which is the #1740
//!   false-positive guard the issue warns must not be traded away.

use neat_ai_discovery::CandidateSynapseJson;
use neat_ai_discovery::analysis::constants::{
    GAIN_FLOOR_NOISE_BACKSTOP, MIN_EXPECTED_CREATURE_SCORE_GAIN, NEURON_PREDICTION_CALIBRATION,
    SYNAPSE_PREDICTION_CALIBRATION, calibrated_gain_floor, min_expected_gain_floor_for_neurons,
    min_expected_gain_floor_for_synapses,
};
use neat_ai_discovery::analysis::scoring::calibration_correction::MIN_CALIBRATION_CORRECTION;
use neat_ai_discovery::analysis::synapse::post_processing::apply_min_expected_gain_floor_for_synapses;
use neat_ai_discovery::analysis::synapse::scoring::{
    apply_logistic_prediction_calibration, apply_neuron_pessimism_discount,
    apply_saturation_prediction_discount, apply_synapse_pessimism_discount,
};

/// Largest realised score delta observed in the production candidate cache
/// (#1737): accepted changes land around `1e-7`, rejected ones up to `~1e-3`.
/// A break-even raw gain above this band is unreachable in practice.
const OBSERVED_REALISED_DELTA_CEILING: f32 = 1e-3;

/// Smallest realised delta of an *accepted* production change (#1740): the
/// harmful-neuron removal that realised `+1.95e-7`. The floor must sit below
/// this or it rejects the achievable band.
const SMALLEST_REALISED_ACCEPTED_DELTA: f32 = 1.95e-7;

/// The production coordinated `change-squash` that estimated `4.17e-10` and
/// realised `-8.65e-4`. The #1740 false-positive guard requires this to stay
/// rejected.
const PRODUCTION_NOISE_ESTIMATE: f32 = 4.17e-10;

/// Run the shipped add-neuron discount stack over a raw creature error
/// reduction, in the order `apply_creature_level_metrics` applies it.
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
    let gain =
        apply_synapse_pessimism_discount(raw_error_reduction, improved, total, magnitude_ratio);
    apply_logistic_prediction_calibration(
        gain,
        improved,
        total,
        SYNAPSE_PREDICTION_CALIBRATION * correction,
    )
}

/// Raw creature error reduction needed for a post-discount gain to reach the
/// effective floor, given a multiplier measured from a unit raw gain.
fn break_even_raw_gain(effective_floor: f32, unit_multiplier: f32) -> f32 {
    effective_floor / unit_multiplier
}

fn synapse_candidate(gain: f32) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: format!("source-{gain}"),
        to_neuron_uuid: format!("target-{gain}"),
        from_neuron_index: None,
        to_neuron_index: None,
        weight: 1.0,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 10,
        total_count: 10,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.5,
        expected_score_gain_confidence_interval: [gain, gain],
        comment: None,
        variant_key: None,
    }
}

#[test]
fn effective_floors_are_the_screen_converted_into_the_calibrated_scale() {
    let neuron_floor = min_expected_gain_floor_for_neurons();
    let synapse_floor = min_expected_gain_floor_for_synapses();

    let expected_neuron = MIN_EXPECTED_CREATURE_SCORE_GAIN * NEURON_PREDICTION_CALIBRATION;
    let expected_synapse = MIN_EXPECTED_CREATURE_SCORE_GAIN * SYNAPSE_PREDICTION_CALIBRATION;

    assert!(
        (neuron_floor - expected_neuron).abs() <= expected_neuron * 1e-3,
        "neuron floor {neuron_floor:e} should be the screen converted by the \
         neuron calibration constant ({expected_neuron:e})"
    );
    assert!(
        (synapse_floor - expected_synapse).abs() <= expected_synapse * 1e-3,
        "synapse floor {synapse_floor:e} should be the screen converted by the \
         synapse calibration constant ({expected_synapse:e})"
    );
}

#[test]
fn effective_floors_sit_below_the_smallest_realised_accepted_delta() {
    // The whole point of the fix: the screen must not sit above the band the
    // network can actually realise.
    for (label, floor) in [
        ("neuron", min_expected_gain_floor_for_neurons()),
        ("synapse", min_expected_gain_floor_for_synapses()),
    ] {
        assert!(
            floor < SMALLEST_REALISED_ACCEPTED_DELTA,
            "{label} floor {floor:e} must sit below the smallest realised \
             accepted delta {SMALLEST_REALISED_ACCEPTED_DELTA:e}"
        );
    }
}

#[test]
fn perfect_add_neuron_break_even_is_below_the_realised_band() {
    let multiplier = neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), None, 1.0);
    let break_even = break_even_raw_gain(min_expected_gain_floor_for_neurons(), multiplier);

    // Was 3.5e-3 (0.35 %) before the fix; now ~1.0e-5 (0.001 %).
    assert!(
        break_even < OBSERVED_REALISED_DELTA_CEILING,
        "perfect add-neuron break-even {break_even:e} must sit below the \
         {OBSERVED_REALISED_DELTA_CEILING:e} realised ceiling"
    );
    assert!(
        (5.0e-6..5.0e-5).contains(&break_even),
        "perfect add-neuron break-even {break_even:e} outside the characterised band"
    );
}

#[test]
fn typical_add_neuron_break_even_is_below_the_realised_band() {
    // A plainly good candidate on a converged network: most samples improve,
    // improvements are a third of the available magnitude, target unsaturated.
    let multiplier = neuron_post_discount_gain(1.0, 0.8, 70, 100, Some(0.33), None, 1.0);
    let break_even = break_even_raw_gain(min_expected_gain_floor_for_neurons(), multiplier);

    // Was ~1.0e-2 (1 %) before the fix.
    assert!(
        break_even < OBSERVED_REALISED_DELTA_CEILING,
        "typical add-neuron break-even {break_even:e} must sit below the \
         {OBSERVED_REALISED_DELTA_CEILING:e} realised ceiling"
    );
}

#[test]
fn perfect_add_synapse_break_even_is_below_the_realised_band() {
    let multiplier = synapse_post_discount_gain(1.0, 100, 100, Some(1.0), 1.0);
    let break_even = break_even_raw_gain(min_expected_gain_floor_for_synapses(), multiplier);

    // Was 3.5e-2 (3.5 %) before the fix.
    assert!(
        break_even < OBSERVED_REALISED_DELTA_CEILING,
        "perfect add-synapse break-even {break_even:e} must sit below the \
         {OBSERVED_REALISED_DELTA_CEILING:e} realised ceiling"
    );
}

#[test]
fn neuron_and_synapse_floors_impose_the_same_raw_selectivity() {
    // The 10× gap between the two calibration constants is a property of
    // candidate *type*, not of candidate merit. Dividing it out of the floor
    // removes the unintended type bias the issue's break-even table exposed
    // (0.35 % for neurons vs 3.5 % for synapses under identical conditions).
    let neuron_break_even = break_even_raw_gain(
        min_expected_gain_floor_for_neurons(),
        neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), None, 1.0),
    );
    let synapse_break_even = break_even_raw_gain(
        min_expected_gain_floor_for_synapses(),
        synapse_post_discount_gain(1.0, 100, 100, Some(1.0), 1.0),
    );

    let ratio = synapse_break_even / neuron_break_even;
    assert!(
        (0.8..1.25).contains(&ratio),
        "add-neuron and add-synapse should demand comparable raw gains, got \
         {neuron_break_even:e} vs {synapse_break_even:e} (ratio {ratio})"
    );
}

#[test]
fn calibration_correction_at_its_floor_leaves_the_gain_floor_reachable() {
    // The per-creature correction stays on the candidate side by design — it is
    // evidence that this creature's predictions over-shoot, so it should still
    // tighten acceptance. What it must no longer do is push the requirement
    // past a 100 % creature error reduction, which is unreachable at any
    // network size or convergence state.
    let clamped = neuron_post_discount_gain(
        1.0,
        1.0,
        100,
        100,
        Some(1.0),
        None,
        MIN_CALIBRATION_CORRECTION,
    );
    let break_even = break_even_raw_gain(min_expected_gain_floor_for_neurons(), clamped);

    assert!(
        break_even < 1.0,
        "with the correction clamped to {MIN_CALIBRATION_CORRECTION}, break-even \
         raw gain {break_even:e} must stay below a 100 % creature error reduction"
    );
    // Still meaningfully stricter than the neutral case — the correction has
    // not been cancelled out.
    let neutral = neuron_post_discount_gain(1.0, 1.0, 100, 100, Some(1.0), None, 1.0);
    let neutral_break_even = break_even_raw_gain(min_expected_gain_floor_for_neurons(), neutral);
    assert!(
        break_even > neutral_break_even * 100.0,
        "the correction must still tighten acceptance ({break_even:e} vs \
         {neutral_break_even:e})"
    );
}

#[test]
fn production_noise_estimate_is_still_rejected_by_the_synapse_floor() {
    // #1740's false-positive guard: the 4.17e-10 estimate that realised
    // -8.65e-4 must not become acceptable as a side effect of the fix.
    let mut candidates = vec![
        synapse_candidate(PRODUCTION_NOISE_ESTIMATE),
        synapse_candidate(SMALLEST_REALISED_ACCEPTED_DELTA),
    ];
    let dropped = apply_min_expected_gain_floor_for_synapses(&mut candidates);

    assert_eq!(dropped, 1, "the noise-level estimate must still be dropped");
    assert_eq!(candidates.len(), 1);
    assert!(
        (candidates[0].expected_creature_score_gain - SMALLEST_REALISED_ACCEPTED_DELTA).abs()
            < f32::EPSILON,
        "the achievable-band candidate must survive"
    );
}

#[test]
fn floor_never_opens_below_the_noise_backstop() {
    // A very small calibration constant must not shrink the screen into the
    // band where the estimate is uncorrelated with the outcome.
    let floor = calibrated_gain_floor(1e-12);
    assert!(
        (floor - GAIN_FLOOR_NOISE_BACKSTOP).abs() < f32::EPSILON * 10.0,
        "floor {floor:e} should clamp to the backstop {GAIN_FLOOR_NOISE_BACKSTOP:e}"
    );
    const {
        assert!(
            GAIN_FLOOR_NOISE_BACKSTOP > PRODUCTION_NOISE_ESTIMATE,
            "the backstop must stay above the production noise estimate"
        );
    }
}

#[test]
fn non_conversion_calibration_falls_back_to_the_unconverted_screen() {
    // A value outside (0.0, 1.0] is not a units conversion; widening the screen
    // on it would be a silent failure. Fail safe to the unconverted screen.
    for bad in [0.0_f32, -1.0, 2.0, f32::NAN, f32::INFINITY] {
        let floor = calibrated_gain_floor(bad);
        assert!(
            (floor - MIN_EXPECTED_CREATURE_SCORE_GAIN).abs() < f32::EPSILON,
            "calibration {bad} should fall back to the unconverted screen, got {floor:e}"
        );
    }
}
