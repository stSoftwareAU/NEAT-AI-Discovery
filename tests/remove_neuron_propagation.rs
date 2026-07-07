//! Propagation-aware remove-neuron effect at production depth (Issue #1517).
//!
//! Executable specification for the #1516 root-cause fix. Removing a neuron that
//! sits many layers from the output(s) has an effect that is diluted / squashed
//! through every intervening weight and activation function on the way to the
//! output. The current NEAT-AI (Deno) placeholder ignores this propagation and
//! fabricates a gain of `+0.17882921` for the recorded failure example, whereas
//! the empirically measured effect is only `~-0.000194` — ~920× too large and
//! opposite in sign.
//!
//! Fixtures (committed under `tests/fixtures/remove_neuron_propagation/` for
//! hermeticity):
//! - `network.json` — the production GRQ-cluster creature (topology), in which
//!   the target neuron `neuron-1802938338` is many steps from the single output.
//! - `v2_remove-neuron_neuron-1802938338.json` — the recorded GRQ-Discovery
//!   failure, carrying the placeholder gain and the measured actual effect.
//!
//! This file ships two tests:
//! - [`placeholder_gain_is_wrong_at_depth`] (runnable) — since #1518 landed the
//!   propagation-aware estimator, this is inverted into a regression guard: it
//!   asserts the estimator no longer emits the floor-clamped `[0.1, 0.5]`
//!   placeholder range for this deep neuron. If the placeholder path is ever
//!   accidentally reinstated (e.g. a fallback branch resurfaces), it fails.
//! - [`remove_neuron_effect_at_production_depth`] — the estimator spec. Now that
//!   the #1516/#1518 propagation-aware estimator lands, the `#[ignore]` is
//!   removed and this is the permanent regression gate: the estimate must match
//!   the measured actual within one order of magnitude and in sign.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::estimate_remove_neuron_gain;
use neat_ai_discovery::focus::compute_impacts_public;
use std::path::{Path, PathBuf};

/// The fabricated placeholder gain recorded for the failure example
/// (`expectedCreatureScoreGain`). Reproduced exactly by the NEAT-AI #2483
/// over-threshold sink `0.1 + (log10(err) − 10)/10 × 0.4`.
const PLACEHOLDER_GAIN: f64 = 0.178_829_211_850_295_82;

/// The empirically measured effect of the removal on the output
/// (`actualErrorReduction`, ~69k samples). Negative: the removal made the
/// network slightly worse, the opposite of the placeholder's sign.
const MEASURED_ACTUAL: f64 = -0.000_194_478_429_877_298_35;

/// The target neuron, sitting many layers from the single output.
const TARGET_NEURON: &str = "neuron-1802938338";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remove_neuron_propagation")
}

/// Load the recorded failure fixture and return
/// `(expected_creature_score_gain, actual_error_reduction, neuron_uuid)`.
///
/// A missing / corrupt fixture fails here with the fixture path, so hermeticity
/// breakage is caught in the same CI run rather than at runtime in GRQ-cluster.
fn load_failure_fixture() -> (f64, f64, String) {
    let path = fixture_dir().join("v2_remove-neuron_neuron-1802938338.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read failure fixture {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse failure fixture {}: {e}", path.display()));

    let candidate = &json["rustRequest"]["harmfulNeuronCandidate"];
    let gain = candidate["expectedCreatureScoreGain"]
        .as_f64()
        .expect("fixture missing rustRequest.harmfulNeuronCandidate.expectedCreatureScoreGain");
    let actual = json["actualErrorReduction"]
        .as_f64()
        .expect("fixture missing actualErrorReduction");
    let uuid = candidate["neuronUuid"]
        .as_str()
        .expect("fixture missing rustRequest.harmfulNeuronCandidate.neuronUuid")
        .to_string();
    (gain, actual, uuid)
}

/// Load the production creature topology from the committed fixture.
fn load_network() -> CreatureJson {
    let path = fixture_dir().join("network.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read network fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse network fixture {}: {e}", path.display()))
}

/// Ratio of two magnitudes, guarding against a zero denominator.
fn magnitude_ratio(a: f64, b: f64) -> f64 {
    let denom = b.abs();
    assert!(denom > 0.0, "magnitude_ratio: zero denominator");
    a.abs() / denom
}

/// True when `a` and `b` share magnitude to within one order of magnitude (10×).
fn within_one_order(a: f64, b: f64) -> bool {
    (0.1..=10.0).contains(&magnitude_ratio(a, b))
}

/// Propagation-aware structural impact of the target neuron on the output(s):
/// the squash-aware product of downstream weights and saturation bounds all the
/// way to the output, normalised to a fraction of influence in `[0, 1]`. Deep
/// neurons attenuate to a tiny value.
fn propagation_aware_impact(creature: &CreatureJson, neuron_uuid: &str) -> f64 {
    let impacts = compute_impacts_public(creature);
    let impact = impacts
        .get(neuron_uuid)
        .copied()
        .unwrap_or_else(|| panic!("target neuron {neuron_uuid} not present in impact map"));
    f64::from(impact)
}

/// Regression guard (runnable in CI). Inverted by #1518: the propagation-aware
/// estimator has replaced the placeholder, so this now asserts that the
/// estimator no longer emits the floor-clamped `[0.1, 0.5]` placeholder range
/// for this deep neuron. If the placeholder path is ever accidentally
/// reinstated (e.g. a fallback branch resurfaces and clamps the gain back up),
/// this test turns CI red at that commit.
///
/// It still pins the recorded fixture values so drift in the known-bad
/// placeholder / measured-actual constants is caught in the same run.
#[test]
fn placeholder_gain_is_wrong_at_depth() {
    let (recorded_gain, recorded_actual, uuid) = load_failure_fixture();
    let creature = load_network();

    // The fixture still encodes the known-bad placeholder and measured actual.
    // Drift in either value flips this guard red.
    assert!(
        (recorded_gain - PLACEHOLDER_GAIN).abs() < 1e-9,
        "recorded placeholder gain {recorded_gain} drifted from the known-bad value {PLACEHOLDER_GAIN}; \
         re-validate the fixture and spec"
    );
    assert!(
        (recorded_actual - MEASURED_ACTUAL).abs() < 1e-9,
        "recorded measured actual {recorded_actual} drifted from {MEASURED_ACTUAL}; \
         re-validate the fixture and spec"
    );
    assert_eq!(uuid, TARGET_NEURON, "fixture target neuron changed");

    // Inverted guard: the estimator must NOT emit the floor-clamped
    // `[0.1, 0.5]` placeholder range for this deep neuron. The honest gain is a
    // tiny (~1e-4) value, far below the fabricated floor of 0.1.
    let estimate = estimate_remove_neuron_gain(&creature, &uuid)
        .expect("estimator must return a gain for the target hidden neuron");
    assert!(
        !(0.1..=0.5).contains(&estimate.abs()),
        "estimator emitted a value {estimate} inside the retired placeholder floor range \
         [0.1, 0.5]; the fabricated floor-clamped placeholder path has resurfaced"
    );
}

/// Estimator spec — the permanent regression gate (#1516/#1518).
///
/// The propagation-aware estimator ([`estimate_remove_neuron_gain`]) replaces
/// the fabricated placeholder. This case asserts the estimate matches the
/// measured actual within one order of magnitude AND in sign on the committed
/// fixtures. Any future estimator change that breaks the scale or sign fidelity
/// fails `cargo test` in CI before merge.
///
/// The propagation reference (computed live from the committed topology)
/// corroborates that the true effect at production depth really is tiny —
/// within an order of magnitude of the measured actual and thousands of times
/// below the retired placeholder.
#[test]
fn remove_neuron_effect_at_production_depth() {
    let (placeholder_gain, measured_actual, uuid) = load_failure_fixture();
    let creature = load_network();

    // Reference computation: propagate the neuron's contribution through all
    // downstream weights and squash bounds to the output(s).
    let reference = propagation_aware_impact(&creature, &uuid);

    // The propagation-aware reference is within an order of magnitude of the
    // measured actual — the correct scale for a neuron this deep.
    assert!(
        within_one_order(reference, measured_actual),
        "propagation reference {reference:e} should be within one order of magnitude of \
         the measured actual {measured_actual:e}"
    );

    // ...and thousands of times below the fabricated placeholder.
    assert!(
        magnitude_ratio(placeholder_gain, reference) > 100.0,
        "placeholder {placeholder_gain} should dwarf the propagation reference {reference:e}"
    );

    // The executable specification: the propagation-aware estimator matches the
    // measured actual within one order of magnitude AND in sign.
    let estimate = estimate_remove_neuron_gain(&creature, &uuid)
        .expect("estimator must return a gain for the target hidden neuron");
    assert!(
        within_one_order(estimate, measured_actual)
            && estimate.signum() == measured_actual.signum(),
        "estimate {estimate:e} must match the measured actual {measured_actual:e} \
         within one order of magnitude and in sign"
    );
}
