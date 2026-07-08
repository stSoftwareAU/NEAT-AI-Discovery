//! Propagation-aware change-squash effect at production depth (Issue #1532).
//!
//! Executable specification extending the #1516/#1518 propagation-aware fix from
//! the **remove-neuron** estimate path to the **change-squash** estimate path —
//! the second estimate path cited on GRQ-Discovery commit `2596f073`.
//!
//! Changing a neuron's activation function on a converged network perturbs the
//! neuron's emitted output; that perturbation is diluted/squashed through every
//! intervening weight and activation on the way to the output(s), and — because
//! the downstream layers were trained around the neuron's *original* activation
//! — it typically makes the trained creature slightly *worse*. The recorded
//! pipeline placeholder ignores all of this and fabricates a near-zero gain of
//! `+8.6e-10` for `neuron-1481550544` (`SELU → SQUARE`), whereas the empirically
//! measured effect is `-0.000341` — ~400,000× too small and opposite in sign.
//!
//! Fixtures (committed under `tests/fixtures/change_squash_propagation/` for
//! hermeticity):
//! - `v2_change-squash_neuron-1481550544.json` — the recorded GRQ-Discovery
//!   failure, carrying the placeholder gain, the neuron's local errors under the
//!   current/proposed squash, and the measured actual effect.
//! - the production GRQ-cluster creature topology is shared with the remove-neuron
//!   fixture (`../remove_neuron_propagation/network.json`) — the failure is on the
//!   same creature — so it is not duplicated.
//!
//! This file ships two guards, mirroring `tests/remove_neuron_propagation.rs`:
//! - [`change_squash_placeholder_is_wrong_at_depth`] — the placeholder-guard: the
//!   propagation-aware estimator must never emit the near-zero placeholder range
//!   for this deep candidate. If a fallback branch resurrects the inaccurate
//!   near-zero formula, CI turns red.
//! - [`change_squash_effect_at_production_depth`] — the estimator spec: the
//!   estimate must match the measured actual within one order of magnitude AND in
//!   sign on the committed fixtures (the #1529 pass criterion).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::estimate_change_squash_gain;
use std::path::{Path, PathBuf};

/// The near-zero placeholder gain recorded for the failure example
/// (`expectedCreatureScoreGain`). Topology-blind and activation-blind.
const PLACEHOLDER_GAIN: f64 = 8.602_393_603_479_653e-10;

/// The empirically measured effect of the squash change on the output
/// (`actualErrorReduction`, ~69k samples). Negative: the change made the trained
/// network slightly worse, the opposite of the placeholder's sign.
const MEASURED_ACTUAL: f64 = -0.000_341_394_806_272_044;

/// The target neuron, sitting many layers from the single output.
const TARGET_NEURON: &str = "neuron-1481550544";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/change_squash_propagation")
}

/// Recorded change-squash failure: the fields the estimator consumes plus the
/// measured actual it is graded against.
struct FailureFixture {
    neuron_uuid: String,
    /// The neuron's local error under its current squash (`currentError`).
    current_local_error: f64,
    /// The neuron's local error under the proposed squash (`improvedError`).
    proposed_local_error: f64,
    /// The recorded placeholder `expectedCreatureScoreGain`.
    placeholder_gain: f64,
    /// The measured `actualErrorReduction`.
    measured_actual: f64,
}

/// Load the recorded failure fixture. A missing / corrupt fixture fails here with
/// the fixture path, so hermeticity breakage is caught in the same CI run rather
/// than at runtime in GRQ-cluster.
fn load_failure_fixture() -> FailureFixture {
    let path = fixture_dir().join("v2_change-squash_neuron-1481550544.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read failure fixture {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse failure fixture {}: {e}", path.display()));

    let candidate = &json["rustRequest"]["squashCandidate"];
    FailureFixture {
        neuron_uuid: candidate["neuronUuid"]
            .as_str()
            .expect("fixture missing rustRequest.squashCandidate.neuronUuid")
            .to_string(),
        current_local_error: candidate["currentError"]
            .as_f64()
            .expect("fixture missing rustRequest.squashCandidate.currentError"),
        proposed_local_error: candidate["improvedError"]
            .as_f64()
            .expect("fixture missing rustRequest.squashCandidate.improvedError"),
        placeholder_gain: candidate["expectedCreatureScoreGain"]
            .as_f64()
            .expect("fixture missing rustRequest.squashCandidate.expectedCreatureScoreGain"),
        measured_actual: json["actualErrorReduction"]
            .as_f64()
            .expect("fixture missing actualErrorReduction"),
    }
}

/// Load the production creature topology. Shared with the remove-neuron fixture
/// (the failure is on the same creature), so it is not duplicated here.
fn load_network() -> CreatureJson {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/remove_neuron_propagation/network.json");
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

/// The #1529 accuracy pass criterion: an estimate passes when it matches the
/// measured actual error change within one order of magnitude (10×) AND in sign.
fn meets_pass_criterion(estimate: f64, measured_actual: f64) -> bool {
    within_one_order(estimate, measured_actual) && estimate.signum() == measured_actual.signum()
}

/// Placeholder-guard (runnable in CI). The propagation-aware estimator has
/// replaced the near-zero placeholder, so this asserts the estimator no longer
/// emits a value anywhere near the fabricated `~8.6e-10` placeholder for this
/// deep candidate. If the near-zero placeholder path is ever accidentally
/// reinstated (e.g. a fallback branch resurfaces), this test turns CI red at that
/// commit.
///
/// It still pins the recorded fixture values so drift in the known-bad
/// placeholder / measured-actual constants is caught in the same run.
#[test]
fn change_squash_placeholder_is_wrong_at_depth() {
    let fixture = load_failure_fixture();
    let creature = load_network();

    // The fixture still encodes the known-bad placeholder and measured actual.
    // Drift in either value flips this guard red.
    assert!(
        (fixture.placeholder_gain - PLACEHOLDER_GAIN).abs() < 1e-18,
        "recorded placeholder gain {} drifted from the known-bad value {PLACEHOLDER_GAIN}; \
         re-validate the fixture and spec",
        fixture.placeholder_gain
    );
    assert!(
        (fixture.measured_actual - MEASURED_ACTUAL).abs() < 1e-12,
        "recorded measured actual {} drifted from {MEASURED_ACTUAL}; \
         re-validate the fixture and spec",
        fixture.measured_actual
    );
    assert_eq!(
        fixture.neuron_uuid, TARGET_NEURON,
        "fixture target neuron changed"
    );

    let estimate = estimate_change_squash_gain(
        &creature,
        &fixture.neuron_uuid,
        fixture.current_local_error,
        fixture.proposed_local_error,
    )
    .expect("estimator must return a gain for the target hidden neuron");

    // The honest estimate must be hundreds of thousands of times larger than the
    // near-zero placeholder — nowhere near the fabricated `~8.6e-10`.
    assert!(
        magnitude_ratio(estimate, PLACEHOLDER_GAIN) > 1_000.0,
        "estimate {estimate:e} is within the retired near-zero placeholder range \
         (placeholder {PLACEHOLDER_GAIN:e}); the fabricated near-zero path has resurfaced"
    );
}

/// Estimator spec — the permanent regression gate (#1532).
///
/// The propagation-aware estimator ([`estimate_change_squash_gain`]) replaces the
/// fabricated near-zero placeholder. This case asserts the estimate matches the
/// measured actual within one order of magnitude AND in sign on the committed
/// fixtures. Any future estimator change that breaks the scale or sign fidelity
/// fails `cargo test` in CI before merge.
#[test]
fn change_squash_effect_at_production_depth() {
    let fixture = load_failure_fixture();
    let creature = load_network();

    // Sanity-check we are exercising the intended production-scale creature so the
    // accuracy claim is anchored to the 1,666 / 21,532 GRQ-cluster snapshot.
    assert_eq!(
        creature.neurons.len(),
        1666,
        "fixture creature is not the expected 1,666-neuron production snapshot"
    );
    assert_eq!(
        creature.synapses.len(),
        21_532,
        "fixture creature is not the expected 21,532-synapse production snapshot"
    );

    let estimate = estimate_change_squash_gain(
        &creature,
        &fixture.neuron_uuid,
        fixture.current_local_error,
        fixture.proposed_local_error,
    )
    .expect("estimator must return a gain for the target hidden neuron");

    // The near-zero placeholder FAILS the #1529 pass criterion — wrong sign
    // (fabricated positive vs measured negative) and vastly more than 10× off.
    assert!(
        !meets_pass_criterion(fixture.placeholder_gain, fixture.measured_actual),
        "placeholder {:e} must fail the #1529 pass criterion against the measured \
         actual {:e}",
        fixture.placeholder_gain,
        fixture.measured_actual
    );

    // The placeholder fails on BOTH counts: wrong sign (fabricated positive) and
    // vastly off in magnitude (ratio ~2.5e-6, far below the 0.1 floor).
    assert_ne!(
        fixture.placeholder_gain.signum(),
        fixture.measured_actual.signum(),
        "placeholder {:e} should have the wrong sign vs the measured actual {:e}",
        fixture.placeholder_gain,
        fixture.measured_actual
    );
    assert!(
        !within_one_order(fixture.placeholder_gain, fixture.measured_actual),
        "placeholder {:e} should be far more than 10× off the measured actual {:e}",
        fixture.placeholder_gain,
        fixture.measured_actual
    );

    // The propagation-aware estimator PASSES the #1529 criterion — correct sign,
    // within one order of magnitude.
    assert!(
        meets_pass_criterion(estimate, fixture.measured_actual),
        "propagation-aware estimate {estimate:e} must pass the #1529 pass criterion \
         against the measured actual {:e}",
        fixture.measured_actual
    );
}

/// The estimator returns `None` for an output neuron (output activations are
/// governed by the loss contract, not change-squash candidates) and for a neuron
/// absent from the topology.
#[test]
fn change_squash_gain_is_none_for_non_candidates() {
    let creature = load_network();
    assert!(
        estimate_change_squash_gain(&creature, "neuron-does-not-exist", 1.0, 0.5).is_none(),
        "absent neuron must not be a change-squash candidate"
    );
    let output_uuid = creature
        .neurons
        .iter()
        .find(|n| n.neuron_type == "output")
        .map(|n| n.uuid.clone())
        .expect("production creature must have at least one output neuron");
    assert!(
        estimate_change_squash_gain(&creature, &output_uuid, 1.0, 0.5).is_none(),
        "output neuron must not be a change-squash candidate"
    );
}

/// A swap that does not reduce the neuron's local error induces no perturbation,
/// so the honest gain is zero (never a fabricated win). Guards the `max(0.0)`
/// floor on the local perturbation.
#[test]
fn change_squash_non_improving_swap_yields_zero_gain() {
    let creature = load_network();
    // proposed_local_error >= current_local_error → no local improvement.
    let gain = estimate_change_squash_gain(&creature, TARGET_NEURON, 100.0, 100.0)
        .expect("estimator must return a gain for the target hidden neuron");
    assert_eq!(
        gain, 0.0,
        "a non-improving swap must yield exactly zero gain"
    );

    let worse = estimate_change_squash_gain(&creature, TARGET_NEURON, 100.0, 250.0)
        .expect("estimator must return a gain for the target hidden neuron");
    assert_eq!(
        worse, 0.0,
        "a swap that increases local error must not fabricate a non-zero gain"
    );
}

/// The honest gain is always non-positive: a change-squash on a converged network
/// disrupts the downstream layers trained around the original activation.
#[test]
fn change_squash_gain_is_non_positive() {
    let fixture = load_failure_fixture();
    let creature = load_network();
    let estimate = estimate_change_squash_gain(
        &creature,
        &fixture.neuron_uuid,
        fixture.current_local_error,
        fixture.proposed_local_error,
    )
    .expect("estimator must return a gain for the target hidden neuron");
    assert!(
        estimate <= 0.0,
        "propagation-aware change-squash gain {estimate:e} must be non-positive"
    );
}
