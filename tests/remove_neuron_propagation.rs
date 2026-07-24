//! Propagation-aware remove-neuron effect at depth (Issue #1517, re-based by
//! Issue #1722).
//!
//! Executable specification for the #1516 root-cause fix. Removing a neuron that
//! sits many layers from the output(s) has an effect that is diluted / squashed
//! through every intervening weight and activation function on the way to the
//! output. The retired NEAT-AI (Deno) placeholder ignored this propagation and
//! fabricated a large positive gain from the neuron's squash error alone,
//! whereas the true propagated effect at depth is tiny and negative.
//!
//! Fixtures (committed under `tests/fixtures/remove_neuron_propagation/`, all
//! hand-authored and synthetic so this public repository stays self-contained —
//! Issue #1722):
//! - `network.json` — a 13-hop deep-chain creature. The target neuron `spine-0`
//!   reaches the single output through 13 halving hops, so its propagation-aware
//!   influence is exactly `0.5^13 = 1.220703125e-4`.
//! - `v2_remove-neuron_spine-0.json` — a remove-neuron candidate record carrying
//!   the retired placeholder gain and the closed-form propagated effect.
//!
//! Both reference values are derivable by hand — the placeholder from the public
//! `#2483` floor formula, the propagated effect from the halving-hop count — so
//! the fixtures are a genuine oracle rather than a recording of whatever the
//! implementation happened to emit.
//!
//! This file ships three tests:
//! - [`propagation_estimate_beats_placeholder_at_depth`] (Issue #1531) — the
//!   explicit before/after accuracy comparison. It computes both the retired
//!   NEAT-AI #2483 placeholder floor formula and the propagation-aware
//!   [`estimate_remove_neuron_gain`] against the analytic reference effect, then
//!   asserts the placeholder *fails* the #1529 pass criterion (wrong sign / >10×
//!   off) while the new estimator *passes* it and is measurably closer.
//! - [`placeholder_gain_is_wrong_at_depth`] — since #1518 landed the
//!   propagation-aware estimator, this is a regression guard: it asserts the
//!   estimator no longer emits the floor-clamped `[0.1, 0.5]` placeholder range
//!   for this deep neuron. If the placeholder path is ever accidentally
//!   reinstated (e.g. a fallback branch resurfaces), it fails.
//! - [`remove_neuron_effect_at_depth`] — the estimator spec and permanent
//!   regression gate: the estimate must equal the analytic reference effect.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::estimate_remove_neuron_gain;
use neat_ai_discovery::focus::compute_impacts_public;
use std::path::{Path, PathBuf};

/// The placeholder gain carried by the candidate record
/// (`expectedCreatureScoreGain`). Reproduced exactly by the NEAT-AI #2483
/// over-threshold sink `0.1 + (log10(err) − 10)/10 × 0.4` at `err = 1e12`.
const PLACEHOLDER_GAIN: f64 = 0.18;

/// The closed-form propagated effect of removing the target neuron
/// (`analyticErrorReduction`): the negation of its exact `0.5^13` share of the
/// output's weight budget. Negative — the removal costs the network that
/// contribution, the opposite of the placeholder's sign.
const REFERENCE_EFFECT: f64 = -0.000_122_070_312_5;

/// The target neuron, 13 halving hops from the single output.
const TARGET_NEURON: &str = "spine-0";

/// The synthetic snapshot's dimensions. If the committed topology ever
/// deserialises to different dimensions, the analytic references no longer hold,
/// so every test fails fast rather than grading against a changed shape.
const EXPECTED_NEURONS: usize = 27;
const EXPECTED_SYNAPSES: usize = 40;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remove_neuron_propagation")
}

fn read_candidate_fixture() -> serde_json::Value {
    let path = fixture_dir().join("v2_remove-neuron_spine-0.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read candidate fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse candidate fixture {}: {e}", path.display()))
}

/// Load the candidate record and return
/// `(expected_creature_score_gain, analytic_error_reduction, neuron_uuid)`.
///
/// A missing / corrupt fixture fails here with the fixture path, so hermeticity
/// breakage is caught in the same CI run rather than downstream.
fn load_candidate_fixture() -> (f64, f64, String) {
    let json = read_candidate_fixture();
    let candidate = &json["rustRequest"]["harmfulNeuronCandidate"];
    let gain = candidate["expectedCreatureScoreGain"]
        .as_f64()
        .expect("fixture missing rustRequest.harmfulNeuronCandidate.expectedCreatureScoreGain");
    let reference = json["analyticErrorReduction"]
        .as_f64()
        .expect("fixture missing analyticErrorReduction");
    let uuid = candidate["neuronUuid"]
        .as_str()
        .expect("fixture missing rustRequest.harmfulNeuronCandidate.neuronUuid")
        .to_string();
    (gain, reference, uuid)
}

/// Load the squash-error magnitude (`errorMagnitude`) that the retired #2483
/// placeholder formula consumed. This is the sole input the placeholder ever
/// looked at — it is topology-blind — so reproducing the placeholder gain needs
/// only this value.
fn load_error_magnitude() -> f64 {
    read_candidate_fixture()["rustRequest"]["harmfulNeuronCandidate"]["errorMagnitude"]
        .as_f64()
        .expect("fixture missing rustRequest.harmfulNeuronCandidate.errorMagnitude")
}

/// The retired NEAT-AI #2483 over-threshold placeholder floor formula:
/// `0.1 + (log10(err) − 10)/10 × 0.4`, clamped to the synthetic `[0.1, 0.5]`
/// floor. It is topology-blind — a deep neuron and a shallow one with the same
/// squash error get the same fabricated positive "gain" — which is exactly the
/// defect #1516/#1518 removed. Reproduced here so the before/after test can show
/// what the pipeline used to emit.
fn placeholder_floor_gain(error_magnitude: f64) -> f64 {
    (0.1 + (error_magnitude.log10() - 10.0) / 10.0 * 0.4).clamp(0.1, 0.5)
}

/// The #1529 accuracy pass criterion: an estimate passes when it matches the
/// reference error change within one order of magnitude (10×) **and** in sign.
/// Both the placeholder and the estimator are graded against this single
/// criterion so the before/after comparison is apples-to-apples.
fn meets_pass_criterion(estimate: f64, reference: f64) -> bool {
    within_one_order(estimate, reference) && estimate.signum() == reference.signum()
}

/// Load the deep-chain creature topology from the committed fixture.
fn load_network() -> CreatureJson {
    let path = fixture_dir().join("network.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read network fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse network fixture {}: {e}", path.display()))
}

/// Load the topology and assert it is the shape the analytic references were
/// derived from. A changed shape invalidates every reference value here, so it
/// must fail loudly rather than silently grade against new arithmetic.
fn load_expected_network() -> CreatureJson {
    let creature = load_network();
    assert_eq!(
        creature.neurons.len(),
        EXPECTED_NEURONS,
        "fixture creature is not the expected {EXPECTED_NEURONS}-neuron deep-chain snapshot"
    );
    assert_eq!(
        creature.synapses.len(),
        EXPECTED_SYNAPSES,
        "fixture creature is not the expected {EXPECTED_SYNAPSES}-synapse deep-chain snapshot"
    );
    creature
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

/// Regression guard. Inverted by #1518: the propagation-aware estimator has
/// replaced the placeholder, so this asserts the estimator no longer emits the
/// floor-clamped `[0.1, 0.5]` placeholder range for this deep neuron. If the
/// placeholder path is ever accidentally reinstated (e.g. a fallback branch
/// resurfaces and clamps the gain back up), this test turns CI red at that
/// commit.
///
/// It still pins the committed fixture values so drift in the known-bad
/// placeholder / analytic reference constants is caught in the same run.
#[test]
fn placeholder_gain_is_wrong_at_depth() {
    let (recorded_gain, recorded_reference, uuid) = load_candidate_fixture();
    let creature = load_expected_network();

    // The fixture still encodes the known-bad placeholder and analytic reference.
    // Drift in either value flips this guard red.
    assert!(
        (recorded_gain - PLACEHOLDER_GAIN).abs() < 1e-9,
        "fixture placeholder gain {recorded_gain} drifted from the known-bad value \
         {PLACEHOLDER_GAIN}; re-validate the fixture and spec"
    );
    assert!(
        (recorded_reference - REFERENCE_EFFECT).abs() < 1e-12,
        "fixture analytic reference {recorded_reference} drifted from {REFERENCE_EFFECT}; \
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
/// the fabricated placeholder. Because the committed topology attenuates by
/// exactly one half per hop, the honest gain is derivable by hand: `−0.5^13`.
/// This case asserts the estimator emits precisely that. Any future estimator
/// change that breaks the scale or sign fidelity fails `cargo test` in CI before
/// merge.
#[test]
fn remove_neuron_effect_at_depth() {
    let (placeholder_gain, reference, uuid) = load_candidate_fixture();
    let creature = load_expected_network();

    // Reference computation: propagate the neuron's contribution through all
    // downstream weights and squash bounds to the output(s). Thirteen halving
    // hops, all weights 1 and all squashes IDENTITY, so the arithmetic is exact.
    let impact = propagation_aware_impact(&creature, &uuid);
    assert!(
        (impact - reference.abs()).abs() < 1e-12,
        "propagation impact {impact:e} must equal the analytic influence {:e} \
         (0.5^13 over the committed deep chain)",
        reference.abs()
    );

    // ...and the placeholder dwarfs it by three orders of magnitude.
    assert!(
        magnitude_ratio(placeholder_gain, impact) > 100.0,
        "placeholder {placeholder_gain} should dwarf the propagation reference {impact:e}"
    );

    // The executable specification: the propagation-aware estimator emits the
    // analytic reference effect exactly — correct magnitude and correct sign.
    let estimate = estimate_remove_neuron_gain(&creature, &uuid)
        .expect("estimator must return a gain for the target hidden neuron");
    assert!(
        (estimate - reference).abs() < 1e-12,
        "estimate {estimate:e} must equal the analytic reference effect {reference:e}"
    );
    assert!(
        meets_pass_criterion(estimate, reference),
        "estimate {estimate:e} must pass the #1529 pass criterion against the analytic \
         reference {reference:e}"
    );
}

/// Before/after accuracy proof (Issue #1531).
///
/// The explicit user ask: don't just show the new estimator passes — prove it
/// yields a **more accurate** estimate than the retired #2483 placeholder at
/// depth. Against the committed deep-chain fixture this computes:
///
/// - **before** — the retired placeholder floor formula on the recorded
///   `errorMagnitude` (reproducing the fabricated `+0.18` the pipeline emitted),
///   and
/// - **after** — the propagation-aware [`estimate_remove_neuron_gain`],
///
/// then grades both against the analytic reference effect using the single #1529
/// pass criterion. The placeholder must **fail** it (wrong sign and >10× off)
/// while the new estimator **passes** it, and the estimator's absolute error must
/// be strictly smaller — documenting the regression the fix removed.
///
/// If a future change lets the estimator drift outside the pass criterion, or
/// the placeholder branch unexpectedly starts passing (the comparison has
/// degenerated), this turns CI red before merge.
#[test]
fn propagation_estimate_beats_placeholder_at_depth() {
    let (recorded_gain, reference, uuid) = load_candidate_fixture();
    let creature = load_expected_network();
    let error_magnitude = load_error_magnitude();

    // BEFORE: the retired #2483 placeholder floor formula reproduces the
    // fabricated `+0.18` gain from the topology-blind `errorMagnitude` alone.
    let placeholder = placeholder_floor_gain(error_magnitude);
    assert!(
        (placeholder - PLACEHOLDER_GAIN).abs() < 1e-9,
        "placeholder formula {placeholder} should reproduce the known-bad fabricated \
         gain {PLACEHOLDER_GAIN}"
    );
    assert!(
        (placeholder - recorded_gain).abs() < 1e-9,
        "placeholder formula {placeholder} should reproduce the fixture's recorded \
         expectedCreatureScoreGain {recorded_gain}"
    );

    // AFTER: the propagation-aware estimator.
    let estimate = estimate_remove_neuron_gain(&creature, &uuid)
        .expect("estimator must return a gain for the target hidden neuron");

    // The placeholder FAILS the #1529 pass criterion — wrong sign (fabricated
    // positive vs propagated negative) and far more than 10× off in magnitude.
    assert!(
        !meets_pass_criterion(placeholder, reference),
        "placeholder {placeholder} must fail the #1529 pass criterion against the \
         analytic reference {reference:e}"
    );
    assert_ne!(
        placeholder.signum(),
        reference.signum(),
        "placeholder {placeholder} should have the wrong sign vs the analytic reference \
         {reference:e}"
    );
    assert!(
        magnitude_ratio(placeholder, reference) > 10.0,
        "placeholder {placeholder} should be >10× off the analytic reference {reference:e}"
    );

    // The new estimator PASSES the same criterion — correct sign, within 10×.
    assert!(
        meets_pass_criterion(estimate, reference),
        "propagation-aware estimate {estimate:e} must pass the #1529 pass criterion \
         against the analytic reference {reference:e}"
    );

    // The headline before/after claim: the propagation-aware estimate is
    // measurably closer to the reference effect than the placeholder ever was.
    let placeholder_error = (placeholder - reference).abs();
    let estimate_error = (estimate - reference).abs();
    assert!(
        estimate_error < placeholder_error,
        "propagation-aware estimate error {estimate_error:e} must be strictly smaller \
         than the placeholder error {placeholder_error:e} (analytic reference \
         {reference:e}): the fix must be more accurate, not just passing"
    );

    // And by a wide margin at depth: the placeholder is >100× less accurate, so
    // the improvement is unambiguous rather than marginal.
    assert!(
        placeholder_error / estimate_error > 100.0,
        "placeholder error {placeholder_error:e} should dwarf the estimate error \
         {estimate_error:e} by >100× at depth"
    );
}
