//! Propagation-aware change-squash effect at depth (Issue #1532, re-based by
//! Issue #1722).
//!
//! Executable specification extending the #1516/#1518 propagation-aware fix from
//! the **remove-neuron** estimate path to the **change-squash** estimate path.
//!
//! Changing a neuron's activation function on a converged network perturbs the
//! neuron's emitted output; that perturbation is diluted/squashed through every
//! intervening weight and activation on the way to the output(s), and — because
//! the downstream layers were trained around the neuron's *original* activation
//! — it typically makes the trained creature slightly *worse*. The retired
//! pipeline placeholder ignored all of this and fabricated a near-zero positive
//! gain regardless of topology or activation.
//!
//! Fixtures (committed under `tests/fixtures/change_squash_propagation/`, all
//! hand-authored and synthetic so this public repository stays self-contained —
//! Issue #1722):
//! - `v2_change-squash_spine-1.json` — a change-squash candidate record carrying
//!   the retired near-zero placeholder gain, the neuron's local errors under the
//!   current/proposed squash, and the closed-form propagated effect.
//! - the deep-chain creature topology is shared with the remove-neuron fixture
//!   (`../remove_neuron_propagation/network.json`) — both candidates sit on the
//!   same creature — so it is not duplicated.
//!
//! The reference effect is derivable by hand: `spine-1` sits 12 halving hops
//! from the output (influence exactly `0.5^12 = 2.44140625e-4`) and the swap
//! reduces the neuron's local error by `3.0 − 1.0 = 2.0`, so the honest gain is
//! `−(2.44140625e-4 × 2.0) = −4.8828125e-4`.
//!
//! This file ships the guards that mirror `tests/remove_neuron_propagation.rs`:
//! - [`change_squash_placeholder_is_wrong_at_depth`] — the placeholder-guard: the
//!   propagation-aware estimator must never emit the near-zero placeholder range
//!   for this deep candidate. If a fallback branch resurrects the inaccurate
//!   near-zero formula, CI turns red.
//! - [`change_squash_effect_at_depth`] — the estimator spec: the estimate must
//!   equal the analytic reference effect (and so pass the #1529 criterion).

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::estimate_change_squash_gain;
use std::path::{Path, PathBuf};

/// The near-zero placeholder gain carried by the candidate record
/// (`expectedCreatureScoreGain`). Topology-blind and activation-blind.
const PLACEHOLDER_GAIN: f64 = 5e-10;

/// The closed-form propagated effect of the squash swap
/// (`analyticErrorReduction`): `−(0.5^12 × 2.0)`. Negative — the change makes
/// the trained network slightly worse, the opposite of the placeholder's sign.
const REFERENCE_EFFECT: f64 = -0.000_488_281_25;

/// The target neuron, 12 halving hops from the single output.
const TARGET_NEURON: &str = "spine-1";

/// The synthetic snapshot's dimensions — the shape the analytic reference was
/// derived from.
const EXPECTED_NEURONS: usize = 27;
const EXPECTED_SYNAPSES: usize = 40;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/change_squash_propagation")
}

/// Change-squash candidate record: the fields the estimator consumes plus the
/// analytic reference effect it is graded against.
struct CandidateFixture {
    neuron_uuid: String,
    /// The neuron's local error under its current squash (`currentError`).
    current_local_error: f64,
    /// The neuron's local error under the proposed squash (`improvedError`).
    proposed_local_error: f64,
    /// The recorded placeholder `expectedCreatureScoreGain`.
    placeholder_gain: f64,
    /// The closed-form `analyticErrorReduction`.
    reference_effect: f64,
}

/// Load the candidate record. A missing / corrupt fixture fails here with the
/// fixture path, so hermeticity breakage is caught in the same CI run rather
/// than downstream.
fn load_candidate_fixture() -> CandidateFixture {
    let path = fixture_dir().join("v2_change-squash_spine-1.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read candidate fixture {}: {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse candidate fixture {}: {e}", path.display()));

    let candidate = &json["rustRequest"]["squashCandidate"];
    CandidateFixture {
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
        reference_effect: json["analyticErrorReduction"]
            .as_f64()
            .expect("fixture missing analyticErrorReduction"),
    }
}

/// Load the deep-chain creature topology. Shared with the remove-neuron fixture
/// (both candidates sit on the same creature), so it is not duplicated here.
fn load_network() -> CreatureJson {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/remove_neuron_propagation/network.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read network fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse network fixture {}: {e}", path.display()))
}

/// Load the topology and assert it is the shape the analytic reference was
/// derived from. A changed shape invalidates the reference, so it must fail
/// loudly rather than silently grade against new arithmetic.
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

/// The #1529 accuracy pass criterion: an estimate passes when it matches the
/// reference error change within one order of magnitude (10×) AND in sign.
fn meets_pass_criterion(estimate: f64, reference: f64) -> bool {
    within_one_order(estimate, reference) && estimate.signum() == reference.signum()
}

/// Placeholder-guard. The propagation-aware estimator has replaced the near-zero
/// placeholder, so this asserts the estimator no longer emits a value anywhere
/// near the fabricated `~5e-10` placeholder for this deep candidate. If the
/// near-zero placeholder path is ever accidentally reinstated (e.g. a fallback
/// branch resurfaces), this test turns CI red at that commit.
///
/// It still pins the committed fixture values so drift in the known-bad
/// placeholder / analytic reference constants is caught in the same run.
#[test]
fn change_squash_placeholder_is_wrong_at_depth() {
    let fixture = load_candidate_fixture();
    let creature = load_expected_network();

    // The fixture still encodes the known-bad placeholder and analytic reference.
    // Drift in either value flips this guard red.
    assert!(
        (fixture.placeholder_gain - PLACEHOLDER_GAIN).abs() < 1e-18,
        "fixture placeholder gain {} drifted from the known-bad value {PLACEHOLDER_GAIN}; \
         re-validate the fixture and spec",
        fixture.placeholder_gain
    );
    assert!(
        (fixture.reference_effect - REFERENCE_EFFECT).abs() < 1e-12,
        "fixture analytic reference {} drifted from {REFERENCE_EFFECT}; \
         re-validate the fixture and spec",
        fixture.reference_effect
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
    // near-zero placeholder — nowhere near the fabricated `~5e-10`.
    assert!(
        magnitude_ratio(estimate, PLACEHOLDER_GAIN) > 1_000.0,
        "estimate {estimate:e} is within the retired near-zero placeholder range \
         (placeholder {PLACEHOLDER_GAIN:e}); the fabricated near-zero path has resurfaced"
    );
}

/// Estimator spec — the permanent regression gate (#1532).
///
/// The propagation-aware estimator ([`estimate_change_squash_gain`]) replaces the
/// fabricated near-zero placeholder. Because the committed topology attenuates by
/// exactly one half per hop, the honest gain is derivable by hand:
/// `−(0.5^12 × 2.0)`. This case asserts the estimator emits precisely that. Any
/// future estimator change that breaks the scale or sign fidelity fails
/// `cargo test` in CI before merge.
#[test]
fn change_squash_effect_at_depth() {
    let fixture = load_candidate_fixture();
    let creature = load_expected_network();

    let estimate = estimate_change_squash_gain(
        &creature,
        &fixture.neuron_uuid,
        fixture.current_local_error,
        fixture.proposed_local_error,
    )
    .expect("estimator must return a gain for the target hidden neuron");

    // The near-zero placeholder FAILS the #1529 pass criterion — wrong sign
    // (fabricated positive vs propagated negative) and vastly more than 10× off.
    assert!(
        !meets_pass_criterion(fixture.placeholder_gain, fixture.reference_effect),
        "placeholder {:e} must fail the #1529 pass criterion against the analytic \
         reference {:e}",
        fixture.placeholder_gain,
        fixture.reference_effect
    );

    // The placeholder fails on BOTH counts: wrong sign (fabricated positive) and
    // vastly off in magnitude (ratio ~1e-6, far below the 0.1 floor).
    assert_ne!(
        fixture.placeholder_gain.signum(),
        fixture.reference_effect.signum(),
        "placeholder {:e} should have the wrong sign vs the analytic reference {:e}",
        fixture.placeholder_gain,
        fixture.reference_effect
    );
    assert!(
        !within_one_order(fixture.placeholder_gain, fixture.reference_effect),
        "placeholder {:e} should be far more than 10× off the analytic reference {:e}",
        fixture.placeholder_gain,
        fixture.reference_effect
    );

    // The executable specification: the propagation-aware estimator emits the
    // analytic reference effect exactly, and so passes the #1529 criterion.
    assert!(
        (estimate - fixture.reference_effect).abs() < 1e-12,
        "estimate {estimate:e} must equal the analytic reference effect {:e}",
        fixture.reference_effect
    );
    assert!(
        meets_pass_criterion(estimate, fixture.reference_effect),
        "propagation-aware estimate {estimate:e} must pass the #1529 pass criterion \
         against the analytic reference {:e}",
        fixture.reference_effect
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
        .expect("creature must have at least one output neuron");
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
    let fixture = load_candidate_fixture();
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
