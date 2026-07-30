//! Dispatch-path wiring for the propagation-aware remove-neuron estimator
//! (Issue #1530) — make milestone #1516 live end-to-end.
//!
//! Milestone #1516 merged [`estimate_remove_neuron_gain`] (PR #1523), but
//! nothing in `src/ffi/` or `discovery_dispatch.rs` invoked it, so the live
//! remove-neuron path still reported the fabricated NEAT-AI `#2483` placeholder
//! gain — a large positive value derived from the neuron's squash error alone,
//! regardless of how deep in the network it sat.
//!
//! [`apply_honest_remove_neuron_gain`] is the dispatch seam the orchestration
//! calls on the assembled coordinated candidates before the drought demotion
//! and final gain floor. These tests drive that seam with a deliberately wrong
//! request-supplied gain and assert the returned gain is derived from the
//! honest, propagation-aware estimator — it does **not** echo the supplied value.
//!
//! The depth test reproduces the failure end-to-end on the committed deep-chain
//! topology fixture (hand-authored and synthetic — Issue #1722): a deep neuron
//! carrying the placeholder gain is corrected to a value derived from the
//! analytic propagated effect rather than the placeholder.
//!
//! # Updated by Issue #1812
//!
//! The seam no longer writes the estimator's output verbatim. That output is a
//! **unitless** influence fraction, negated — a cost term — and writing it into
//! `expectedCreatureScoreGain` compared it against a floor denominated in
//! creature-score benefit, dropping every sole-op removal by construction
//! (#1785/#1810). The seam now writes
//! `saving − |influence| × REMOVE_INFLUENCE_CALIBRATION`, so these tests assert
//! on the estimator's tracking of the analytic reference (unchanged) *and* on
//! the conversion that turns it into the emitted gain.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use std::path::{Path, PathBuf};

use neat_ai_discovery::analysis::discovery_dispatch::apply_honest_remove_neuron_gain;
use neat_ai_discovery::analysis::{
    analysis_cost_of_growth, estimate_remove_neuron_gain, removal_net_gain,
};
use neat_ai_discovery::focus::{SynapseCounts, calculate_removal_savings};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson,
};

/// The fabricated placeholder gain (`expectedCreatureScoreGain`) carried by the
/// committed candidate record — reproduced exactly by the NEAT-AI `#2483`
/// over-threshold sink `0.1 + (log10(err) − 10)/10 × 0.4` at `err = 1e12`.
const PLACEHOLDER_GAIN: f32 = 0.18;

/// The closed-form propagated effect of the removal on the output
/// (`analyticErrorReduction`): the negation of the target neuron's exact `0.5^13`
/// share of the output's weight budget. Negative: removal makes the network
/// slightly worse — the opposite of the placeholder's sign.
const REFERENCE_EFFECT: f64 = -0.000_122_070_312_5;

/// The target neuron, 13 halving hops from the single output.
const TARGET_NEURON: &str = "spine-0";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remove_neuron_propagation")
}

/// Load the deep-chain creature topology from the committed fixture.
fn load_network() -> CreatureJson {
    let path = fixture_dir().join("network.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read network fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse network fixture {}: {e}", path.display()))
}

fn remove_neuron_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some("request-supplied placeholder gain".to_string()),
    }
}

/// Ratio of two magnitudes, guarding against a zero denominator.
fn magnitude_ratio(a: f64, b: f64) -> f64 {
    let denom = b.abs();
    assert!(denom > 0.0, "magnitude_ratio: zero denominator");
    a.abs() / denom
}

/// A small synthetic creature with a hidden neuron that carries genuine (if
/// small) downstream influence, so its honest remove-gain is strictly negative.
fn creature_with_deep_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "deep", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "deep", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "deep", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 19.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

/// The dispatch seam replaces the request-supplied remove-neuron gain with the
/// honest, propagation-aware estimate — the returned gain does not echo the
/// deliberately wrong supplied value.
#[test]
fn dispatch_replaces_request_supplied_gain_with_honest_estimate() {
    let creature = creature_with_deep_neuron();

    // Drive the dispatch path with a deliberately wrong request-supplied gain.
    let mut candidates = vec![remove_neuron_candidate("deep", 0.5)];
    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(
        overridden, 1,
        "the single RemoveNeuron candidate is overridden"
    );

    let influence = estimate_remove_neuron_gain(&creature, "deep").expect("estimator gain");
    // `deep` has one incoming and one outgoing synapse.
    let saving = calculate_removal_savings(1, 1, analysis_cost_of_growth());
    let expected = f64::from(removal_net_gain(saving, influence));
    let reported = f64::from(candidates[0].expected_creature_score_gain);

    assert!(
        (reported - expected).abs() < 1e-12,
        "dispatch must report the net benefit derived from the honest estimate {influence} \
         (expected {expected}), not the supplied 0.5 (got {reported})"
    );
    assert!(
        (reported - 0.5).abs() > 1e-6,
        "dispatch must NOT echo the request-supplied gain 0.5 (got {reported})"
    );
    assert!(
        reported < 0.0,
        "a genuinely-connected neuron's removal must net a negative benefit, got {reported}"
    );
}

/// End-to-end reproduction of the failure on the committed deep-chain topology:
/// the deep neuron's request-supplied placeholder gain (`+0.18`) is corrected to
/// an estimate that tracks the analytic propagated effect (`−1.22e-4`) in sign
/// and magnitude.
#[test]
fn dispatch_tracks_analytic_reference_at_depth() {
    let creature = load_network();

    // The request supplies the known-bad placeholder gain for the deep neuron.
    let mut candidates = vec![remove_neuron_candidate(TARGET_NEURON, PLACEHOLDER_GAIN)];
    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(overridden, 1, "the depth candidate is overridden");

    let reported = f64::from(candidates[0].expected_creature_score_gain);

    // No longer the placeholder fingerprint (large positive ~0.18).
    assert!(
        !(0.1..=0.5).contains(&reported.abs()),
        "reported gain {reported} still sits in the retired placeholder floor range [0.1, 0.5]"
    );
    assert!(
        magnitude_ratio(f64::from(PLACEHOLDER_GAIN), reported) > 100.0,
        "the placeholder {PLACEHOLDER_GAIN} should dwarf the reported gain {reported} (>100×)"
    );

    // The estimator — the gain's cost term — still tracks the analytic
    // reference's sign and magnitude. This is the #1530 guarantee, unchanged by
    // #1812's re-denomination.
    let influence = estimate_remove_neuron_gain(&creature, TARGET_NEURON).expect("estimator gain");
    assert!(
        influence < 0.0 && influence.signum() == REFERENCE_EFFECT.signum(),
        "the estimated cost {influence} must share the analytic reference's negative sign"
    );
    let ratio = magnitude_ratio(influence, REFERENCE_EFFECT);
    assert!(
        (0.1..=10.0).contains(&ratio),
        "the estimated cost {influence} must be within one order of magnitude of the analytic \
         reference {REFERENCE_EFFECT} (ratio {ratio})"
    );

    // #1812: the emitted gain is that cost, converted onto the creature-score
    // scale and netted against the exact complexity saving. Still a net loss —
    // the removal does not pay for the influence it destroys.
    let counts = SynapseCounts::new(&creature);
    let (incoming, outgoing) = counts.get(TARGET_NEURON);
    let saving = calculate_removal_savings(incoming, outgoing, analysis_cost_of_growth());
    let expected = f64::from(removal_net_gain(saving, influence));
    assert!(
        (reported - expected).abs() < 1e-12,
        "emitted gain {reported} must be the net benefit {expected} (saving {saving} minus the \
         calibrated cost of {influence})"
    );
    assert!(
        reported < 0.0,
        "a neuron carrying real downstream influence must net a negative benefit, got {reported}"
    );
}
