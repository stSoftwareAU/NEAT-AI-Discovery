//! Hygiene removal-eligibility, ranked with an honest gain (Issue #1519).
//!
//! Over-threshold "broken" neurons (squash error > [`MAX_REASONABLE_SQUASH_ERROR`])
//! must stay **removal-eligible** — they break downstream WASM compilation of the
//! exported network, so the NEAT-AI `#2483` hygiene guarantee must not regress.
//! But they must no longer be ranked with the fabricated `[0.1, 0.5]` floor gain
//! that crowds out realistic (~`1e-4`) improvement candidates.
//!
//! [`assess_remove_neuron`] decouples the two concerns:
//! - **Removal-eligibility** comes from the hygiene threshold (squash error), so
//!   a broken neuron is removed regardless of its honest gain.
//! - **Ranking value** is the honest, propagation-aware
//!   [`estimate_remove_neuron_gain`] estimate (≈0 or negative), never the
//!   synthetic floor.

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::{
    MAX_REASONABLE_SQUASH_ERROR, RemoveNeuronAssessment, assess_remove_neuron,
    estimate_remove_neuron_gain,
};

/// Build a `CreatureJson` from a compact JSON description.
fn creature(json: &str) -> CreatureJson {
    serde_json::from_str(json).expect("valid creature JSON")
}

/// A creature whose hidden neuron `broken` sits behind a heavily-shared output
/// inbound budget (weight 1 of 20), so its propagation-aware influence — and
/// hence its honest remove gain magnitude — is a small `0.05`, well below the
/// retired `[0.1, 0.5]` floor. The neuron is genuinely connected (it carries
/// real, if small, downstream influence) so its honest gain is non-positive.
fn creature_with_deep_neuron() -> CreatureJson {
    creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "broken", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "broken", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "broken", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 19.0}
            ]
        }"#,
    )
}

/// Hygiene guarantee (#2483 must not regress): an over-threshold neuron stays
/// removal-eligible even when its honest gain is ≈0 or negative — the property
/// that would otherwise exclude it from the gain-driven removal path.
#[test]
fn over_threshold_neuron_removed_despite_nonpositive_gain() {
    let c = creature_with_deep_neuron();

    // Squash error well above the hygiene threshold — the neuron is "broken".
    let squash_error = MAX_REASONABLE_SQUASH_ERROR * 10.0;

    let assessment = assess_remove_neuron(&c, "broken", squash_error)
        .expect("hidden neuron must yield an assessment");

    // Its honest gain is non-positive (removal cannot fabricate a win) — exactly
    // the case that would normally keep it out of the removal set.
    assert!(
        assessment.gain <= 0.0,
        "honest gain should be non-positive, got {}",
        assessment.gain
    );

    // ...yet hygiene keeps it removal-eligible regardless of that gain.
    assert!(
        assessment.removal_eligible,
        "over-threshold broken neuron must remain removal-eligible (hygiene, #2483)"
    );
}

/// Fabricated-gain guarantee: the reported gain equals the honest
/// propagation-aware estimate (≈0 or negative), is NOT inside the retired
/// `[0.1, 0.5]` synthetic floor, and ranks below a genuine ~`1e-4` candidate so
/// it no longer crowds real improvements out of the top of the list.
#[test]
fn over_threshold_gain_is_honest_not_floored() {
    let c = creature_with_deep_neuron();
    let squash_error = MAX_REASONABLE_SQUASH_ERROR * 10.0;

    let assessment = assess_remove_neuron(&c, "broken", squash_error)
        .expect("hidden neuron must yield an assessment");

    // The reported gain IS the honest estimate — not synthesised from the error.
    let honest = estimate_remove_neuron_gain(&c, "broken").expect("estimator gain");
    assert!(
        (assessment.gain - honest).abs() < 1e-12,
        "reported gain {} must equal the honest estimate {honest}",
        assessment.gain
    );

    // It must NOT sit in the retired fabricated floor range [0.1, 0.5].
    assert!(
        !(0.1..=0.5).contains(&assessment.gain.abs()),
        "gain {} landed in the retired synthetic floor range [0.1, 0.5]",
        assessment.gain
    );

    // A genuine ~1e-4 improvement candidate ranks strictly above the broken
    // neuron — no crowd-out. (Ranking is by gain, descending.)
    let genuine_candidate_gain = 1e-4_f64;
    assert!(
        genuine_candidate_gain > assessment.gain,
        "a genuine {genuine_candidate_gain} candidate must outrank the broken neuron's honest gain {}",
        assessment.gain
    );
}

/// Decoupling in the other direction: a neuron whose squash error is *below* the
/// hygiene threshold is NOT made removal-eligible by this hygiene path — its
/// eligibility is left to the ordinary gain-driven removal logic. This proves
/// removal-eligibility is driven by the threshold, not by the gain value (which
/// is identical to the over-threshold case).
#[test]
fn below_threshold_neuron_is_not_hygiene_eligible() {
    let c = creature_with_deep_neuron();

    // Just under the threshold — a normal, non-broken neuron.
    let squash_error = MAX_REASONABLE_SQUASH_ERROR * 0.5;

    let assessment = assess_remove_neuron(&c, "broken", squash_error)
        .expect("hidden neuron must yield an assessment");

    assert!(
        !assessment.removal_eligible,
        "a below-threshold neuron must not be forced removal-eligible by the hygiene path"
    );
    // The honest gain is reported regardless of eligibility (decoupled).
    let honest = estimate_remove_neuron_gain(&c, "broken").expect("estimator gain");
    assert!((assessment.gain - honest).abs() < 1e-12);
}

/// Output and unknown neurons are never removal candidates, so no assessment is
/// produced even at an over-threshold error.
#[test]
fn output_and_unknown_neurons_yield_no_assessment() {
    let c = creature_with_deep_neuron();
    let squash_error = MAX_REASONABLE_SQUASH_ERROR * 10.0;

    assert!(
        assess_remove_neuron(&c, "out-0", squash_error).is_none(),
        "output neurons are never remove candidates"
    );
    assert!(
        assess_remove_neuron(&c, "does-not-exist", squash_error).is_none(),
        "unknown neurons yield no assessment"
    );
}

/// The assessment is a plain value type carrying the two decoupled fields.
#[test]
fn assessment_fields_are_decoupled_values() {
    let a = RemoveNeuronAssessment {
        removal_eligible: true,
        gain: -0.05,
    };
    assert!(a.removal_eligible);
    assert!((a.gain - -0.05).abs() < 1e-12);
}
