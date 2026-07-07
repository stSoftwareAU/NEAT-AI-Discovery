//! Unit tests for the propagation-aware remove-neuron gain estimator
//! (Issue #1518).
//!
//! These exercise the estimator on small synthetic topologies where the
//! expected attenuation is easy to reason about, complementing the
//! production-depth fixture spec in `remove_neuron_propagation.rs`.

use neat_ai_discovery::CreatureJson;
use neat_ai_discovery::analysis::estimate_remove_neuron_gain;

/// Build a `CreatureJson` from a compact JSON description.
fn creature(json: &str) -> CreatureJson {
    serde_json::from_str(json).expect("valid creature JSON")
}

/// A hidden neuron one hop from the output, sharing the output's inbound weight
/// budget with another hidden neuron, has a bounded (< 1.0) but non-trivial
/// influence — so removing it yields a small negative honest gain.
#[test]
fn shallow_hidden_neuron_has_small_negative_gain() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "h1", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "h2", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "h1", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "h2", "weight": 1.0},
                {"fromUUID": "h1", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "h2", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );

    let gain = estimate_remove_neuron_gain(&c, "h1").expect("gain for hidden neuron");
    // Honest gain is non-positive (removal cannot fabricate a win).
    assert!(gain <= 0.0, "expected non-positive gain, got {gain}");
    // h1 carries half the output's inbound weight, so |gain| ≈ 0.5.
    assert!(
        (gain - -0.5).abs() < 1e-6,
        "expected ~-0.5 for a 1-of-2 inbound neuron, got {gain}"
    );
}

/// A neuron deeper in the network — sharing its inbound weight budget with a
/// sibling at each layer on the way to the output — attenuates further, so its
/// honest gain magnitude is smaller than a shallower neuron's. This is the
/// propagation-aware behaviour the placeholder lacked.
///
/// Topology (all IDENTITY, all weight 1):
/// `deep`/`other` → `mid`; `mid`/`shallow` → `out`.
/// - `shallow` shares the output's inbound budget once → influence 0.5.
/// - `deep` shares `mid`'s budget (0.5) and `mid` shares the output's (0.5) →
///   influence 0.25.
#[test]
fn deeper_neuron_attenuates_below_shallower_neuron() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "deep", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "other", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "mid", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "shallow", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "deep", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "other", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "shallow", "weight": 1.0},
                {"fromUUID": "deep", "toUUID": "mid", "weight": 1.0},
                {"fromUUID": "other", "toUUID": "mid", "weight": 1.0},
                {"fromUUID": "mid", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "shallow", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );

    let deep = estimate_remove_neuron_gain(&c, "deep").expect("deep gain");
    let shallow = estimate_remove_neuron_gain(&c, "shallow").expect("shallow gain");

    assert!(deep <= 0.0 && shallow <= 0.0);
    assert!(
        deep.abs() < shallow.abs(),
        "deep neuron {deep} should attenuate below shallow neuron {shallow}"
    );
}

/// Output neurons are never remove-neuron candidates.
#[test]
fn output_neuron_returns_none() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "h1", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "h1", "weight": 1.0},
                {"fromUUID": "h1", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );

    assert!(estimate_remove_neuron_gain(&c, "out-0").is_none());
}

/// An unknown neuron UUID yields `None` rather than a fabricated value.
#[test]
fn unknown_neuron_returns_none() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );

    assert!(estimate_remove_neuron_gain(&c, "does-not-exist").is_none());
}

/// A hidden neuron with no downstream path to the output has zero influence, so
/// its honest gain is ~0 — neither a fabricated win nor a large loss.
#[test]
fn disconnected_neuron_has_zero_gain() {
    let c = creature(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "orphan", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    );

    let gain = estimate_remove_neuron_gain(&c, "orphan").expect("gain for orphan");
    assert!(
        gain.abs() < 1e-9,
        "disconnected neuron should have ~0 gain, got {gain}"
    );
}
