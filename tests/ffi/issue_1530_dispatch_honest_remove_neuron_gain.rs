//! Dispatch-path wiring for the propagation-aware remove-neuron estimator
//! (Issue #1530) — make milestone #1516 live end-to-end.
//!
//! Milestone #1516 merged [`estimate_remove_neuron_gain`] (PR #1523), but
//! nothing in `src/ffi/` or `discovery_dispatch.rs` invoked it, so the live
//! remove-neuron path still reported the fabricated NEAT-AI `#2483` placeholder
//! gain (`+0.17879` on creature `45a04ef1`, versus a measured `−0.00032`).
//!
//! [`apply_honest_remove_neuron_gain`] is the dispatch seam the orchestration
//! calls on the assembled coordinated candidates before the drought demotion
//! and final gain floor. These tests drive that seam with a deliberately wrong
//! request-supplied gain and assert the returned gain comes from the honest,
//! propagation-aware estimator — it does **not** echo the supplied value.
//!
//! The `production_depth` test reproduces the `45a04ef1` failure end-to-end on
//! the committed production topology fixture: a deep neuron carrying the
//! placeholder gain is corrected to an estimate that tracks the measured actual
//! effect in sign and magnitude rather than the `+0.17879` placeholder.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts (Issue #873)

use std::path::{Path, PathBuf};

use neat_ai_discovery::analysis::discovery_dispatch::apply_honest_remove_neuron_gain;
use neat_ai_discovery::analysis::estimate_remove_neuron_gain;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson,
};

/// The fabricated placeholder gain (`expectedCreatureScoreGain`) recorded for
/// the failure example — reproduced exactly by the NEAT-AI `#2483`
/// over-threshold sink `0.1 + (log10(err) − 10)/10 × 0.4`.
const PLACEHOLDER_GAIN: f32 = 0.178_829_21;

/// The empirically measured effect of the removal on the output
/// (`actualErrorReduction`, ~69k samples). Negative: removal made the network
/// slightly worse — the opposite of the placeholder's sign.
const MEASURED_ACTUAL: f64 = -0.000_194_478_429_877_298_35;

/// The target neuron, sitting many layers from the single output.
const TARGET_NEURON: &str = "neuron-1802938338";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/remove_neuron_propagation")
}

/// Load the production creature topology from the committed fixture.
fn load_network() -> CreatureJson {
    let path = fixture_dir().join("network.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read network fixture {}: {e}", path.display()));
    serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("failed to parse network fixture {}: {e}", path.display()))
}

fn remove_neuron_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
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

    let honest = estimate_remove_neuron_gain(&creature, "deep").expect("estimator gain");
    let reported = f64::from(candidates[0].expected_creature_score_gain);

    assert!(
        (reported - honest).abs() < 1e-6,
        "dispatch must report the honest estimate {honest}, not the supplied 0.5 (got {reported})"
    );
    assert!(
        (reported - 0.5).abs() > 1e-6,
        "dispatch must NOT echo the request-supplied gain 0.5 (got {reported})"
    );
    assert!(
        reported < 0.0,
        "a genuinely-connected neuron's honest remove gain is negative, got {reported}"
    );
}

/// End-to-end reproduction of the `45a04ef1` failure on the committed
/// production topology: the deep neuron's request-supplied placeholder gain
/// (`+0.17879`) is corrected to an estimate that tracks the measured actual
/// effect (`−0.000194`) in sign and magnitude.
#[test]
fn dispatch_tracks_measured_actual_at_production_depth() {
    let creature = load_network();

    // The request supplies the known-bad placeholder gain for the deep neuron.
    let mut candidates = vec![remove_neuron_candidate(TARGET_NEURON, PLACEHOLDER_GAIN)];
    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(
        overridden, 1,
        "the production-depth candidate is overridden"
    );

    let reported = f64::from(candidates[0].expected_creature_score_gain);

    // No longer the placeholder fingerprint (large positive ~0.17879).
    assert!(
        !(0.1..=0.5).contains(&reported.abs()),
        "reported gain {reported} still sits in the retired placeholder floor range [0.1, 0.5]"
    );
    assert!(
        magnitude_ratio(f64::from(PLACEHOLDER_GAIN), reported) > 100.0,
        "the placeholder {PLACEHOLDER_GAIN} should dwarf the honest gain {reported} (>100×)"
    );

    // The honest gain tracks the measured actual's sign and magnitude: removing
    // a neuron that still carries downstream influence is a small net loss.
    assert!(
        reported < 0.0 && reported.signum() == MEASURED_ACTUAL.signum(),
        "honest gain {reported} must share the measured actual's negative sign"
    );
    let ratio = magnitude_ratio(reported, MEASURED_ACTUAL);
    assert!(
        (0.1..=10.0).contains(&ratio),
        "honest gain {reported} must be within one order of magnitude of the measured actual \
         {MEASURED_ACTUAL} (ratio {ratio})"
    );
}
