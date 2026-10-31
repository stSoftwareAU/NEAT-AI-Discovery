//! Regression tests for Issue #2375 (CWE-682): `detect_squash_weight_rescale_candidates`
//! computed `new_weight = old_weight * rescale_factor` without checking the product.
//! A finite incoming weight near `f32::MAX` multiplied by a factor above 1.0
//! overflowed to `±inf`, and `squash_weight_rescale_to_coordinated_candidates`
//! then emitted a `setWeight` operation carrying that infinity.
//!
//! The fixture makes `find_best_rescale_factor` pick a factor of 4.0 using
//! ordinary inputs: the hidden neuron is IDENTITY and its recorded targets are
//! `tanh(4x)`, so TANH at factor 4.0 reproduces the targets exactly. With an
//! incoming weight of 1e38 the product is 4e38, which exceeds `f32::MAX`
//! (about 3.4e38).
//!
//! The positive control runs the identical fixture with an incoming weight of
//! 0.5. That must produce a candidate whose `setWeight` equals `0.5 * 4.0`, which
//! shows the fixture does reach the rescale. Only then does the overflow case
//! being dropped prove anything about the overflow itself.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use neat_ai_discovery::analysis::detection::squash_weight_rescale::{
    SquashWeightRescaleCandidate, detect_squash_weight_rescale_candidates,
    squash_weight_rescale_to_coordinated_candidates,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};

/// Rescale factor the grid search must select for this fixture.
const EXPECTED_FACTOR: f32 = 4.0;

/// Ordinary incoming weight used by the positive control.
const ORDINARY_WEIGHT: f32 = 0.5;

/// Incoming weight whose product with `EXPECTED_FACTOR` overflows `f32`.
const HUGE_WEIGHT: f32 = 1e38;

// =============================================================================
// Helpers
// =============================================================================

fn neuron(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// `input-1` -> `hidden-1` (incoming weight `incoming_weight`) -> `output-1`.
fn make_creature(incoming_weight: f32) -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-1", "input", "IDENTITY"),
            neuron("hidden-1", "hidden", "IDENTITY"),
            neuron("output-1", "output", "TANH"),
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: incoming_weight,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    }
}

/// 40 records for `hidden-1` with pre-activation `x` in [-2.0, 1.9].
///
/// The hidden neuron is IDENTITY, so its activation is `x`. The target is
/// `tanh(4x)`, and `error = activation - target` as the detector expects.
fn make_neuron_records() -> Vec<(String, Vec<DiscoverRecord>)> {
    let records = (0..40u32)
        .map(|i| {
            let x = (i as f32 - 20.0) / 10.0;
            let activation = x;
            let target = (EXPECTED_FACTOR * x).tanh();
            DiscoverRecord {
                obs_index: i,
                neuron_uuid: "hidden-1".to_string(),
                value: Some(x),
                activation,
                errors: vec![activation - target],
            }
        })
        .collect();
    vec![("hidden-1".to_string(), records)]
}

/// Runs detection and coordinated conversion for the fixture with the given weight.
fn run(
    incoming_weight: f32,
) -> (
    Vec<SquashWeightRescaleCandidate>,
    Vec<CoordinatedStructuralCandidateJson>,
) {
    let creature = make_creature(incoming_weight);
    let hidden = vec![("hidden-1".to_string(), "IDENTITY".to_string(), 0.0)];
    let records = make_neuron_records();
    let candidates = detect_squash_weight_rescale_candidates(&creature, &hidden, &records);
    let coordinated = squash_weight_rescale_to_coordinated_candidates(&candidates);
    (candidates, coordinated)
}

// =============================================================================
// Positive control
// =============================================================================

#[test]
fn ordinary_weight_fixture_reaches_the_rescale() {
    let (candidates, coordinated) = run(ORDINARY_WEIGHT);

    assert_eq!(
        candidates.len(),
        1,
        "ordinary-weight fixture must yield exactly one candidate, got {}",
        candidates.len()
    );
    let candidate = &candidates[0];
    assert_eq!(candidate.neuron_uuid, "hidden-1");
    assert_eq!(candidate.recommended_squash, "TANH");

    let expected = ORDINARY_WEIGHT * EXPECTED_FACTOR;
    assert_eq!(candidate.rescaled_weights.len(), 1);
    let (from, to, weight) = &candidate.rescaled_weights[0];
    assert_eq!(from, "input-1");
    assert_eq!(to, "hidden-1");
    assert!(
        weight.is_finite(),
        "rescaled weight must be finite, got {weight}"
    );
    assert!(
        weight.abs() > ORDINARY_WEIGHT.abs(),
        "rescaled weight {weight} must exceed old weight {ORDINARY_WEIGHT} in magnitude"
    );
    assert!(
        (weight - expected).abs() < 1e-6,
        "rescaled weight must equal old_weight * factor = {expected}, got {weight}"
    );

    assert_eq!(coordinated.len(), 1);
    let set_weights: Vec<f32> = coordinated[0]
        .operations
        .iter()
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::SetWeight { weight, .. } => Some(*weight),
            _ => None,
        })
        .collect();
    assert_eq!(
        set_weights.len(),
        1,
        "expected one setWeight, got {set_weights:?}"
    );
    assert!(
        (set_weights[0] - expected).abs() < 1e-6,
        "setWeight must carry {expected}, got {}",
        set_weights[0]
    );
}

// =============================================================================
// Regression
// =============================================================================

#[test]
fn rescaled_weights_are_never_non_finite() {
    let (candidates, coordinated) = run(HUGE_WEIGHT);

    // Every rescaled weight kept on a detected candidate must be finite.
    for candidate in &candidates {
        for (from, to, weight) in &candidate.rescaled_weights {
            assert!(
                weight.is_finite(),
                "detected rescaled weight {from} -> {to} is non-finite: {weight}"
            );
        }
    }

    // Every setWeight actually emitted must be finite.
    for candidate in &coordinated {
        for op in &candidate.operations {
            if let CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } = op
            {
                assert!(
                    weight.is_finite(),
                    "emitted setWeight {from_neuron_uuid} -> {to_neuron_uuid} is non-finite: {weight}"
                );
            }
        }
    }

    // The overflowing candidate is dropped for that neuron, not emitted.
    // The positive control above shows this fixture otherwise yields a candidate.
    assert!(
        candidates.is_empty(),
        "candidate with overflowing rescale must be dropped, got {} candidate(s)",
        candidates.len()
    );
    assert!(
        coordinated.is_empty(),
        "no coordinated candidate may be emitted for the overflowing rescale"
    );
}
