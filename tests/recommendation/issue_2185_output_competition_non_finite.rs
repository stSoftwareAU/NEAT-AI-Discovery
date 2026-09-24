//! Tests for Issue #2185: hostile-arithmetic hardening of the output
//! competition recommender, wired into dispatch by the same change.
//!
//! `co_activation` accumulates `min(a, b)` over every co-firing sample into an
//! `f32`. Records are read from a discovery parquet file the analyser does not
//! author, so an activation may be non-finite or large enough to overflow the
//! accumulator. `detect_output_competition` then ranks candidates by
//! `estimated_improvement.total_cmp` in descending order, where `+inf` sorts
//! first — a corrupt pair would displace every genuine competitor and carry an
//! unbounded expected gain into the coordinated candidate stream.

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::recommendation::output_competition::{
    detect_output_competition, output_competition_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// The gain ceiling documented by `output_competition`: an estimated
/// improvement is `co_activation_score * COMPETITION_GAIN_SCALE`, and a
/// co-activation score cannot exceed 1.0 for a squashed output neuron.
const DOCUMENTED_GAIN_CEILING: f32 = 0.01;

/// Enough aligned observations to clear `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 40;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    }
}

fn record(uuid: &str, idx: u32, activation: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(0.0),
        activation,
        errors: vec![0.0],
    }
}

/// Every output fires on the same observation indices at the activation the
/// caller nominates, so which pairs compete is decided purely by activation.
fn creature_and_records(
    outputs: &[(&str, f32)],
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("input-1", "input")];
    let mut synapses = Vec::new();
    let mut records = Vec::new();
    for (uuid, activation) in outputs {
        neurons.push(neuron(uuid, "output"));
        synapses.push(SynapseJson {
            from_uuid: "input-1".to_string(),
            to_uuid: (*uuid).to_string(),
            weight: 0.5,
            synapse_type: None,
        });
        records.push((
            (*uuid).to_string(),
            (0..SAMPLES).map(|i| record(uuid, i, *activation)).collect(),
        ));
    }
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: outputs.len(),
    };
    (creature, records)
}

fn one_hot() -> TaskDescriptor {
    TaskDescriptor::from_name("CATEGORICAL_ERROR", 2)
}

// =============================================================================
// 1. Finite activations that overflow the accumulator are dropped, not ranked.
// =============================================================================
#[test]
fn an_overflowing_accumulator_cannot_produce_a_candidate() {
    // `big-a` and `big-b` co-fire at f32::MAX: min(a, b) is f32::MAX and the
    // running sum saturates to +inf after the second sample.
    let (creature, records) = creature_and_records(&[("big-a", f32::MAX), ("big-b", f32::MAX)]);

    let candidates = detect_output_competition(&creature, &records, &one_hot());

    assert!(
        candidates.is_empty(),
        "a pair whose co-activation sum overflows f32 has no representable mean and must be dropped, got {candidates:?}",
    );
}

// =============================================================================
// 2. A poisoned pair cannot displace a genuine competitor at the top of the
//    ranking.
// =============================================================================
#[test]
fn a_poisoned_pair_cannot_outrank_a_genuine_competitor() {
    let (creature, records) = creature_and_records(&[
        ("output-a", 0.85),
        ("output-b", 0.80),
        ("big-a", f32::MAX),
        ("big-b", f32::MAX),
    ]);

    let candidates = detect_output_competition(&creature, &records, &one_hot());

    assert!(
        !candidates.is_empty(),
        "the genuine output-a ↔ output-b competition must still be detected",
    );
    for c in &candidates {
        assert!(
            c.co_activation_score.is_finite(),
            "co-activation score must be finite, got {} for {} ↔ {}",
            c.co_activation_score,
            c.from_output_uuid,
            c.to_output_uuid,
        );
        assert!(
            c.estimated_improvement.is_finite(),
            "estimated improvement must be finite, got {} for {} ↔ {}",
            c.estimated_improvement,
            c.from_output_uuid,
            c.to_output_uuid,
        );
        assert!(
            c.estimated_improvement <= DOCUMENTED_GAIN_CEILING,
            "estimated improvement must stay within the documented gain ceiling, got {} for {} ↔ {}",
            c.estimated_improvement,
            c.from_output_uuid,
            c.to_output_uuid,
        );
        assert!(
            !(c.from_output_uuid == "big-a" && c.to_output_uuid == "big-b"),
            "the overflowing big-a ↔ big-b pair must not be proposed at all",
        );
    }
}

// =============================================================================
// 3. Non-finite activations are ignored rather than counted.
// =============================================================================
#[test]
fn non_finite_activations_are_ignored() {
    for hostile in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let (creature, records) = creature_and_records(&[("output-a", 0.85), ("bad", hostile)]);

        let candidates = detect_output_competition(&creature, &records, &one_hot());

        assert!(
            candidates.is_empty(),
            "an output whose activations are all {hostile} co-fires with nothing, got {candidates:?}",
        );
    }
}

// =============================================================================
// 4. A poisoned pair cannot leak an unbounded gain into the coordinated
//    candidate stream either.
// =============================================================================
#[test]
fn coordinated_candidates_carry_a_bounded_expected_gain() {
    let (creature, records) = creature_and_records(&[
        ("output-a", 0.85),
        ("output-b", 0.80),
        ("big-a", f32::MAX),
        ("big-b", f32::MAX),
    ]);

    let candidates = detect_output_competition(&creature, &records, &one_hot());
    let coordinated = output_competition_to_coordinated_candidates(&candidates);

    assert!(!coordinated.is_empty(), "test prerequisite");
    for c in &coordinated {
        assert!(
            c.expected_creature_score_gain.is_finite()
                && c.expected_creature_score_gain <= DOCUMENTED_GAIN_CEILING,
            "expected gain must be finite and bounded, got {}",
            c.expected_creature_score_gain,
        );
    }
}

// =============================================================================
// 5. Ordinary activations are unaffected by the hardening.
// =============================================================================
#[test]
fn ordinary_co_firing_outputs_still_score_as_before() {
    let (creature, records) = creature_and_records(&[("output-a", 0.85), ("output-b", 0.80)]);

    let candidates = detect_output_competition(&creature, &records, &one_hot());

    assert_eq!(candidates.len(), 1, "one competing pair expected");
    let c = &candidates[0];
    assert!(
        (c.co_activation_score - 0.80).abs() < 1e-6,
        "co-activation is the mean of min(0.85, 0.80), got {}",
        c.co_activation_score,
    );
    assert_eq!(c.sample_count, SAMPLES as usize);
    assert!(
        (c.estimated_improvement - 0.008).abs() < 1e-6,
        "gain is the score scaled by 0.01, got {}",
        c.estimated_improvement,
    );
}
