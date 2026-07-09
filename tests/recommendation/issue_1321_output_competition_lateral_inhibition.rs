//! Tests for Issue #1321: Output competition / lateral-inhibition recommender
//! under `OneHot` / `Simplex` target topologies.
//!
//! Acceptance criteria from the issue:
//!
//! 1. Under a `OneHot`/`Simplex` descriptor, the recommender proposes
//!    inhibitory output↔output connections (or a normalisation hint) when
//!    two output neurons co-fire on the same samples.
//! 2. `OTHER` / `Unknown` / absent ⇒ nothing emitted (regression guard).

#![allow(clippy::cast_precision_loss)]

use neat_ai_discovery::CoordinatedStructuralOpJson;
use neat_ai_discovery::analysis::recommendation::output_competition::{
    detect_output_competition, output_competition_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

fn make_record(uuid: &str, idx: u32, value: f32, activation: f32, error: f32) -> DiscoverRecord {
    DiscoverRecord {
        obs_index: idx,
        neuron_uuid: uuid.to_string(),
        value: Some(value),
        activation,
        errors: vec![error],
    }
}

fn neuron(uuid: &str, neuron_type: &str, bias: f32) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias,
    }
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Creature with one input feeding two output neurons (forward-only order
/// preserves `output-a` before `output-b`).
fn creature_with_two_outputs() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-a", "output", 0.0),
            neuron("output-b", "output", 0.0),
        ],
        synapses: vec![
            synapse("input-1", "output-a", 0.5),
            synapse("input-1", "output-b", 0.5),
        ],
        input: 1,
        output: 2,
    }
}

fn creature_with_one_output() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("input-1", "input", 0.0),
            neuron("output-a", "output", 0.0),
        ],
        synapses: vec![synapse("input-1", "output-a", 0.5)],
        input: 1,
        output: 1,
    }
}

/// Records where two output neurons fire strongly on the same samples —
/// classic "competition" pattern that lateral inhibition should resolve.
fn co_firing_records() -> Vec<(String, Vec<DiscoverRecord>)> {
    let mut a_records = Vec::new();
    let mut b_records = Vec::new();
    for i in 0..40 {
        // On every sample both outputs fire strongly: this is the
        // co-activation pattern we want to inhibit. Target alternates
        // between the two classes so neither output is the dominant
        // truth signal.
        let (target_a, target_b) = if i % 2 == 0 { (1.0, 0.0) } else { (0.0, 1.0) };
        a_records.push(make_record("output-a", i, target_a, 0.85, target_a - 0.85));
        b_records.push(make_record("output-b", i, target_b, 0.80, target_b - 0.80));
    }
    vec![
        ("output-a".to_string(), a_records),
        ("output-b".to_string(), b_records),
    ]
}

/// Records where two output neurons fire on disjoint sample sets — they
/// are *not* competing. No lateral inhibition should be proposed.
fn disjoint_firing_records() -> Vec<(String, Vec<DiscoverRecord>)> {
    let mut a_records = Vec::new();
    let mut b_records = Vec::new();
    for i in 0..40 {
        if i < 20 {
            // a fires strongly, b stays low.
            a_records.push(make_record("output-a", i, 1.0, 0.90, 0.10));
            b_records.push(make_record("output-b", i, 0.0, 0.05, -0.05));
        } else {
            // b fires strongly, a stays low.
            a_records.push(make_record("output-a", i, 0.0, 0.05, -0.05));
            b_records.push(make_record("output-b", i, 1.0, 0.90, 0.10));
        }
    }
    vec![
        ("output-a".to_string(), a_records),
        ("output-b".to_string(), b_records),
    ]
}

// =============================================================================
// 1. OneHot descriptor emits at least one inhibitory output↔output candidate.
// =============================================================================
#[test]
fn one_hot_descriptor_emits_inhibitory_recommendation() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 2);

    let candidates = detect_output_competition(&creature, &records, &descriptor);

    assert!(
        !candidates.is_empty(),
        "OneHot + co-firing outputs must produce at least one lateral-inhibition candidate",
    );
    for c in &candidates {
        assert!(
            c.recommended_weight < 0.0,
            "lateral inhibition must use a negative weight (got {})",
            c.recommended_weight,
        );
        assert_ne!(
            c.from_output_uuid, c.to_output_uuid,
            "self-loops must not be proposed",
        );
    }
}

// =============================================================================
// 2. Simplex descriptor (CROSS_ENTROPY) — same behaviour.
// =============================================================================
#[test]
fn simplex_descriptor_emits_inhibitory_recommendation() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let descriptor = TaskDescriptor::from_name("CROSS_ENTROPY", 2);

    let candidates = detect_output_competition(&creature, &records, &descriptor);

    assert!(
        !candidates.is_empty(),
        "Simplex + co-firing outputs must produce a candidate",
    );
    assert!(candidates.iter().all(|c| c.recommended_weight < 0.0));
}

// =============================================================================
// 3. Neutral / Unknown descriptor — nothing emitted (regression guard).
// =============================================================================
#[test]
fn unknown_descriptor_emits_nothing() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let neutral = TaskDescriptor::neutral();

    let candidates = detect_output_competition(&creature, &records, &neutral);

    assert!(
        candidates.is_empty(),
        "Unknown descriptor must produce no candidates (got {})",
        candidates.len(),
    );
}

// =============================================================================
// 4. OTHER descriptor — nothing emitted (regression guard).
// =============================================================================
#[test]
fn other_descriptor_emits_nothing() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let other = TaskDescriptor::from_name("OTHER", 2);

    let candidates = detect_output_competition(&creature, &records, &other);

    assert!(candidates.is_empty(), "OTHER descriptor must emit nothing");
}

// =============================================================================
// 5. Independent descriptor (MSE) — nothing emitted (regression guard).
// =============================================================================
#[test]
fn independent_descriptor_emits_nothing() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let mse = TaskDescriptor::from_name("MSE", 2);

    let candidates = detect_output_competition(&creature, &records, &mse);

    assert!(
        candidates.is_empty(),
        "Independent (MSE) descriptor must emit nothing",
    );
}

// =============================================================================
// 6. Margin descriptor (HINGE) — nothing emitted (regression guard).
// =============================================================================
#[test]
fn margin_descriptor_emits_nothing() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let hinge = TaskDescriptor::from_name("HINGE", 2);

    let candidates = detect_output_competition(&creature, &records, &hinge);

    assert!(
        candidates.is_empty(),
        "Margin (HINGE) descriptor must emit nothing",
    );
}

// =============================================================================
// 7. Single-output network — nothing to compete with.
// =============================================================================
#[test]
fn single_output_emits_nothing() {
    let creature = creature_with_one_output();
    let records: Vec<(String, Vec<DiscoverRecord>)> = vec![(
        "output-a".to_string(),
        (0..40)
            .map(|i| make_record("output-a", i, 1.0, 0.85, 0.15))
            .collect(),
    )];
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 1);

    let candidates = detect_output_competition(&creature, &records, &descriptor);

    assert!(
        candidates.is_empty(),
        "Single-output network cannot exhibit output competition",
    );
}

// =============================================================================
// 8. Non-competing outputs (disjoint firing) — nothing emitted.
// =============================================================================
#[test]
fn non_competing_outputs_emit_nothing() {
    let creature = creature_with_two_outputs();
    let records = disjoint_firing_records();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 2);

    let candidates = detect_output_competition(&creature, &records, &descriptor);

    assert!(
        candidates.is_empty(),
        "Outputs that fire on disjoint samples are not competing — none expected, got {}",
        candidates.len(),
    );
}

// =============================================================================
// 9. Existing output→output synapse is not duplicated.
// =============================================================================
#[test]
fn existing_synapse_is_not_duplicated() {
    let mut creature = creature_with_two_outputs();
    // Pre-existing inhibitory synapse output-a → output-b — the recommender
    // must not propose another AddSynapse for the same pair.
    creature
        .synapses
        .push(synapse("output-a", "output-b", -0.2));

    let records = co_firing_records();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 2);

    let candidates = detect_output_competition(&creature, &records, &descriptor);

    for c in &candidates {
        assert!(
            !(c.from_output_uuid == "output-a" && c.to_output_uuid == "output-b"),
            "must not propose an AddSynapse for an existing (output-a → output-b) pair",
        );
    }
}

// =============================================================================
// 10. The coordinated-candidate converter emits AddSynapse operations with a
//     negative weight (inhibitory connection).
// =============================================================================
#[test]
fn coordinated_candidates_use_add_synapse_with_negative_weight() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 2);

    let candidates = detect_output_competition(&creature, &records, &descriptor);
    assert!(!candidates.is_empty(), "test prerequisite");

    let coordinated = output_competition_to_coordinated_candidates(&candidates);
    assert_eq!(
        coordinated.len(),
        candidates.len(),
        "each candidate maps to exactly one coordinated candidate",
    );
    for c in &coordinated {
        assert_eq!(c.operations.len(), 1, "single AddSynapse op per candidate");
        match &c.operations[0] {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => {
                assert_ne!(from_neuron_uuid, to_neuron_uuid);
                assert!(
                    *weight < 0.0,
                    "lateral inhibition must use a negative weight (got {weight})",
                );
            }
            other => panic!("expected AddSynapse, got {other:?}"),
        }
        assert!(
            c.expected_creature_score_gain >= 0.0,
            "expected gain must be non-negative",
        );
    }
}

// =============================================================================
// 11. Forward-only ordering: only proposals where `from` precedes `to` in
//     `creature.neurons[]` are emitted.
// =============================================================================
#[test]
fn forward_only_ordering_is_respected() {
    let creature = creature_with_two_outputs();
    let records = co_firing_records();
    let descriptor = TaskDescriptor::from_name("CATEGORICAL_ERROR", 2);

    let candidates = detect_output_competition(&creature, &records, &descriptor);

    let index_of = |uuid: &str| {
        creature
            .neurons
            .iter()
            .position(|n| n.uuid == uuid)
            .expect("uuid must exist in creature.neurons")
    };

    for c in &candidates {
        assert!(
            index_of(&c.from_output_uuid) < index_of(&c.to_output_uuid),
            "candidate {} → {} violates forward-only ordering",
            c.from_output_uuid,
            c.to_output_uuid,
        );
    }
}
