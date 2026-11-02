//! Regression coverage for synthetic detection neuron UUID collisions (Issue #2412).

use neat_ai_discovery::analysis::detection::correlated_error::{
    CorrelatedErrorGroup, correlated_errors_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::fanin_polarity_conflict::{
    FaninPolarityConflictCandidate, fanin_polarity_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::input_sensitivity::{
    ThresholdEffectCandidate, threshold_effects_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::monotonicity::{
    MonotonicityCandidate, non_monotonic_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::sentinel_gating::{
    SentinelGatingCandidate, sentinel_gating_to_coordinated_candidates,
};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};

fn neuron(uuid: &str, kind: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
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

fn creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in", "input"),
            neuron("source", "input"),
            neuron("target", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in", "target", -2.0),
            synapse("source", "target", 2.0),
            synapse("target", "out", 1.0),
        ],
        input: 2,
        output: 1,
    }
}

fn add_neuron_uuids(candidates: &[CoordinatedStructuralCandidateJson]) -> Vec<String> {
    candidates
        .iter()
        .flat_map(|candidate| &candidate.operations)
        .filter_map(|operation| match operation {
            CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => Some(neuron_uuid.clone()),
            _ => None,
        })
        .collect()
}

fn add_neuron_uuid(candidate: &CoordinatedStructuralCandidateJson) -> &str {
    candidate
        .operations
        .iter()
        .find_map(|operation| match operation {
            CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => {
                Some(neuron_uuid.as_str())
            }
            _ => None,
        })
        .expect("candidate adds a neuron")
}

fn assert_synapses_use_uuid(candidate: &CoordinatedStructuralCandidateJson) {
    let uuid = add_neuron_uuid(candidate);
    let add_synapses: Vec<_> = candidate
        .operations
        .iter()
        .filter_map(|operation| match operation {
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            } => Some((from_neuron_uuid, to_neuron_uuid)),
            _ => None,
        })
        .collect();
    if !add_synapses.is_empty() {
        assert!(
            add_synapses
                .iter()
                .all(|(from, to)| from.as_str() == uuid || to.as_str() == uuid)
        );
    }
}

fn sentinel_candidate() -> SentinelGatingCandidate {
    SentinelGatingCandidate {
        neuron_uuid: "in".to_string(),
        sentinel_value: -1.0,
        sentinel_fraction: 0.4,
        useful_range_min: 0.2,
        useful_range_max: 1.0,
        sample_count: 100,
        estimated_improvement: 0.1,
    }
}

fn fanin_candidate() -> FaninPolarityConflictCandidate {
    FaninPolarityConflictCandidate {
        neuron_uuid: "target".to_string(),
        positive_weight_sum: 2.0,
        negative_weight_sum: 2.0,
        positive_count: 1,
        negative_count: 1,
        conflict_score: 1.0,
        sample_count: 100,
        estimated_improvement: 0.1,
    }
}

fn monotonicity_candidate() -> MonotonicityCandidate {
    MonotonicityCandidate {
        neuron_uuid: "target".to_string(),
        monotonicity_score: 0.0,
        sample_count: 100,
        estimated_improvement: 0.1,
    }
}

fn correlated_error_group() -> CorrelatedErrorGroup {
    CorrelatedErrorGroup {
        output_neuron_uuids: vec!["out".to_string(), "out-2".to_string()],
        mean_correlation: 1.0,
        shared_error_sample_count: 10,
        total_sample_count: 10,
        predictive_input_uuids: vec!["in".to_string()],
        estimated_improvement: 0.1,
    }
}

fn threshold_candidate() -> ThresholdEffectCandidate {
    ThresholdEffectCandidate {
        input_neuron_uuid: "in".to_string(),
        intermediate_neuron_uuid: Some("target".to_string()),
        target_neuron_uuid: "out".to_string(),
        gradient_magnitude: 20.0,
        threshold_proximity: 0.1,
        recommended_action: "addNeuron".to_string(),
        sample_count: 20,
        estimated_improvement: 0.1,
    }
}

fn assert_two_pass<F>(mut topology: CreatureJson, convert: F)
where
    F: Fn(&CreatureJson) -> Vec<CoordinatedStructuralCandidateJson>,
{
    let first = convert(&topology);
    let first_uuids = add_neuron_uuids(&first);
    assert!(!first_uuids.is_empty(), "first pass must add a neuron");
    for candidate in &first {
        if !add_neuron_uuids(std::slice::from_ref(candidate)).is_empty() {
            assert_synapses_use_uuid(candidate);
        }
    }
    for uuid in first_uuids {
        topology.neurons.push(neuron(&uuid, "hidden"));
    }

    let second = convert(&topology);
    let second_uuids = add_neuron_uuids(&second);
    assert!(
        !second_uuids.is_empty(),
        "second pass must remain non-empty"
    );
    for uuid in &second_uuids {
        assert!(
            !topology
                .neurons
                .iter()
                .any(|existing| existing.uuid == *uuid),
            "converter re-emitted occupied UUID {uuid}"
        );
    }
    for candidate in &second {
        if !add_neuron_uuids(std::slice::from_ref(candidate)).is_empty() {
            assert_synapses_use_uuid(candidate);
        }
    }
}

#[test]
fn sentinel_gating_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(creature(), |topology| {
        sentinel_gating_to_coordinated_candidates(&[sentinel_candidate()], topology)
    });
}

#[test]
fn fanin_polarity_conflict_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(creature(), |topology| {
        fanin_polarity_conflicts_to_coordinated_candidates(&[fanin_candidate()], topology)
    });
}

#[test]
fn monotonicity_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(creature(), |topology| {
        non_monotonic_neurons_to_coordinated_candidates(&[monotonicity_candidate()], topology)
    });
}

#[test]
fn correlated_error_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(creature(), |topology| {
        correlated_errors_to_coordinated_candidates(&[correlated_error_group()], topology)
    });
}

#[test]
fn threshold_effect_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(creature(), |topology| {
        threshold_effects_to_coordinated_candidates(&[threshold_candidate()], topology)
    });
}

fn assert_skips_occupied_salts<F>(mut topology: CreatureJson, legacy_uuid: &str, convert: F)
where
    F: Fn(&CreatureJson) -> Vec<CoordinatedStructuralCandidateJson>,
{
    for uuid in [
        legacy_uuid.to_string(),
        format!("{legacy_uuid}-1"),
        format!("{legacy_uuid}-2"),
    ] {
        topology.neurons.push(neuron(&uuid, "hidden"));
    }
    let first = convert(&topology);
    let uuids = add_neuron_uuids(&first);
    assert!(
        !uuids.is_empty(),
        "occupied legacy salts must not suppress candidates"
    );
    assert!(
        uuids
            .iter()
            .all(|uuid| !topology.neurons.iter().any(|n| n.uuid == *uuid))
    );
    assert_eq!(
        uuids,
        add_neuron_uuids(&convert(&topology)),
        "resalting must be deterministic"
    );
    for candidate in &first {
        assert_synapses_use_uuid(candidate);
    }
}

#[test]
fn caller_supplied_collisions_skip_multiple_salts_for_each_converter() {
    assert_skips_occupied_salts(creature(), "gate-sentinel-in", |topology| {
        sentinel_gating_to_coordinated_candidates(&[sentinel_candidate()], topology)
    });
    assert_skips_occupied_salts(creature(), "fanin-split-target", |topology| {
        fanin_polarity_conflicts_to_coordinated_candidates(&[fanin_candidate()], topology)
    });
    assert_skips_occupied_salts(creature(), "split-target", |topology| {
        non_monotonic_neurons_to_coordinated_candidates(&[monotonicity_candidate()], topology)
    });
    assert_skips_occupied_salts(creature(), "dampening-in-out", |topology| {
        threshold_effects_to_coordinated_candidates(&[threshold_candidate()], topology)
    });
    assert_skips_occupied_salts(creature(), "cs-15f1c91d4170b2e7", |topology| {
        correlated_errors_to_coordinated_candidates(&[correlated_error_group()], topology)
    });
}

#[test]
fn legacy_uuid_is_preserved_and_repeatable_when_free() {
    let topology = creature();
    let cases: Vec<(Vec<String>, &str)> = vec![
        (
            add_neuron_uuids(&sentinel_gating_to_coordinated_candidates(
                &[sentinel_candidate()],
                &topology,
            )),
            "gate-sentinel-in",
        ),
        (
            add_neuron_uuids(&fanin_polarity_conflicts_to_coordinated_candidates(
                &[fanin_candidate()],
                &topology,
            )),
            "fanin-split-target",
        ),
        (
            add_neuron_uuids(&non_monotonic_neurons_to_coordinated_candidates(
                &[monotonicity_candidate()],
                &topology,
            )),
            "split-target",
        ),
        (
            add_neuron_uuids(&threshold_effects_to_coordinated_candidates(
                &[threshold_candidate()],
                &topology,
            )),
            "dampening-in-out",
        ),
        (
            add_neuron_uuids(&correlated_errors_to_coordinated_candidates(
                &[correlated_error_group()],
                &topology,
            )),
            "cs-15f1c91d4170b2e7",
        ),
    ];
    for (uuids, expected) in cases {
        assert_eq!(uuids, [expected]);
    }
}
