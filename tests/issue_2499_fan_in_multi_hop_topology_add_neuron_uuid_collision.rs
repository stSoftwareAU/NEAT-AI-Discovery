//! Regression coverage for synthetic neuron UUID collisions in the fan-in,
//! multi-hop and topology-diversification converters (Issue #2499).

use neat_ai_discovery::analysis::detection::topology_diversification::{
    TopologyDiversificationCandidate, topology_diversification_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::recommendation::fan_in::{
    FanInCandidate, fan_in_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::recommendation::multi_hop::{
    MultiHopCandidate, multi_hop_to_coordinated_candidates,
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

/// Inserts a hidden neuron before the last neuron. Every fixture ends with its
/// output neuron, so this keeps the input, hidden..., output ordering.
fn insert_hidden(creature: &mut CreatureJson, uuid: &str) {
    let at = creature.neurons.len() - 1;
    creature.neurons.insert(at, neuron(uuid, "hidden"));
}

fn fan_in_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-a", "input"),
            neuron("in-b", "input"),
            neuron("out", "output"),
        ],
        synapses: vec![synapse("in-a", "out", 0.5), synapse("in-b", "out", -0.5)],
        input: 2,
        output: 1,
    }
}

fn fan_in_candidate() -> FanInCandidate {
    FanInCandidate {
        input_uuids: vec!["in-a".to_string(), "in-b".to_string()],
        target_uuid: "out".to_string(),
        input_weights: vec![0.6, -0.4],
        output_weight: 0.1,
        activation: "TANH".to_string(),
        estimated_improvement: 0.1,
        mean_input_error_correlation: 0.5,
        input_mutual_correlation: 0.2,
        sample_count: 10,
        reason: "issue 2499 fan-in fixture".to_string(),
    }
}

fn multi_hop_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in", "input"),
            neuron("hidden", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![synapse("in", "hidden", 0.5), synapse("hidden", "out", 0.5)],
        input: 1,
        output: 1,
    }
}

fn multi_hop_candidate() -> MultiHopCandidate {
    MultiHopCandidate {
        path: vec!["in".to_string(), "hidden".to_string(), "out".to_string()],
        estimated_improvement: 0.1,
        correlation_strength: 0.5,
    }
}

fn topology_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![neuron("in", "input"), neuron("out", "output")],
        synapses: vec![synapse("in", "out", 0.5)],
        input: 1,
        output: 1,
    }
}

fn topology_candidate() -> TopologyDiversificationCandidate {
    TopologyDiversificationCandidate {
        output_neuron_uuid: "out".to_string(),
        source_input_uuid: "in".to_string(),
        max_hidden_depth: 0,
        direct_path_count: 1,
        mean_output_error: 0.5,
        estimated_improvement: 0.1,
    }
}

fn convert_fan_in(creature: &CreatureJson) -> Vec<CoordinatedStructuralCandidateJson> {
    fan_in_to_coordinated_candidates(&[fan_in_candidate()], creature)
}

fn convert_multi_hop(creature: &CreatureJson) -> Vec<CoordinatedStructuralCandidateJson> {
    multi_hop_to_coordinated_candidates(&[multi_hop_candidate()], creature)
}

fn convert_topology(creature: &CreatureJson) -> Vec<CoordinatedStructuralCandidateJson> {
    topology_diversification_to_coordinated_candidates(&[topology_candidate()], creature)
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

fn assert_synapses_use_uuid(candidate: &CoordinatedStructuralCandidateJson, uuid: &str) {
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
    assert!(
        !add_synapses.is_empty(),
        "candidate must wire the new neuron"
    );
    assert!(
        add_synapses
            .iter()
            .all(|(from, to)| from.as_str() == uuid || to.as_str() == uuid)
    );
}

/// (a) Two passes: the second pass must not re-emit a UUID the creature already has.
fn assert_two_pass<F>(mut topology: CreatureJson, convert: F)
where
    F: Fn(&CreatureJson) -> Vec<CoordinatedStructuralCandidateJson>,
{
    let first = convert(&topology);
    let first_uuids = add_neuron_uuids(&first);
    assert!(!first_uuids.is_empty(), "first pass must add a neuron");
    for candidate in &first {
        for uuid in add_neuron_uuids(std::slice::from_ref(candidate)) {
            assert_synapses_use_uuid(candidate, &uuid);
        }
    }
    for uuid in &first_uuids {
        insert_hidden(&mut topology, uuid);
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
        for uuid in add_neuron_uuids(std::slice::from_ref(candidate)) {
            assert_synapses_use_uuid(candidate, &uuid);
        }
    }
}

/// (b) Caller-supplied collision: `legacy`, `legacy-1` and `legacy-2` are occupied,
/// so the converter must land on `legacy-3`, deterministically.
fn assert_skips_occupied_salts<F>(mut topology: CreatureJson, convert: F)
where
    F: Fn(&CreatureJson) -> Vec<CoordinatedStructuralCandidateJson>,
{
    let probe = convert(&topology);
    let legacy = add_neuron_uuids(&probe)
        .into_iter()
        .next()
        .expect("unmodified creature must yield a candidate");
    for uuid in [legacy.clone(), format!("{legacy}-1"), format!("{legacy}-2")] {
        insert_hidden(&mut topology, &uuid);
    }

    let first = convert(&topology);
    let uuids = add_neuron_uuids(&first);
    assert!(
        !uuids.is_empty(),
        "occupied legacy salts must not suppress candidates"
    );
    for uuid in &uuids {
        assert!(
            !topology.neurons.iter().any(|n| n.uuid == *uuid),
            "emitted UUID {uuid} names an existing neuron"
        );
    }
    assert_eq!(uuids, [format!("{legacy}-3")]);
    assert_eq!(
        uuids,
        add_neuron_uuids(&convert(&topology)),
        "resalting must be deterministic"
    );
    for candidate in &first {
        for uuid in add_neuron_uuids(std::slice::from_ref(candidate)) {
            assert_synapses_use_uuid(candidate, &uuid);
        }
    }
}

/// (c) The legacy UUID is kept, unsalted, and repeatable when it is free.
fn assert_legacy_preserved<F>(topology: &CreatureJson, prefix: &str, convert: F)
where
    F: Fn(&CreatureJson) -> Vec<CoordinatedStructuralCandidateJson>,
{
    let first = add_neuron_uuids(&convert(topology));
    assert_eq!(first.len(), 1, "expected exactly one AddNeuron");
    let uuid = &first[0];
    assert!(
        uuid.starts_with(prefix),
        "legacy UUID {uuid} must start with {prefix}"
    );
    assert_eq!(
        uuid.len(),
        prefix.len() + 16,
        "legacy UUID {uuid} must carry no salt suffix"
    );
    assert_eq!(first, add_neuron_uuids(&convert(topology)));
}

#[test]
fn fan_in_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(fan_in_creature(), convert_fan_in);
}

#[test]
fn multi_hop_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(multi_hop_creature(), convert_multi_hop);
}

#[test]
fn topology_diversification_uses_a_free_uuid_on_each_pass() {
    assert_two_pass(topology_creature(), convert_topology);
}

#[test]
fn fan_in_caller_supplied_collisions_skip_multiple_salts() {
    assert_skips_occupied_salts(fan_in_creature(), convert_fan_in);
}

#[test]
fn multi_hop_caller_supplied_collisions_skip_multiple_salts() {
    assert_skips_occupied_salts(multi_hop_creature(), convert_multi_hop);
}

#[test]
fn topology_diversification_caller_supplied_collisions_skip_multiple_salts() {
    assert_skips_occupied_salts(topology_creature(), convert_topology);
}

#[test]
fn fan_in_legacy_uuid_is_preserved_when_free() {
    assert_legacy_preserved(&fan_in_creature(), "fan-in-", convert_fan_in);
}

#[test]
fn multi_hop_legacy_uuid_is_preserved_when_free() {
    assert_legacy_preserved(&multi_hop_creature(), "mh-", convert_multi_hop);
}

#[test]
fn topology_diversification_legacy_uuid_is_preserved_when_free() {
    assert_legacy_preserved(&topology_creature(), "td-", convert_topology);
}

#[test]
fn topology_diversification_skips_candidate_already_applied() {
    let mut topology = topology_creature();
    let legacy = add_neuron_uuids(&convert_topology(&topology))
        .into_iter()
        .next()
        .expect("unmodified creature must yield a candidate");

    // The legacy neuron exists and already feeds the output: the candidate was applied.
    insert_hidden(&mut topology, &legacy);
    topology.synapses.push(synapse(&legacy, "out", 0.1));

    let result = convert_topology(&topology);
    assert!(
        add_neuron_uuids(&result).is_empty(),
        "already-applied candidate must not be re-emitted under a new salt"
    );
    assert!(result.is_empty());
}
