//! Regression tests for Issue #2359: detection converters give new hidden
//! neurons deterministic synthetic UUIDs (`hsc-…`, `ocs-…`, `bp-…`) but never
//! checked them against `creature.neurons`. A second discovery pass on a
//! creature that had already applied an earlier candidate therefore re-emitted
//! `AddNeuron` for a UUID the creature already owned.

use neat_ai_discovery::analysis::detection::bimodal_neuron::{
    BimodalNeuronCandidate, bimodal_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::bottleneck::{
    BottleneckNeuronCandidate, bottleneck_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::bounded_range::{
    BoundedRangeCandidate, bounded_range_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::hard_sample_cluster::{
    HardSampleCluster, hard_sample_clusters_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::output_conflict::{
    OutputConflictNeuron, output_conflicts_to_coordinated_candidates,
};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, CreatureJson, NeuronJson,
    SynapseJson,
};

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
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

/// Two inputs, hidden `h-conflict` and `h-up`, two outputs. Forward-only.
fn base_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            neuron("in-0", "input"),
            neuron("in-1", "input"),
            neuron("h-up", "hidden"),
            neuron("h-conflict", "hidden"),
            neuron("out-0", "output"),
            neuron("out-1", "output"),
        ],
        synapses: vec![
            synapse("in-0", "h-up", 0.5),
            synapse("in-1", "h-up", 0.4),
            synapse("h-up", "h-conflict", 0.8),
            synapse("h-conflict", "out-0", 2.0),
            synapse("h-conflict", "out-1", 1.0),
        ],
        input: 2,
        output: 2,
    }
}

fn clusters() -> Vec<HardSampleCluster> {
    vec![HardSampleCluster {
        hard_obs_indices: vec![1, 2, 3, 4, 5],
        mean_error: 0.8,
        easy_mean_error: 0.1,
        hard_to_easy_ratio: 8.0,
        dominant_input_uuids: vec!["in-0".to_string()],
        output_neuron_count: 2,
        estimated_improvement: 0.05,
    }]
}

/// Harms output 1 where there is no direct synapse conflict->output-1 once the
/// existing synapse is removed, so the converter emits a split `AddNeuron`.
fn conflicts() -> Vec<OutputConflictNeuron> {
    vec![OutputConflictNeuron {
        neuron_uuid: "h-conflict".to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
        per_output_mean_error: vec![-0.3, 0.5],
        conflict_severity: 0.15,
        sample_count: 50,
        estimated_improvement: 0.05,
    }]
}

/// The conflict fixture needs no direct h-conflict -> out-1 synapse.
fn conflict_creature() -> CreatureJson {
    let mut c = base_creature();
    c.synapses
        .retain(|s| !(s.from_uuid == "h-conflict" && s.to_uuid == "out-1"));
    c
}

fn bottlenecks() -> Vec<BottleneckNeuronCandidate> {
    vec![BottleneckNeuronCandidate {
        neuron_uuid: "h-conflict".to_string(),
        fan_in: 4,
        fan_out: 1,
        error_contribution_ratio: 0.6,
        bottleneck_score: 4.0,
        estimated_improvement: 0.05,
        recommended_actions: vec!["addParallelNeuron".to_string()],
        upstream_uuids: vec!["h-up".to_string()],
        downstream_uuids: vec!["out-0".to_string()],
    }]
}

fn add_neuron_uuids(cands: &[CoordinatedStructuralCandidateJson]) -> Vec<String> {
    cands
        .iter()
        .flat_map(|c| c.operations.iter())
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddNeuron { neuron_uuid, .. } => Some(neuron_uuid.clone()),
            _ => None,
        })
        .collect()
}

/// Apply the candidates' `AddNeuron` / `AddSynapse` ops as an applied candidate
/// would, inserting each new neuron before its anchor to keep forward order.
fn apply(creature: &mut CreatureJson, cands: &[CoordinatedStructuralCandidateJson]) {
    for op in cands.iter().flat_map(|c| c.operations.iter()) {
        match op {
            CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid,
                neuron_type,
                squash,
                bias,
                insert_before_neuron_uuid,
            } => {
                if creature.neurons.iter().any(|n| &n.uuid == neuron_uuid) {
                    continue;
                }
                let new = NeuronJson {
                    uuid: neuron_uuid.clone(),
                    neuron_type: neuron_type.clone(),
                    squash: squash.clone(),
                    bias: *bias,
                };
                let pos = insert_before_neuron_uuid
                    .as_ref()
                    .and_then(|a| creature.neurons.iter().position(|n| &n.uuid == a))
                    .or_else(|| {
                        creature
                            .neurons
                            .iter()
                            .position(|n| n.neuron_type == "output")
                    })
                    .unwrap_or(creature.neurons.len());
                creature.neurons.insert(pos, new);
            }
            CoordinatedStructuralOpJson::AddSynapse {
                from_neuron_uuid,
                to_neuron_uuid,
                weight,
            } => creature
                .synapses
                .push(synapse(from_neuron_uuid, to_neuron_uuid, *weight)),
            _ => {}
        }
    }
}

fn hsc_and_ocs(creature: &CreatureJson) -> Vec<CoordinatedStructuralCandidateJson> {
    let mut out = hard_sample_clusters_to_coordinated_candidates(&clusters(), creature);
    out.extend(output_conflicts_to_coordinated_candidates(
        &conflicts(),
        creature,
        &None,
    ));
    out
}

#[test]
fn emitted_add_neuron_never_reuses_an_existing_neuron_uuid() {
    let mut creature = conflict_creature();
    let first = hsc_and_ocs(&creature);
    let first_uuids = add_neuron_uuids(&first);
    assert!(
        first_uuids.iter().any(|u| u.starts_with("hsc-"))
            && first_uuids.iter().any(|u| u.starts_with("ocs-")),
        "first pass must emit both kinds of synthetic neuron, got {first_uuids:?}"
    );
    apply(&mut creature, &first);

    let second = hsc_and_ocs(&creature);
    let second_uuids = add_neuron_uuids(&second);
    assert!(
        !second_uuids.is_empty(),
        "second pass must still emit an AddNeuron (otherwise the test is vacuous)"
    );
    for uuid in &second_uuids {
        assert!(
            !creature.neurons.iter().any(|n| &n.uuid == uuid),
            "AddNeuron re-emitted existing neuron uuid {uuid}"
        );
    }
}

#[test]
fn bottleneck_parallel_neuron_never_reuses_an_existing_neuron_uuid() {
    let mut creature = base_creature();
    let first = bottleneck_neurons_to_coordinated_candidates(&bottlenecks(), &creature, None);
    let first_uuids = add_neuron_uuids(&first);
    assert!(!first_uuids.is_empty(), "first pass must emit AddNeuron");
    apply(&mut creature, &first);

    let second = bottleneck_neurons_to_coordinated_candidates(&bottlenecks(), &creature, None);
    let second_uuids = add_neuron_uuids(&second);
    assert!(
        !second_uuids.is_empty(),
        "second pass must still emit an AddNeuron (otherwise the test is vacuous)"
    );
    for uuid in &second_uuids {
        assert!(
            !creature.neurons.iter().any(|n| &n.uuid == uuid),
            "AddNeuron re-emitted existing neuron uuid {uuid}"
        );
    }
}

#[test]
fn synthetic_uuid_unchanged_when_no_collision() {
    let creature = conflict_creature();
    let a = add_neuron_uuids(&hsc_and_ocs(&creature));
    let b = add_neuron_uuids(&hsc_and_ocs(&creature));
    assert!(!a.is_empty());
    assert_eq!(a, b);

    let c = add_neuron_uuids(&bottleneck_neurons_to_coordinated_candidates(
        &bottlenecks(),
        &creature,
        None,
    ));
    let d = add_neuron_uuids(&bottleneck_neurons_to_coordinated_candidates(
        &bottlenecks(),
        &creature,
        None,
    ));
    assert!(!c.is_empty());
    assert_eq!(c, d);
}

fn bimodal_candidates() -> Vec<BimodalNeuronCandidate> {
    vec![BimodalNeuronCandidate {
        neuron_uuid: "h-conflict".to_string(),
        current_squash: "TANH".to_string(),
        current_bias: 0.0,
        lower_mode_mean: -3.0,
        upper_mode_mean: 3.0,
        lower_mode_count: 50,
        upper_mode_count: 50,
        bimodality_score: 2.0,
        sample_count: 100,
        estimated_improvement: 0.05,
    }]
}

fn bounded_candidates() -> Vec<BoundedRangeCandidate> {
    vec![BoundedRangeCandidate {
        neuron_uuid: "h-conflict".to_string(),
        boundary_value: -1.0,
        boundary_fraction: 0.4,
        useful_range_min: 0.1,
        useful_range_max: 0.8,
        sample_count: 100,
        detection_confidence: 0.85,
        estimated_improvement: 0.01,
    }]
}

#[test]
fn bimodal_split_neuron_never_reuses_an_existing_neuron_uuid() {
    let mut creature = base_creature();
    creature
        .neurons
        .push(neuron("h-conflict-split-lower", "hidden"));
    let out = bimodal_neurons_to_coordinated_candidates(&bimodal_candidates(), &creature);
    assert_eq!(out.len(), 1, "candidate must still be emitted");
    let uuids = add_neuron_uuids(&out);
    assert_eq!(uuids.len(), 1);
    assert!(
        !creature.neurons.iter().any(|n| n.uuid == uuids[0]),
        "AddNeuron re-emitted existing neuron uuid {}",
        uuids[0]
    );
}

#[test]
fn bounded_range_gate_neuron_never_reuses_an_existing_neuron_uuid() {
    let mut creature = base_creature();
    creature
        .neurons
        .push(neuron("gate-bounded-h-conflict", "hidden"));
    let out = bounded_range_to_coordinated_candidates(&bounded_candidates(), &creature);
    assert_eq!(out.len(), 1, "candidate must still be emitted");
    let uuids = add_neuron_uuids(&out);
    assert_eq!(uuids.len(), 1);
    assert!(
        !creature.neurons.iter().any(|n| n.uuid == uuids[0]),
        "AddNeuron re-emitted existing neuron uuid {}",
        uuids[0]
    );
    let synapse_targets: Vec<&String> = out[0]
        .operations
        .iter()
        .filter_map(|op| match op {
            CoordinatedStructuralOpJson::AddSynapse { to_neuron_uuid, .. } => Some(to_neuron_uuid),
            _ => None,
        })
        .collect();
    assert_eq!(synapse_targets, vec![&uuids[0]]);
}

#[test]
fn legacy_uuid_kept_when_no_collision() {
    let creature = base_creature();
    let bi = add_neuron_uuids(&bimodal_neurons_to_coordinated_candidates(
        &bimodal_candidates(),
        &creature,
    ));
    assert_eq!(bi, vec!["h-conflict-split-lower".to_string()]);
    let bo = add_neuron_uuids(&bounded_range_to_coordinated_candidates(
        &bounded_candidates(),
        &creature,
    ));
    assert_eq!(bo, vec!["gate-bounded-h-conflict".to_string()]);
}
