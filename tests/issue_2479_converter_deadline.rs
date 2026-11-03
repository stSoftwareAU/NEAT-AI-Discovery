//! Issue #2479: public-API coverage for the per-candidate deadline checks in the
//! seven coordinated-candidate converters and the squash-weight rescale detector.
//!
//! For each entry point this file asserts, through the public API only:
//!
//! 1. with no deadline, the fixture yields its full candidate set (positive control);
//! 2. an already-elapsed deadline stops the loop before the first candidate, so the
//!    result is empty;
//! 3. the legacy function without a deadline matches `_with_deadline(.., &None)`.

use std::time::SystemTime;

use neat_ai_discovery::analysis::detection::bottleneck::{
    BottleneckNeuronCandidate, bottleneck_neurons_to_coordinated_candidates,
    bottleneck_neurons_to_coordinated_candidates_with_deadline,
};
use neat_ai_discovery::analysis::detection::monotonicity::{
    MonotonicityCandidate, non_monotonic_neurons_to_coordinated_candidates,
    non_monotonic_neurons_to_coordinated_candidates_with_deadline,
};
use neat_ai_discovery::analysis::detection::operating_point::{
    OperatingPointIssue, operating_point_to_coordinated_candidates,
    operating_point_to_coordinated_candidates_with_deadline,
};
use neat_ai_discovery::analysis::detection::output_range_compression::{
    OutputRangeCompressionNeuron, output_range_compression_to_coordinated_candidates,
    output_range_compression_to_coordinated_candidates_with_deadline,
};
use neat_ai_discovery::analysis::detection::restricted_range::{
    RestrictedRangeNeuron, restricted_range_to_coordinated_candidates,
    restricted_range_to_coordinated_candidates_with_deadline,
};
use neat_ai_discovery::analysis::detection::sentinel_gating::{
    SentinelGatingCandidate, sentinel_gating_to_coordinated_candidates,
    sentinel_gating_to_coordinated_candidates_with_deadline,
};
use neat_ai_discovery::analysis::detection::squash_weight_rescale::{
    detect_squash_weight_rescale_candidates, detect_squash_weight_rescale_candidates_with_deadline,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CreatureJson, NeuronJson, SynapseJson,
};

/// Builds a neuron with a zero bias.
fn neuron_json(uuid: &str, neuron_type: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Builds a synapse with no branch classification.
fn synapse_json(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Serialises coordinated candidates so legacy and deadline outputs can be compared.
fn to_json(candidates: &[CoordinatedStructuralCandidateJson]) -> String {
    serde_json::to_string(candidates).expect("coordinated candidates serialise")
}

/// Two input neurons plus one `neuron_type` neuron per uuid in `targets`, each fed
/// by both inputs. `output` is the creature's declared output count.
fn fan_in_creature(
    targets: &[String],
    neuron_type: &str,
    squash: &str,
    output: usize,
) -> CreatureJson {
    let mut neurons = vec![
        neuron_json("input-a", "input", "IDENTITY"),
        neuron_json("input-b", "input", "IDENTITY"),
    ];
    let mut synapses = Vec::new();
    for uuid in targets {
        neurons.push(neuron_json(uuid, neuron_type, squash));
        for from in ["input-a", "input-b"] {
            synapses.push(synapse_json(from, uuid, 1.0));
        }
    }
    CreatureJson {
        neurons,
        synapses,
        input: 2,
        output,
    }
}

fn three_uuids(prefix: &str) -> [String; 3] {
    ["alpha", "beta", "gamma"].map(|suffix| format!("{prefix}-{suffix}"))
}

// ---------------------------------------------------------------------------
// bottleneck_neurons_to_coordinated_candidates_with_deadline
// ---------------------------------------------------------------------------

/// `count` hidden neurons `h{i}`, each fed by `in` and feeding `out`. Every hidden
/// neuron is a parallel-neuron candidate, so each yields one coordinated candidate.
fn bottleneck_fixture(count: usize) -> (CreatureJson, Vec<BottleneckNeuronCandidate>) {
    let mut neurons = vec![neuron_json("in", "input", "IDENTITY")];
    let mut synapses = Vec::new();
    let mut candidates = Vec::new();
    for i in 0..count {
        let uuid = format!("h{i}");
        neurons.push(neuron_json(&uuid, "hidden", "RELU"));
        synapses.push(synapse_json("in", &uuid, 0.5));
        synapses.push(synapse_json(&uuid, "out", 1.0));
        candidates.push(BottleneckNeuronCandidate {
            neuron_uuid: uuid,
            fan_in: 1,
            fan_out: 1,
            error_contribution_ratio: 0.0,
            bottleneck_score: 0.1,
            estimated_improvement: 0.001,
            recommended_actions: vec!["addParallelNeuron".to_string()],
            upstream_uuids: vec!["in".to_string()],
            downstream_uuids: vec!["out".to_string()],
        });
    }
    neurons.push(neuron_json("out", "output", "IDENTITY"));
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    };
    (creature, candidates)
}

#[test]
fn bottleneck_conversion_builds_every_candidate_without_deadline() {
    let (creature, candidates) = bottleneck_fixture(3);

    let full = bottleneck_neurons_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        None,
        &None,
    );

    assert_eq!(
        full.len(),
        3,
        "fixture must yield one candidate per bottleneck"
    );
}

#[test]
fn bottleneck_conversion_stops_at_an_elapsed_deadline() {
    let (creature, candidates) = bottleneck_fixture(3);

    let partial = bottleneck_neurons_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        None,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first candidate, got {} candidates",
        partial.len()
    );
}

#[test]
fn bottleneck_legacy_conversion_matches_deadline_variant_without_deadline() {
    let (creature, candidates) = bottleneck_fixture(3);

    let legacy = bottleneck_neurons_to_coordinated_candidates(&candidates, &creature, None);
    let with_none = bottleneck_neurons_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        None,
        &None,
    );

    assert_eq!(to_json(&legacy), to_json(&with_none));
}

// ---------------------------------------------------------------------------
// non_monotonic_neurons_to_coordinated_candidates_with_deadline
// ---------------------------------------------------------------------------

/// `count` hidden neurons `h{i}`, each fed by `in` and feeding `out`. Every hidden
/// neuron has incoming and outgoing synapses, so each yields an `AddNeuron` candidate.
fn monotonicity_fixture(count: usize) -> (CreatureJson, Vec<MonotonicityCandidate>) {
    let mut neurons = vec![neuron_json("in", "input", "IDENTITY")];
    let mut synapses = Vec::new();
    let mut candidates = Vec::new();
    for i in 0..count {
        let uuid = format!("h{i}");
        neurons.push(neuron_json(&uuid, "hidden", "RELU"));
        synapses.push(synapse_json("in", &uuid, 0.5));
        synapses.push(synapse_json(&uuid, "out", 1.0));
        candidates.push(MonotonicityCandidate {
            neuron_uuid: uuid,
            monotonicity_score: 0.0,
            sample_count: 30,
            estimated_improvement: 0.001,
        });
    }
    neurons.push(neuron_json("out", "output", "IDENTITY"));
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    };
    (creature, candidates)
}

#[test]
fn monotonicity_conversion_builds_every_candidate_without_deadline() {
    let (creature, candidates) = monotonicity_fixture(3);

    let full = non_monotonic_neurons_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        &None,
    );

    assert_eq!(
        full.len(),
        3,
        "fixture must yield one candidate per non-monotonic neuron"
    );
}

#[test]
fn monotonicity_conversion_stops_at_an_elapsed_deadline() {
    let (creature, candidates) = monotonicity_fixture(3);

    let partial = non_monotonic_neurons_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first candidate, got {} candidates",
        partial.len()
    );
}

#[test]
fn monotonicity_legacy_conversion_matches_deadline_variant_without_deadline() {
    let (creature, candidates) = monotonicity_fixture(3);

    let legacy = non_monotonic_neurons_to_coordinated_candidates(&candidates, &creature);
    let with_none = non_monotonic_neurons_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        &None,
    );

    assert_eq!(to_json(&legacy), to_json(&with_none));
}

// ---------------------------------------------------------------------------
// operating_point_to_coordinated_candidates_with_deadline
// ---------------------------------------------------------------------------

fn operating_point_issue(uuid: &str) -> OperatingPointIssue {
    OperatingPointIssue {
        neuron_uuid: uuid.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
        value_min: -0.5,
        value_max: 0.5,
        active_zone_min: -2.0,
        active_zone_max: 2.0,
        dynamic_range_utilisation: 0.1,
        sample_count: 30,
    }
}

/// Three issues, each yielding three coordinated candidates: nine in total.
fn operating_point_fixture() -> (CreatureJson, Vec<OperatingPointIssue>) {
    let uuids = three_uuids("hidden");
    let creature = fan_in_creature(&uuids, "hidden", "TANH", 1);
    let detected = uuids.iter().map(|u| operating_point_issue(u)).collect();
    (creature, detected)
}

#[test]
fn operating_point_conversion_builds_every_candidate_without_deadline() {
    let (creature, detected) = operating_point_fixture();

    let full = operating_point_to_coordinated_candidates_with_deadline(&detected, &creature, &None);

    assert_eq!(
        full.len(),
        9,
        "fixture must yield three candidates per issue"
    );
}

#[test]
fn operating_point_conversion_stops_at_an_elapsed_deadline() {
    let (creature, detected) = operating_point_fixture();

    let partial = operating_point_to_coordinated_candidates_with_deadline(
        &detected,
        &creature,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first issue, got {} candidates",
        partial.len()
    );
}

#[test]
fn operating_point_legacy_conversion_matches_deadline_variant_without_deadline() {
    let (creature, detected) = operating_point_fixture();

    let legacy = operating_point_to_coordinated_candidates(&detected, &creature);
    let with_none =
        operating_point_to_coordinated_candidates_with_deadline(&detected, &creature, &None);

    assert_eq!(to_json(&legacy), to_json(&with_none));
}

// ---------------------------------------------------------------------------
// restricted_range_to_coordinated_candidates_with_deadline
// ---------------------------------------------------------------------------

fn restricted_range_neuron(uuid: &str) -> RestrictedRangeNeuron {
    RestrictedRangeNeuron {
        neuron_uuid: uuid.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
        activation_min: -0.2,
        activation_max: 0.2,
        theoretical_range: 2.0,
        range_utilisation: 0.1,
        sample_count: 30,
    }
}

/// Three neurons, each yielding three coordinated candidates: nine in total.
fn restricted_range_fixture() -> (CreatureJson, Vec<RestrictedRangeNeuron>) {
    let uuids = three_uuids("hidden");
    let creature = fan_in_creature(&uuids, "hidden", "TANH", 1);
    let detected = uuids.iter().map(|u| restricted_range_neuron(u)).collect();
    (creature, detected)
}

#[test]
fn restricted_range_conversion_builds_every_candidate_without_deadline() {
    let (creature, detected) = restricted_range_fixture();

    let full =
        restricted_range_to_coordinated_candidates_with_deadline(&detected, &creature, &None);

    assert_eq!(
        full.len(),
        9,
        "fixture must yield three candidates per neuron"
    );
}

#[test]
fn restricted_range_conversion_stops_at_an_elapsed_deadline() {
    let (creature, detected) = restricted_range_fixture();

    let partial = restricted_range_to_coordinated_candidates_with_deadline(
        &detected,
        &creature,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first neuron, got {} candidates",
        partial.len()
    );
}

#[test]
fn restricted_range_legacy_conversion_matches_deadline_variant_without_deadline() {
    let (creature, detected) = restricted_range_fixture();

    let legacy = restricted_range_to_coordinated_candidates(&detected, &creature);
    let with_none =
        restricted_range_to_coordinated_candidates_with_deadline(&detected, &creature, &None);

    assert_eq!(to_json(&legacy), to_json(&with_none));
}

// ---------------------------------------------------------------------------
// output_range_compression_to_coordinated_candidates_with_deadline
// ---------------------------------------------------------------------------

fn output_range_compression_neuron(uuid: &str) -> OutputRangeCompressionNeuron {
    OutputRangeCompressionNeuron {
        neuron_uuid: uuid.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
        activation_min: -0.2,
        activation_max: 0.2,
        theoretical_range: 2.0,
        range_utilisation: 0.1,
        sample_count: 30,
    }
}

/// Three output neurons, each yielding two coordinated candidates (a changeSquash
/// plus a setBias/setWeight pair): six in total.
fn output_range_compression_fixture() -> (CreatureJson, Vec<OutputRangeCompressionNeuron>) {
    let uuids = three_uuids("out");
    let creature = fan_in_creature(&uuids, "output", "TANH", uuids.len());
    let detected = uuids
        .iter()
        .map(|u| output_range_compression_neuron(u))
        .collect();
    (creature, detected)
}

#[test]
fn output_range_compression_conversion_builds_every_candidate_without_deadline() {
    let (creature, detected) = output_range_compression_fixture();

    let full = output_range_compression_to_coordinated_candidates_with_deadline(
        &detected, &creature, &None,
    );

    assert_eq!(
        full.len(),
        6,
        "fixture must yield two candidates per neuron"
    );
}

#[test]
fn output_range_compression_conversion_stops_at_an_elapsed_deadline() {
    let (creature, detected) = output_range_compression_fixture();

    let partial = output_range_compression_to_coordinated_candidates_with_deadline(
        &detected,
        &creature,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first neuron, got {} candidates",
        partial.len()
    );
}

#[test]
fn output_range_compression_legacy_conversion_matches_deadline_variant_without_deadline() {
    let (creature, detected) = output_range_compression_fixture();

    let legacy = output_range_compression_to_coordinated_candidates(&detected, &creature);
    let with_none = output_range_compression_to_coordinated_candidates_with_deadline(
        &detected, &creature, &None,
    );

    assert_eq!(to_json(&legacy), to_json(&with_none));
}

// ---------------------------------------------------------------------------
// sentinel_gating_to_coordinated_candidates_with_deadline
// ---------------------------------------------------------------------------

fn sentinel_gating_candidate(uuid: &str) -> SentinelGatingCandidate {
    SentinelGatingCandidate {
        neuron_uuid: uuid.to_string(),
        sentinel_value: -1.0,
        sentinel_fraction: 0.3,
        useful_range_min: 0.0,
        useful_range_max: 1.0,
        sample_count: 30,
        estimated_improvement: 0.01,
    }
}

/// `count` observation inputs `in-{i}`, each feeding `out-a` and `out-b`, so every
/// candidate has downstream synapses and converts to one coordinated candidate.
fn sentinel_gating_fixture(count: usize) -> (CreatureJson, Vec<SentinelGatingCandidate>) {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    let mut candidates = Vec::new();
    for i in 0..count {
        let uuid = format!("in-{i}");
        neurons.push(neuron_json(&uuid, "input", "IDENTITY"));
        for target in ["out-a", "out-b"] {
            synapses.push(synapse_json(&uuid, target, 0.5));
        }
        candidates.push(sentinel_gating_candidate(&uuid));
    }
    for target in ["out-a", "out-b"] {
        neurons.push(neuron_json(target, "output", "IDENTITY"));
    }
    let creature = CreatureJson {
        neurons,
        synapses,
        input: count,
        output: 2,
    };
    (creature, candidates)
}

#[test]
fn sentinel_gating_conversion_builds_every_candidate_without_deadline() {
    let (creature, candidates) = sentinel_gating_fixture(3);

    let full =
        sentinel_gating_to_coordinated_candidates_with_deadline(&candidates, &creature, &None);

    assert_eq!(
        full.len(),
        3,
        "fixture must yield one candidate per observation"
    );
}

#[test]
fn sentinel_gating_conversion_stops_at_an_elapsed_deadline() {
    let (creature, candidates) = sentinel_gating_fixture(3);

    let partial = sentinel_gating_to_coordinated_candidates_with_deadline(
        &candidates,
        &creature,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first candidate, got {} candidates",
        partial.len()
    );
}

#[test]
fn sentinel_gating_legacy_conversion_matches_deadline_variant_without_deadline() {
    let (creature, candidates) = sentinel_gating_fixture(3);

    let legacy = sentinel_gating_to_coordinated_candidates(&candidates, &creature);
    let with_none =
        sentinel_gating_to_coordinated_candidates_with_deadline(&candidates, &creature, &None);

    assert_eq!(to_json(&legacy), to_json(&with_none));
}

// ---------------------------------------------------------------------------
// detect_squash_weight_rescale_candidates_with_deadline
// ---------------------------------------------------------------------------

/// 40 records for a RELU neuron with predominantly negative pre-activations and a
/// consistent error, so each hidden neuron built from them qualifies for a candidate.
fn qualifying_records(uuid: &str) -> Vec<DiscoverRecord> {
    (0..40u16)
        .map(|i| {
            let value = (f32::from(i) - 20.0) / 10.0;
            DiscoverRecord {
                obs_index: u32::from(i),
                neuron_uuid: uuid.to_string(),
                value: Some(value),
                activation: value.max(0.0),
                errors: vec![0.15 + (f32::from(i) * 0.002).sin() * 0.02],
            }
        })
        .collect()
}

type SquashFixture = (
    CreatureJson,
    Vec<(String, String, f32)>,
    Vec<(String, Vec<DiscoverRecord>)>,
);

/// `count` RELU hidden neurons `h{i}`, each with two incoming synapses and two
/// non-aggregate outputs, so every hidden neuron performs the aggregate-feed check.
fn squash_fixture(count: usize) -> SquashFixture {
    let mut neurons = vec![
        neuron_json("in-a", "input", "IDENTITY"),
        neuron_json("in-b", "input", "IDENTITY"),
    ];
    let mut synapses = Vec::new();
    let mut hidden = Vec::new();
    let mut records = Vec::new();
    for i in 0..count {
        let uuid = format!("h{i}");
        neurons.push(neuron_json(&uuid, "hidden", "RELU"));
        synapses.push(synapse_json("in-a", &uuid, 1.5));
        synapses.push(synapse_json("in-b", &uuid, -0.8));
        synapses.push(synapse_json(&uuid, "out-a", 1.0));
        synapses.push(synapse_json(&uuid, "out-b", 1.0));
        hidden.push((uuid.clone(), "RELU".to_string(), 0.0));
        records.push((uuid.clone(), qualifying_records(&uuid)));
    }
    for (uuid, squash) in [("out-a", "TANH"), ("out-b", "IDENTITY")] {
        neurons.push(neuron_json(uuid, "output", squash));
    }
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 2,
        output: 2,
    };
    (creature, hidden, records)
}

#[test]
fn squash_weight_rescale_detection_builds_every_candidate_without_deadline() {
    let (creature, hidden, records) = squash_fixture(3);

    let full =
        detect_squash_weight_rescale_candidates_with_deadline(&creature, &hidden, &records, &None);

    assert_eq!(full.len(), 3, "every hidden neuron must qualify");
}

#[test]
fn squash_weight_rescale_detection_stops_at_an_elapsed_deadline() {
    let (creature, hidden, records) = squash_fixture(3);

    let partial = detect_squash_weight_rescale_candidates_with_deadline(
        &creature,
        &hidden,
        &records,
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        partial.is_empty(),
        "an elapsed deadline must stop before the first hidden neuron, got {} candidates",
        partial.len()
    );
}

#[test]
fn squash_weight_rescale_legacy_detection_matches_deadline_variant_without_deadline() {
    let (creature, hidden, records) = squash_fixture(3);

    let legacy = detect_squash_weight_rescale_candidates(&creature, &hidden, &records);
    let with_none =
        detect_squash_weight_rescale_candidates_with_deadline(&creature, &hidden, &records, &None);

    assert_eq!(format!("{legacy:?}"), format!("{with_none:?}"));
}
