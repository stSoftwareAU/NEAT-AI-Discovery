//! Issue #2479: cancellation must stop the per-candidate loop in the seven
//! coordinated-candidate converters and the squash-weight rescale detector.
//!
//! Every deadline-guarded loop also honours the global cancellation flag
//! (Issue #1047), so requesting cancellation before calling a converter must leave
//! it with no candidates. Before this change the converters never consulted the
//! deadline or the flag and returned their full set, so each test below is red on
//! the base and green on the head. It calls only the long-standing public
//! converter functions, so it builds and runs against the base unchanged.
//!
//! The cancellation flag is process-global, so the tests serialise on `GUARD`.

use std::sync::Mutex;

use neat_ai_discovery::analysis::detection::bottleneck::{
    BottleneckNeuronCandidate, bottleneck_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::monotonicity::{
    MonotonicityCandidate, non_monotonic_neurons_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::operating_point::{
    OperatingPointIssue, operating_point_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::output_range_compression::{
    OutputRangeCompressionNeuron, output_range_compression_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::restricted_range::{
    RestrictedRangeNeuron, restricted_range_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::sentinel_gating::{
    SentinelGatingCandidate, sentinel_gating_to_coordinated_candidates,
};
use neat_ai_discovery::analysis::detection::squash_weight_rescale::detect_squash_weight_rescale_candidates;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

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

/// Serialises the tests in this file, which share the process-global flag.
static GUARD: Mutex<()> = Mutex::new(());

/// Runs `convert` once with no cancellation (positive control: it must produce
/// `expected` items) and once with cancellation requested (it must produce none).
fn assert_cancellation_stops<F: Fn() -> usize>(name: &str, expected: usize, convert: F) {
    let _lock = GUARD
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    neat_ai_discovery::cancellation::reset_cancellation();
    let full = convert();
    assert_eq!(
        full, expected,
        "{name}: positive control must build every candidate"
    );

    neat_ai_discovery::cancellation::request_cancellation();
    let cancelled = convert();
    neat_ai_discovery::cancellation::reset_cancellation();
    assert_eq!(
        cancelled, 0,
        "{name}: a cancelled run must stop before the first candidate, got {cancelled}"
    );
}

#[test]
fn bottleneck_stops_when_cancelled() {
    let (creature, candidates) = bottleneck_fixture(3);

    assert_cancellation_stops("bottleneck", 3, || {
        bottleneck_neurons_to_coordinated_candidates(&candidates, &creature, None).len()
    });
}

#[test]
fn monotonicity_stops_when_cancelled() {
    let (creature, candidates) = monotonicity_fixture(3);

    assert_cancellation_stops("monotonicity", 3, || {
        non_monotonic_neurons_to_coordinated_candidates(&candidates, &creature).len()
    });
}

#[test]
fn operating_point_stops_when_cancelled() {
    let (creature, detected) = operating_point_fixture();

    assert_cancellation_stops("operating_point", 9, || {
        operating_point_to_coordinated_candidates(&detected, &creature).len()
    });
}

#[test]
fn restricted_range_stops_when_cancelled() {
    let (creature, detected) = restricted_range_fixture();

    assert_cancellation_stops("restricted_range", 9, || {
        restricted_range_to_coordinated_candidates(&detected, &creature).len()
    });
}

#[test]
fn output_range_compression_stops_when_cancelled() {
    let (creature, detected) = output_range_compression_fixture();

    assert_cancellation_stops("output_range_compression", 6, || {
        output_range_compression_to_coordinated_candidates(&detected, &creature).len()
    });
}

#[test]
fn sentinel_gating_stops_when_cancelled() {
    let (creature, candidates) = sentinel_gating_fixture(3);

    assert_cancellation_stops("sentinel_gating", 3, || {
        sentinel_gating_to_coordinated_candidates(&candidates, &creature).len()
    });
}

#[test]
fn squash_weight_rescale_stops_when_cancelled() {
    let (creature, hidden, records) = squash_fixture(3);

    assert_cancellation_stops("squash_weight_rescale", 3, || {
        detect_squash_weight_rescale_candidates(&creature, &hidden, &records).len()
    });
}
