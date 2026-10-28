//! Issue #2369 (CWE-834): the two per-neuron scans in `detect_topology_issues`
//! were uncancellable, and each scanned a linear list per neuron.
//!
//! - The long-path scan searched every output per neuron (O(H·O)).
//! - The starved scan searched every input per neuron (O(H·I)).
//!
//! The fix builds sorted candidate lists once and stops each scan at the first
//! unconnected candidate. These tests pin the chosen candidates, which are now
//! deterministic (smallest uuid). The deadline checks are covered in-crate by
//! `topology_deadline_tests.rs`, which needs the `#[cfg(test)]` deadline seam.
//!
//! Only the pre-existing `detect_topology_issues` API is used, so these tests
//! compile and run against the base too. There the candidate followed `HashSet`
//! iteration order, which is random per set, so each test repeats the run
//! `REPEATS` times to make a base pass by luck vanishingly unlikely.

use neat_ai_discovery::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use neat_ai_discovery::analysis::detection::topology::{TopologyCandidate, detect_topology_issues};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Observations per hidden neuron. Must be at least `MIN_DISCOVERY_SAMPLE_COUNT`
/// so the neuron qualifies for topology analysis.
const SAMPLES: u32 = 30;

const _: () = assert!(SAMPLES as usize >= MIN_DISCOVERY_SAMPLE_COUNT);

/// Repeats per test. On base each run picks the smallest uuid with probability
/// at most 1/4 per candidate, so 40 runs cannot all pass by chance.
const REPEATS: usize = 40;

type Records = Vec<(String, Vec<DiscoverRecord>)>;

fn neuron(uuid: &str, kind: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: kind.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    }
}

fn hidden(uuid: &str) -> NeuronJson {
    neuron(uuid, "hidden")
}

fn syn(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// `SAMPLES` observations with a non-zero error, so the neuron qualifies and
/// its candidates have a positive estimated improvement.
fn qualified_records(uuid: &str) -> (String, Vec<DiscoverRecord>) {
    let recs = (0..SAMPLES)
        .map(|idx| DiscoverRecord {
            obs_index: idx,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![0.5],
        })
        .collect();
    (uuid.to_string(), recs)
}

fn records_for(uuids: &[&str]) -> Records {
    uuids.iter().map(|u| qualified_records(u)).collect()
}

fn input_neurons() -> Vec<NeuronJson> {
    (0..6)
        .map(|i| neuron(&format!("in-{i}"), "input"))
        .collect()
}

/// Adds the chain `in-0 -> h0 -> h1 -> h2 -> h3 -> h4 -> last_output`. `h0` is
/// 5 hops from the outputs and `h1` is 4, so both are long-path.
fn push_chain(neurons: &mut Vec<NeuronJson>, synapses: &mut Vec<SynapseJson>, last_output: &str) {
    neurons.extend(["h0", "h1", "h2", "h3", "h4"].map(hidden));
    synapses.push(syn("in-0", "h0"));
    for (from, to) in [("h0", "h1"), ("h1", "h2"), ("h2", "h3"), ("h3", "h4")] {
        synapses.push(syn(from, to));
    }
    synapses.push(syn("h4", last_output));
}

/// Adds `rich`, fed by all six inputs, and `starved-0..2`, each fed only by
/// `in-0` and `in-1` (fan-in 2). Every one of them feeds `output`.
///
/// With fan-ins {6, 2, 2, 2} the ratio is 6/2 = 3.0, which meets
/// `MIN_IMBALANCE_RATIO`. The starved threshold is ceil(6/3) = 2, so the three
/// starved neurons are flagged and `rich` is not.
fn push_rich_and_starved(
    neurons: &mut Vec<NeuronJson>,
    synapses: &mut Vec<SynapseJson>,
    output: &str,
) {
    neurons.extend(["rich", "starved-0", "starved-1", "starved-2"].map(hidden));
    for i in 0..6 {
        synapses.push(syn(&format!("in-{i}"), "rich"));
    }
    synapses.push(syn("rich", output));
    for starved in ["starved-0", "starved-1", "starved-2"] {
        synapses.push(syn("in-0", starved));
        synapses.push(syn("in-1", starved));
        synapses.push(syn(starved, output));
    }
}

/// Chain to `out-5`, with eight outputs `out-0..out-7`. Every output is
/// unconnected from `h0` and `h1`, so the smallest-uuid output is `out-0`.
fn long_path_creature() -> (CreatureJson, Records) {
    let mut neurons = input_neurons();
    let mut synapses = Vec::new();
    push_chain(&mut neurons, &mut synapses, "out-5");
    for i in 0..8 {
        neurons.push(neuron(&format!("out-{i}"), "output"));
    }
    let records = records_for(&["h0", "h1", "h2", "h3", "h4"]);
    (
        CreatureJson {
            neurons,
            synapses,
            input: 6,
            output: 8,
        },
        records,
    )
}

/// One output, with `rich` and `starved-0..2` feeding it. Fan-in ratio is 3.0.
fn starved_creature() -> (CreatureJson, Records) {
    let mut neurons = input_neurons();
    let mut synapses = Vec::new();
    neurons.push(neuron("output-0", "output"));
    push_rich_and_starved(&mut neurons, &mut synapses, "output-0");
    let records = records_for(&["rich", "starved-0", "starved-1", "starved-2"]);
    (
        CreatureJson {
            neurons,
            synapses,
            input: 6,
            output: 1,
        },
        records,
    )
}

fn of_kind<'a>(
    candidates: &'a [TopologyCandidate],
    issue_type: &'a str,
) -> impl Iterator<Item = &'a TopologyCandidate> + 'a {
    candidates
        .iter()
        .filter(move |c| c.issue_type == issue_type)
}

#[test]
fn long_path_skip_target_is_the_smallest_unconnected_output() {
    for _ in 0..REPEATS {
        assert_long_path_targets_smallest_output();
    }
}

fn assert_long_path_targets_smallest_output() {
    let (creature, records) = long_path_creature();
    let candidates = detect_topology_issues(&creature, &records, None);

    let long_path: Vec<&TopologyCandidate> = of_kind(&candidates, "long_path").collect();
    assert!(
        !long_path.is_empty(),
        "fixture must yield long-path candidates, got: {candidates:?}"
    );
    for c in &long_path {
        assert_eq!(
            c.skip_target_uuid.as_deref(),
            Some("out-0"),
            "skip target for {} must be the smallest-uuid unconnected output",
            c.neuron_uuid
        );
    }
}

#[test]
fn starved_neuron_source_is_the_smallest_unconnected_input() {
    for _ in 0..REPEATS {
        assert_starved_sources_smallest_input();
    }
}

fn assert_starved_sources_smallest_input() {
    let (creature, records) = starved_creature();
    let candidates = detect_topology_issues(&creature, &records, None);

    let imbalance: Vec<&TopologyCandidate> =
        of_kind(&candidates, "connectivity_imbalance").collect();
    assert!(
        !imbalance.is_empty(),
        "fixture must yield connectivity-imbalance candidates, got: {candidates:?}"
    );
    for c in &imbalance {
        assert_eq!(
            c.add_source_uuid.as_deref(),
            Some("in-2"),
            "add source for {} must be the smallest-uuid unconnected input",
            c.neuron_uuid
        );
    }
}
