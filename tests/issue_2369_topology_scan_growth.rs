//! Regression test for Issue #2369 (CWE-834): `detect_topology_issues` rescans
//! every output neuron for every long-path hidden neuron when choosing a skip
//! target, so the pass grows as O(hidden x outputs) with no cancellation point.
//!
//! The growth test is a ratio test rather than a wall-clock budget: absolute
//! timings vary wildly across CI machines, but the ratio t(4N)/t(N) is stable.
//! Linear growth gives about 4x, quadratic about 16x. Each size is timed as the
//! best of several runs to suppress scheduler noise.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use neat_ai_discovery::analysis::detection::topology::{TopologyCandidate, detect_topology_issues};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Records per qualified hidden neuron (the minimum that qualifies for detection).
const RECORDS_PER_NEURON: u32 = 20;

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

/// Non-zero errors so every long-path candidate has a positive estimated gain.
fn records_for(uuid: &str) -> Vec<DiscoverRecord> {
    (0..RECORDS_PER_NEURON)
        .map(|obs_index| DiscoverRecord {
            obs_index,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.2,
            errors: vec![0.2],
        })
        .collect()
}

/// `k` qualified hidden neurons, each fed from an input and each feeding a
/// shared tail `tail-1 -> tail-2 -> tail-3 -> out-0`. Every `hidden-i` is four
/// hops from an output, beyond the three-hop limit. The tail neurons have no
/// records, so they are not qualified. Outputs `out-0..out-{k-1}` are present,
/// and only `out-0` has a synapse.
fn build(k: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("in-0", "input"), neuron("in-1", "input")];
    let mut synapses = Vec::new();
    let mut records = Vec::new();
    for i in 0..k {
        let uuid = format!("hidden-{i}");
        let input = if i % 2 == 0 { "in-0" } else { "in-1" };
        neurons.push(neuron(&uuid, "hidden"));
        synapses.push(synapse(input, &uuid, 0.5));
        synapses.push(synapse(&uuid, "tail-1", 0.5));
        records.push((uuid.clone(), records_for(&uuid)));
    }
    neurons.push(neuron("tail-1", "hidden"));
    neurons.push(neuron("tail-2", "hidden"));
    neurons.push(neuron("tail-3", "hidden"));
    synapses.push(synapse("tail-1", "tail-2", 0.5));
    synapses.push(synapse("tail-2", "tail-3", 0.5));
    synapses.push(synapse("tail-3", "out-0", 0.5));
    for j in 0..k {
        neurons.push(neuron(&format!("out-{j}"), "output"));
    }
    (
        CreatureJson {
            neurons,
            synapses,
            input: 2,
            output: k,
        },
        records,
    )
}

fn best_of_three(k: usize) -> Duration {
    let (creature, records) = build(k);
    (0..3)
        .map(|_| {
            let start = Instant::now();
            let out = detect_topology_issues(&creature, &records, None);
            let elapsed = start.elapsed();
            std::hint::black_box(out);
            elapsed
        })
        .min()
        .expect("three runs")
}

#[test]
fn long_path_scan_grows_linearly_with_outputs() {
    const N: usize = 500;

    // Correctness on the N-sized run so the timing assertion is not vacuous.
    let (creature, records) = build(N);
    let out = detect_topology_issues(&creature, &records, None);
    let long: Vec<&TopologyCandidate> =
        out.iter().filter(|c| c.issue_type == "long_path").collect();
    assert_eq!(long.len(), N, "one long_path candidate per hidden neuron");
    let by_uuid: HashMap<&str, &TopologyCandidate> =
        long.iter().map(|c| (c.neuron_uuid.as_str(), *c)).collect();
    assert_eq!(by_uuid.len(), N);
    for i in 0..N {
        let uuid = format!("hidden-{i}");
        let c = by_uuid[uuid.as_str()];
        assert_eq!(c.path_length, 4, "{uuid}");
        let target = c
            .skip_target_uuid
            .as_deref()
            .unwrap_or_else(|| panic!("{uuid}: expected a skip target"));
        assert!(target.starts_with("out-"), "{uuid}: got {target}");
        assert!(
            !creature
                .synapses
                .iter()
                .any(|s| s.from_uuid == uuid && s.to_uuid == target),
            "{uuid}: skip target {target} is already connected"
        );
    }

    let small = best_of_three(N);
    let large = best_of_three(4 * N);
    let ratio = large.as_secs_f64() / small.as_secs_f64().max(1e-9);
    eprintln!("t(N)={small:?} t(4N)={large:?} ratio={ratio:.2}");
    assert!(
        ratio < 10.0,
        "long-path scan should grow ~linearly: t({N})={small:?}, t({})={large:?}, ratio={ratio:.2} (linear ~4, quadratic ~16)",
        4 * N
    );
}
