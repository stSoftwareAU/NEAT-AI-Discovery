//! Issue #2370 (discriminating regression): `detect_dead_neurons` ran one
//! depth-first search per dead hidden neuron, each allocating a visited set and
//! stack sized to the whole creature, so a scan cost O(D * (V + E)) -- quadratic
//! on a chain of dead neurons. It is now one shared memoised pass.
//!
//! This uses only the long-standing `detect_dead_neurons` entry point so it
//! compiles and runs against the pre-fix code as well as the fix (the crate
//! already owns the global allocator, so allocation cannot be counted from an
//! integration test). It compares the *ratio* of scan times for a chain and one
//! four times longer, in the same process: a 4x larger chain costs ~4x with
//! shared work and ~16x with a search per neuron, and the 9x bound sits between
//! them with a wide margin on both sides. The fastest of several repetitions is
//! used so a scheduling hiccup cannot inflate the ratio.

use std::time::{Duration, Instant};

use neat_ai_discovery::analysis::detection::dead_neuron::detect_dead_neurons;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Clears `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 30;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "LOGISTIC".to_string(),
        bias: 0.0,
    }
}

fn synapse(from_uuid: &str, to_uuid: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from_uuid.to_string(),
        to_uuid: to_uuid.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// `input-0 -> h0 -> ... -> h{n-1} -> output-0`, every hidden neuron dead.
fn chain(n: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let hidden: Vec<String> = (0..n).map(|h| format!("h{h}")).collect();

    let mut neurons = vec![neuron("input-0", "input")];
    neurons.extend(hidden.iter().map(|u| neuron(u, "hidden")));
    neurons.push(neuron("output-0", "output"));

    let mut synapses = Vec::with_capacity(n + 1);
    let mut prev = "input-0";
    for h in &hidden {
        synapses.push(synapse(prev, h));
        prev = h;
    }
    synapses.push(synapse(prev, "output-0"));

    let records = hidden
        .iter()
        .map(|uuid| {
            let recs = (0..SAMPLES)
                .map(|idx| DiscoverRecord {
                    obs_index: idx,
                    neuron_uuid: uuid.clone(),
                    value: Some(0.0),
                    activation: 0.0,
                    errors: vec![0.0],
                })
                .collect();
            (uuid.clone(), recs)
        })
        .collect();

    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: 1,
    };
    (creature, records)
}

/// Fastest of `REPS` scans of an `n`-neuron dead chain, plus the candidate count.
fn fastest_scan(n: usize) -> (Duration, usize) {
    const REPS: usize = 3;
    let (creature, records) = chain(n);
    let mut best = Duration::MAX;
    let mut found = 0;
    for _ in 0..REPS {
        let start = Instant::now();
        let candidates = detect_dead_neurons(&creature, &records, None);
        best = best.min(start.elapsed());
        found = candidates.len();
    }
    (best, found)
}

#[test]
fn scan_time_grows_linearly_with_dead_count() {
    let (time_small, found_small) = fastest_scan(1_000);
    let (time_large, found_large) = fastest_scan(4_000);

    assert_eq!(found_small, 1_000, "all 1000 dead neurons found");
    assert_eq!(found_large, 4_000, "all 4000 dead neurons found");

    let ratio = time_large.as_secs_f64() / time_small.as_secs_f64().max(1e-9);
    assert!(
        ratio < 9.0,
        "scan work must grow linearly: N=1000 took {time_small:?}, N=4000 took \
         {time_large:?} (ratio {ratio:.1}; linear is ~4x, per-neuron DFS is ~16x)"
    );
}
