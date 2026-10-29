//! Integration tests for the skip-connection pair scan (Issue #2368).
//!
//! Every assertion here is deterministic. Work is counted through the `on_step`
//! observer of `detect_skip_connection_candidates_observed`, never by wall-clock
//! time. The deadline seam (`DeadlineOverrideGuard`) is `#[cfg(test)]`-only inside
//! the library, so the expired-deadline case uses a real `UNIX_EPOCH` deadline.
//! Nothing in this binary touches the global cancellation flag, so no test here
//! needs `#[serial]`.

use std::time::UNIX_EPOCH;

use neat_ai_discovery::analysis::constants::MIN_DISCOVERY_SAMPLE_COUNT;
use neat_ai_discovery::analysis::detection::skip_connection::{
    SkipConnectionScan, SkipConnectionStop, detect_skip_connection_candidates_observed,
    detect_skip_connection_candidates_with_deadline,
};
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Mean absolute error of a shallow hidden neuron (depth <= 2).
const SHALLOW_ERROR: f32 = 0.5;

/// Mean absolute error of an attenuated deep neuron. Its ratio to the shallow mean is 0.1,
/// below the attenuation threshold of 0.5.
const DEEP_ERROR: f32 = 0.05;

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: neuron_type.to_string(),
        squash: "TANH".to_string(),
        bias: 0.0,
    }
}

fn synapse(from: &str, to: &str) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight: 0.5,
        synapse_type: None,
    }
}

/// `MIN_DISCOVERY_SAMPLE_COUNT` records for one hidden neuron, each carrying `error`.
fn hidden_records(uuid: &str, error: f32) -> (String, Vec<DiscoverRecord>) {
    let records = (0..MIN_DISCOVERY_SAMPLE_COUNT)
        .map(|i| DiscoverRecord {
            obs_index: u32::try_from(i).expect("sample index fits in u32"),
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: 0.5,
            errors: vec![error],
        })
        .collect();
    (uuid.to_string(), records)
}

/// `in0 -> c0 -> c1 -> c2 -> c3`. `c0` (depth 1) and `c1` (depth 2) are shallow. `c2`
/// (depth 3) and `c3` (depth 4) are deep and attenuated, so each yields a candidate from
/// `in0`. With `back_edge`, `c3 -> c1` closes a cycle among hidden neurons that is
/// reachable from `in0`.
fn chain_fixture(back_edge: bool) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut synapses = vec![
        synapse("in0", "c0"),
        synapse("c0", "c1"),
        synapse("c1", "c2"),
        synapse("c2", "c3"),
    ];
    if back_edge {
        synapses.push(synapse("c3", "c1"));
    }
    let creature = CreatureJson {
        neurons: vec![
            neuron("in0", "input"),
            neuron("c0", "hidden"),
            neuron("c1", "hidden"),
            neuron("c2", "hidden"),
            neuron("c3", "hidden"),
        ],
        synapses,
        input: 1,
        output: 0,
    };
    let records = vec![
        hidden_records("c0", SHALLOW_ERROR),
        hidden_records("c1", SHALLOW_ERROR),
        hidden_records("c2", DEEP_ERROR),
        hidden_records("c3", DEEP_ERROR),
    ];
    (creature, records)
}

/// The acyclic `chain_fixture` plus `h` deep hidden neurons `d{i}` fed from `c3` (depth 5)
/// and `h` depth-1 hidden neurons `s{i}` fed from `in0`. There are `h + 2` deep targets and
/// `h + 1` depth-1 hidden sources, so a scan that visits every (target, source) pair does
/// about `h^2` visits, while the indexed scan does about `h` steps in its target loop.
fn scaled_fixture(h: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let (mut creature, mut records) = chain_fixture(false);
    for i in 0..h {
        let deep = format!("d{i}");
        let shallow = format!("s{i}");
        creature.neurons.push(neuron(&deep, "hidden"));
        creature.neurons.push(neuron(&shallow, "hidden"));
        creature.synapses.push(synapse("c3", &deep));
        creature.synapses.push(synapse("in0", &shallow));
        records.push(hidden_records(&deep, DEEP_ERROR));
        records.push(hidden_records(&shallow, SHALLOW_ERROR));
    }
    (creature, records)
}

/// Runs the observed scan with no deadline and returns the step count with the result.
fn scan_with_step_count(
    creature: &CreatureJson,
    records: &[(String, Vec<DiscoverRecord>)],
) -> (usize, SkipConnectionScan) {
    let mut steps = 0_usize;
    let scan = detect_skip_connection_candidates_observed(creature, records, &None, || steps += 1);
    (steps, scan)
}

/// Issue #2368: the pair scan must grow linearly with the hidden neuron count. The old scan
/// visited every (deep target, shallow source) pair, roughly `h^2` visits here, so
/// quadrupling `h` would multiply the step count by about 16. The bound of 5x allows for the
/// linear depth pass and the O(targets) target loop.
#[test]
fn pair_scan_grows_linearly_with_hidden_count() {
    let small_h = 200;
    let large_h = 4 * small_h;

    let (small_creature, small_records) = scaled_fixture(small_h);
    let (small_steps, small_scan) = scan_with_step_count(&small_creature, &small_records);

    let (large_creature, large_records) = scaled_fixture(large_h);
    let (large_steps, large_scan) = scan_with_step_count(&large_creature, &large_records);

    // Positive control: the target loop ran and every deep target yielded a candidate.
    assert_eq!(small_scan.stop, None);
    assert_eq!(small_scan.candidates.len(), small_h + 2);
    assert_eq!(large_scan.stop, None);
    assert_eq!(large_scan.candidates.len(), large_h + 2);

    assert!(
        large_steps <= 5 * small_steps,
        "steps grew from {small_steps} at H={small_h} to {large_steps} at H={large_h}, \
         which is more than 5x (quadratic-like growth)"
    );
}

/// Issue #2368: a cycle reachable from an input must end the scan with `CycleDetected` and no
/// candidates rather than looping. The same shape without the back edge is the control.
#[test]
fn cyclic_graph_returns_cycle_detected_not_hang() {
    let (cyclic, cyclic_records) = chain_fixture(true);
    let scan = detect_skip_connection_candidates_with_deadline(&cyclic, &cyclic_records, &None);
    assert!(
        scan.candidates.is_empty(),
        "a cyclic graph must return no candidates, got: {:?}",
        scan.candidates
    );
    assert_eq!(scan.stop, Some(SkipConnectionStop::CycleDetected));

    let (acyclic, acyclic_records) = chain_fixture(false);
    let control =
        detect_skip_connection_candidates_with_deadline(&acyclic, &acyclic_records, &None);
    assert_eq!(control.stop, None);
    assert_eq!(
        control.candidates.len(),
        2,
        "the acyclic control must yield one candidate per deep target"
    );
}

/// Issue #2368: an expired deadline stops the scan with `DeadlinePassed` and no candidates.
/// With no deadline the same fixture yields candidates, which proves the fixture reaches the
/// target loop.
#[test]
fn expired_deadline_stops_scan() {
    let (creature, records) = chain_fixture(false);

    let control = detect_skip_connection_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(control.stop, None);
    assert_eq!(
        control.candidates.len(),
        2,
        "the positive control must yield one candidate per deep target"
    );

    let expired = Some(UNIX_EPOCH);
    let scan = detect_skip_connection_candidates_with_deadline(&creature, &records, &expired);
    assert_eq!(scan.stop, Some(SkipConnectionStop::DeadlinePassed));
    assert!(
        scan.candidates.is_empty(),
        "an expired deadline must return no candidates, got: {:?}",
        scan.candidates
    );
}
