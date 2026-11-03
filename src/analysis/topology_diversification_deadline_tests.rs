//! Work-count, deadline and behaviour tests for the topology-diversification scan (Issue #2489).
//!
//! These use the `DeadlineOverrideGuard` test seam (`#[cfg(test)]`-only) to script
//! `deadline_passed`. The scan makes no check during its setup and one check per output at the
//! top of the per-output loop. The `on_step` observer is the deterministic work counter: one
//! step per synapse in the index build, plus one per predecessor edge examined by each output's
//! unhealthy-intermediate walk.
//!
//! `#[serial]` keeps these tests from interleaving with any other test that installs a
//! deadline override or flips the global cancellation flag.

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use crate::types::DiscoverRecord;
use crate::{CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;

/// Samples per neuron, above `MIN_DISCOVERY_SAMPLE_COUNT` (20).
const SAMPLES: u32 = 30;

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

/// `SAMPLES` records for one neuron. The activation and the error of sample `i` come from
/// the two closures.
fn samples(
    uuid: &str,
    activation: impl Fn(u32) -> f32,
    error: impl Fn(u32) -> f32,
) -> (String, Vec<DiscoverRecord>) {
    let records = (0..SAMPLES)
        .map(|i| DiscoverRecord {
            obs_index: i,
            neuron_uuid: uuid.to_string(),
            value: None,
            activation: activation(i),
            errors: vec![error(i)],
        })
        .collect();
    (uuid.to_string(), records)
}

/// One input `in` with a direct synapse to each of `outputs` output neurons `out-0 ..`.
/// Every output has mean error 0.25 and no hidden neuron, so each one is a candidate.
fn fan_in_fixture(outputs: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("in", "input")];
    let mut synapses = Vec::new();
    let mut records = Vec::new();
    for i in 0..outputs {
        let uuid = format!("out-{i}");
        neurons.push(neuron(&uuid, "output"));
        synapses.push(synapse("in", &uuid));
        records.push(samples(&uuid, |_| 0.5, |_| 0.25));
    }
    let creature = CreatureJson {
        neurons,
        synapses,
        input: 1,
        output: outputs,
    };
    (creature, records)
}

/// The work counter must grow linearly with the outputs. The index is built once (one step per
/// synapse), and each output walks one predecessor edge. Quadrupling the outputs must quadruple
/// the steps; the old per-output rescan made them grow with the square of the outputs.
#[test]
#[serial]
fn work_is_linear_in_outputs_not_quadratic() {
    let mut steps_by_outputs = Vec::new();
    for outputs in [32_usize, 128] {
        let (creature, records) = fan_in_fixture(outputs);
        let mut steps = 0_usize;
        let scan = detect_topology_diversification_observed(&creature, &records, &None, || {
            steps += 1;
        });

        assert_eq!(
            scan.candidates.len(),
            outputs,
            "every output must be a candidate"
        );
        assert_eq!(scan.stop, None);
        assert_eq!(
            steps,
            2 * outputs,
            "one step per synapse in the index build, plus one predecessor edge per output"
        );
        steps_by_outputs.push(steps);
    }

    assert_eq!(
        steps_by_outputs[1],
        4 * steps_by_outputs[0],
        "steps must scale linearly with the outputs, got {steps_by_outputs:?}"
    );
}

/// The deadline is checked once per output at the top of the loop. A stop keeps the candidates
/// from the outputs already processed, in neuron order, and reports `DeadlinePassed`.
#[test]
#[serial]
fn deadline_stops_at_per_output_check_and_keeps_earlier_candidates() {
    let (creature, records) = fan_in_fixture(3);

    // Positive control: with no override all three outputs are candidates, nothing stops, and
    // the result matches the deadline-less wrapper.
    let control =
        detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(control.stop, None);
    assert_eq!(
        control.candidates.len(),
        3,
        "fixture must yield one candidate per output"
    );
    let wrapper: Vec<String> = detect_topology_diversification_candidates(&creature, &records)
        .into_iter()
        .map(|c| c.output_neuron_uuid)
        .collect();
    let control_uuids: Vec<String> = control
        .candidates
        .iter()
        .map(|c| c.output_neuron_uuid.clone())
        .collect();
    assert_eq!(control_uuids, wrapper);

    // [false = check before out-0, true = check before out-1]: out-0 is kept, then the scan stops.
    {
        let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);
        let scan =
            detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
        assert_eq!(scan.stop, Some(TopologyDiversificationStop::DeadlinePassed));
        let kept: Vec<&str> = scan
            .candidates
            .iter()
            .map(|c| c.output_neuron_uuid.as_str())
            .collect();
        assert_eq!(
            kept,
            vec!["out-0"],
            "only the output processed before the stop may be kept"
        );
    }

    // [true] stops before the first output, so nothing is kept.
    {
        let _guard = DeadlineOverrideGuard::with_sequence(vec![true]);
        let scan =
            detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
        assert!(
            scan.candidates.is_empty(),
            "no output was processed before the first check fired, got: {:?}",
            scan.candidates
        );
        assert_eq!(scan.stop, Some(TopologyDiversificationStop::DeadlinePassed));
    }
}

/// A duplicated direct synapse is a second direct path, as the old per-output filter counted
/// it. The source is that input.
#[test]
#[serial]
fn duplicate_direct_synapse_counts_as_two_paths() {
    let creature = CreatureJson {
        neurons: vec![neuron("in", "input"), neuron("out", "output")],
        synapses: vec![synapse("in", "out"), synapse("in", "out")],
        input: 1,
        output: 1,
    };
    let records = vec![
        samples("out", |_| 0.5, |_| 0.25),
        samples("in", |i| (i % 2) as f32, |_| 0.0),
    ];

    let scan = detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(scan.candidates.len(), 1);
    assert_eq!(scan.candidates[0].direct_path_count, 2);
    assert_eq!(scan.candidates[0].source_input_uuid, "in");
}

/// With two direct inputs, the one with the higher activation variance is the source, whichever
/// order the synapses list them in. `lo` is listed first and has zero variance, `hi` has 0.25.
#[test]
#[serial]
fn source_is_highest_activation_variance_input() {
    let creature = CreatureJson {
        neurons: vec![
            neuron("lo", "input"),
            neuron("hi", "input"),
            neuron("out", "output"),
        ],
        synapses: vec![synapse("lo", "out"), synapse("hi", "out")],
        input: 2,
        output: 1,
    };
    let records = vec![
        samples("out", |_| 0.5, |_| 0.25),
        samples("lo", |_| 0.5, |_| 0.0),
        samples("hi", |i| (i % 2) as f32, |_| 0.0),
    ];

    let scan = detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(scan.candidates.len(), 1);
    assert_eq!(scan.candidates[0].direct_path_count, 2);
    assert_eq!(
        scan.candidates[0].source_input_uuid, "hi",
        "the input with the highest activation variance must be the source"
    );
}

/// `h -> ghost -> out`, where `ghost` is a dangling endpoint (no such neuron) and `in -> out` is
/// the direct input. The predecessor walk from `out` reaches `h` through the ghost. `h`'s error
/// series decides whether the output is suppressed.
fn ghost_fixture(
    h_error: impl Fn(u32) -> f32,
) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let creature = CreatureJson {
        neurons: vec![
            neuron("in", "input"),
            neuron("h", "hidden"),
            neuron("out", "output"),
        ],
        synapses: vec![
            synapse("in", "out"),
            synapse("h", "ghost"),
            synapse("ghost", "out"),
        ],
        input: 1,
        output: 1,
    };
    let records = vec![
        samples("out", |_| 0.5, |_| 0.25),
        samples("h", |_| 0.5, h_error),
    ];
    (creature, records)
}

/// Behaviour preserved through the index: a high-CV hidden neuron reached only through a dangling
/// endpoint still suppresses the candidate, as it did before the index. The same fixture with a
/// steady hidden error produces the candidate, so the suppression is what is being tested.
#[test]
#[serial]
fn dangling_endpoint_still_reaches_unhealthy_intermediate() {
    // Control: constant error gives CV 0, so the walk finds nothing unhealthy.
    let (creature, records) = ghost_fixture(|_| 0.01);
    let scan = detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
    assert_eq!(
        scan.candidates.len(),
        1,
        "steady hidden error must leave the candidate in place"
    );
    assert_eq!(scan.candidates[0].source_input_uuid, "in");

    // Alternating 0.01 / 1.0 gives CV of about 0.98, above the 0.8 threshold.
    let (creature, records) = ghost_fixture(|i| if i % 2 == 0 { 0.01 } else { 1.0 });
    let scan = detect_topology_diversification_candidates_with_deadline(&creature, &records, &None);
    assert!(
        scan.candidates.is_empty(),
        "a high-CV hidden neuron behind a dangling endpoint must suppress the candidate, got: {:?}",
        scan.candidates
    );
    assert_eq!(scan.stop, None);
}
