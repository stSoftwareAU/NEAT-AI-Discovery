//! Issue #2236: the O(outputs²) output-competition pair scan honours the
//! analysis deadline (and host cancellation) and reports an early return.
//!
//! Deterministic: an elapsed deadline is `UNIX_EPOCH`, a live one is a day
//! ahead — no sleeps and no timing thresholds.

use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::recommendation::epistatic::ScanTruncation;
use neat_ai_discovery::analysis::recommendation::output_competition::{
    OutputCompetitionCandidate, detect_output_competition, detect_output_competition_with_deadline,
};
use neat_ai_discovery::analysis::task_descriptor::TaskDescriptor;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson};

/// Wide enough that every one of the `n(n-1)/2` pairs is a real candidate.
const OUTPUTS: usize = 64;
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

/// `n` outputs that all co-fire on every sample, with no output→output
/// synapses, so every ordered pair is an output-competition candidate.
fn wide_creature(n: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let mut neurons = vec![neuron("input-1", "input")];
    let mut records = Vec::with_capacity(n);
    for o in 0..n {
        let uuid = format!("output-{o}");
        neurons.push(neuron(&uuid, "output"));
        let recs = (0..SAMPLES)
            .map(|idx| DiscoverRecord {
                obs_index: idx,
                neuron_uuid: uuid.clone(),
                value: Some(0.0),
                activation: 0.9,
                errors: vec![0.0],
            })
            .collect();
        records.push((uuid, recs));
    }
    let creature = CreatureJson {
        neurons,
        synapses: Vec::new(),
        input: 1,
        output: n,
    };
    (creature, records)
}

fn one_hot() -> TaskDescriptor {
    TaskDescriptor::from_name("CATEGORICAL_ERROR", 2)
}

fn far_future_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(24 * 60 * 60))
}

fn pairs(candidates: &[OutputCompetitionCandidate]) -> Vec<(String, String)> {
    candidates
        .iter()
        .map(|c| (c.from_output_uuid.clone(), c.to_output_uuid.clone()))
        .collect()
}

#[test]
fn elapsed_deadline_returns_without_scanning_and_reports_truncation() {
    let (creature, records) = wide_creature(OUTPUTS);

    let scan = detect_output_competition_with_deadline(
        &creature,
        &records,
        &one_hot(),
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(
        scan.candidates.is_empty(),
        "an elapsed deadline must stop the scan before any pair is scored, got {}",
        scan.candidates.len()
    );
    assert_eq!(scan.truncation, Some(ScanTruncation::DeadlinePassed));
}

#[test]
fn far_future_deadline_yields_every_candidate_and_no_truncation() {
    let (creature, records) = wide_creature(OUTPUTS);

    let scan = detect_output_competition_with_deadline(
        &creature,
        &records,
        &one_hot(),
        &far_future_deadline(),
    );

    assert_eq!(scan.truncation, None, "a complete scan is not truncated");
    assert_eq!(scan.candidates.len(), OUTPUTS * (OUTPUTS - 1) / 2);
    // Same result as the deadline-free entry point.
    let unbounded = detect_output_competition(&creature, &records, &one_hot());
    assert_eq!(pairs(&scan.candidates), pairs(&unbounded));
}

#[test]
fn non_competing_topology_is_not_reported_as_truncated() {
    let (creature, records) = wide_creature(OUTPUTS);

    // The topology gate returns before the scan, so even an elapsed deadline
    // has nothing to truncate.
    let scan = detect_output_competition_with_deadline(
        &creature,
        &records,
        &TaskDescriptor::neutral(),
        &Some(SystemTime::UNIX_EPOCH),
    );

    assert!(scan.candidates.is_empty());
    assert_eq!(scan.truncation, None);
}
