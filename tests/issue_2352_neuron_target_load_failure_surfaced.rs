//! Issue #2352: a focus target whose own records fail to load must be reported
//! as `TargetRecordLoadFailed`, not silently dropped and misreported as
//! `NoEligibleSources`.
//!
//! Uses the GPU-free `neuron_target_load_harness`, so no GPU is required.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for test data generation

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::cache::RecordCache;
use neat_ai_discovery::analysis::neuron_target_load_harness::{
    NeuronTargetLoadOutcome, run_target_load_pass,
};
use neat_ai_discovery::analysis::shared::NeuronNoCandidateReason;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeNeuronsInput, CreatureJson, NeuronJson, SynapseJson};

const OUTPUT: &str = "output-0";

fn neuron(uuid: &str, neuron_type: &str) -> NeuronJson {
    serde_json::from_value(serde_json::json!({
        "uuid": uuid,
        "type": neuron_type,
        "squash": "TANH",
        "bias": 0.0,
    }))
    .expect("valid neuron")
}

fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    serde_json::from_value(serde_json::json!({
        "fromUUID": from,
        "toUUID": to,
        "weight": weight,
    }))
    .expect("valid synapse")
}

fn input() -> AnalyzeNeuronsInput {
    let creature = CreatureJson {
        neurons: vec![
            neuron("input-0", "input"),
            neuron("input-1", "input"),
            neuron(OUTPUT, "output"),
        ],
        synapses: vec![
            synapse("input-0", OUTPUT, 0.5),
            synapse("input-1", OUTPUT, 0.3),
        ],
        input: 2,
        output: 1,
    };
    AnalyzeNeuronsInput {
        parquet_file: "unused.parquet".to_string(),
        creature,
        focus_neurons: vec![OUTPUT.to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        task_descriptor: None,
    }
}

fn records_for(uuid: &str) -> Vec<DiscoverRecord> {
    (0..20u32)
        .map(|i| {
            let act = 0.1 + i as f32 * 0.01;
            DiscoverRecord::new(i, uuid.to_string(), Some(act), act, vec![0.05])
        })
        .collect()
}

fn run(target_fails: bool) -> NeuronTargetLoadOutcome {
    let cache = Arc::new(RecordCache::with_loader(
        "unused.parquet",
        Arc::new(move |_file: &str, uuid: &str| {
            if target_fails && uuid == OUTPUT {
                anyhow::bail!("simulated target load failure")
            }
            Ok(records_for(uuid))
        }),
    ));
    run_target_load_pass(&input(), &cache).expect("harness pass")
}

#[test]
fn neuron_target_load_failure_is_not_reported_as_no_eligible_sources() {
    let outcome = run(true);
    let for_output: Vec<_> = outcome
        .summaries
        .iter()
        .filter(|s| s.target_uuid == OUTPUT)
        .collect();
    assert_eq!(for_output.len(), 1, "exactly one summary for the output");
    assert_eq!(
        for_output[0].reason,
        NeuronNoCandidateReason::TargetRecordLoadFailed
    );
    assert_ne!(
        for_output[0].reason,
        NeuronNoCandidateReason::NoEligibleSources
    );
}

#[test]
fn neuron_target_load_failure_is_counted() {
    let outcome = run(true);
    let count = outcome
        .load_failures
        .iter()
        .find(|(uuid, _)| uuid == OUTPUT)
        .map(|(_, n)| *n)
        .expect("output has a diagnostics entry");
    assert!(count >= 1, "load failure must be counted, got {count}");
    assert!(
        outcome.target_load_failed.iter().any(|u| u == OUTPUT),
        "target_load_failed flag must be set for the output"
    );
}

#[test]
fn successful_target_load_is_not_reported_as_failed() {
    let outcome = run(false);
    assert!(outcome.target_load_failed.is_empty());
    assert!(outcome.deadline_exceeded.is_empty());
    assert!(outcome.load_failures.iter().all(|(_, n)| *n == 0));
    assert!(
        outcome
            .summaries
            .iter()
            .all(|s| s.reason != NeuronNoCandidateReason::TargetRecordLoadFailed)
    );
}

/// PR #2407 review: a deadline bail from `RecordCache::get` is a timeout, not a
/// record-store fault, so it must not be reported as `TargetRecordLoadFailed`.
#[test]
fn deadline_bail_on_target_load_is_not_reported_as_load_failure() {
    let cache = Arc::new(RecordCache::with_loader_and_deadline(
        "unused.parquet",
        Arc::new(|_file: &str, uuid: &str| Ok(records_for(uuid))),
        Some(SystemTime::now() - Duration::from_secs(1)),
    ));
    let outcome = run_target_load_pass(&input(), &cache).expect("harness pass");

    assert!(
        outcome.deadline_exceeded.iter().any(|u| u == OUTPUT),
        "the deadline bail must be reported as a deadline, not a load failure"
    );
    assert!(outcome.target_load_failed.is_empty());
    assert!(outcome.load_failures.iter().all(|(_, n)| *n == 0));
    assert!(
        outcome
            .summaries
            .iter()
            .all(|s| s.reason != NeuronNoCandidateReason::TargetRecordLoadFailed)
    );
}
