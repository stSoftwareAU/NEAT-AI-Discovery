//! Production-network discovery regression harness (Issue #1741).
//!
//! This is the repeatable harness milestone #1736 needs to (a) prove the
//! accepted-improvement rate on the production topology objectively and (b)
//! catch future regressions in discovery yield on that topology.
//!
//! The milestone audits concluded a **plateau**, not a bug: the focus/impact
//! math is sound (#1738) and the acceptance gain floors are correctly scaled
//! (#1740); the network is genuinely saturated (#1737). The recorded plateau
//! baseline is 2 accepted runs across the 40-run production window — a 0.05
//! accepted-run rate. The mandated case
//! `accepted_improvement_rate_meets_threshold` therefore asserts the rate has
//! **not regressed below** that recorded baseline (and reports it), rather than
//! asserting the parent's not-yet-met > 50% success criterion. If a future
//! sibling fix lifts yield, `DiscoveryBatchReport::meets_success_threshold`
//! flips the same harness to the success criterion — see
//! `majority_accepted_batch_meets_success` below.
//!
//! Failure detection (per the issue): a broken/malformed fixture, a
//! snapshot-loading failure, or a panicking harness fails on every PR via
//! `cargo test`; a drop in accepted yield fails the threshold assertion.
//!
//! Naming note: the issue names this file and its threshold case with a private
//! deployment token. The shipped-source private-name guards (Issues #1724/#1725)
//! forbid that token in `.rs` files and file names, so the file, module, and
//! case use the concept-level "production discovery" naming instead.

use std::path::PathBuf;

use neat_ai_discovery::analysis::production_discovery_regression::{
    DiscoveryRunBatch, PLATEAU_ACCEPTED_RUN_RATE, compute_batch_report,
};
use neat_ai_discovery::export::{
    ExportOptions, VisualisationSnapshot, export_visualisation_snapshot,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use std::fs::File;
use std::io::BufReader;
use tempfile::tempdir;

/// Path to the committed production-representative run-batch fixture.
fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/production_discovery_regression/run_batch.json")
}

fn load_batch() -> DiscoveryRunBatch {
    DiscoveryRunBatch::from_json_file(&fixture_path())
        .expect("committed run-batch fixture must load and parse")
}

/// The mandated harness case (Issue #1741).
///
/// Runs the shipped acceptance logic over the committed production-representative
/// fixture batch and fails if the accepted-improvement rate drops below the
/// recorded plateau baseline.
#[test]
fn accepted_improvement_rate_meets_threshold() {
    let batch = load_batch();
    let report = compute_batch_report(&batch);

    // Always report the rate so the harness output is the close evidence for
    // #1736 even when the assertion passes.
    println!("{}", report.summary_line());

    assert!(
        !report.regressed_below_baseline(PLATEAU_ACCEPTED_RUN_RATE),
        "accepted-improvement rate {:.4} regressed below the recorded plateau baseline {:.4} \
         — discovery yield on the production topology has dropped (see rejection-diagnosis-1737.md)",
        report.accepted_run_rate,
        PLATEAU_ACCEPTED_RUN_RATE,
    );

    // The recorded plateau: the milestone audits (#1738, #1740) concluded the
    // network is saturated, so the parent's > 50% success criterion is not yet
    // met. Pin that so a fixture edit that silently inflates yield is noticed.
    assert!(
        !report.meets_success_threshold(),
        "accepted rate {:.4} unexpectedly meets the > 50% success criterion — if discovery yield \
         genuinely improved, update the recorded baseline and close #1736 on the success path",
        report.accepted_run_rate,
    );
}

/// The committed fixture matches the recorded plateau (Issue #1737): 40 runs,
/// exactly 2 accepting a harmful-neuron removal.
#[test]
fn fixture_matches_recorded_plateau() {
    let batch = load_batch();
    let report = compute_batch_report(&batch);

    assert_eq!(
        report.total_runs, 40,
        "recorded window is 40 discovery runs"
    );
    assert_eq!(
        report.accepted_runs, 2,
        "recorded plateau is 2 accepted runs"
    );
    assert!(
        (report.accepted_run_rate - PLATEAU_ACCEPTED_RUN_RATE).abs() < 1e-12,
        "fixture rate {:.4} must equal the recorded baseline {:.4}",
        report.accepted_run_rate,
        PLATEAU_ACCEPTED_RUN_RATE,
    );
    // Every accepted candidate is a positive-realised-delta removal (#1737).
    assert_eq!(report.total_accepted_candidates, 2);
}

/// If discovery yield ever lifts to a strict majority, the *same* harness flips
/// to the parent's success criterion — proving the harness is not hard-wired to
/// the plateau conclusion.
#[test]
fn majority_accepted_batch_meets_success() {
    let batch = load_batch();
    let mut runs = batch.runs;
    // Duplicate the two real accepted runs until they form a strict majority.
    let accepted: Vec<_> = runs
        .iter()
        .filter(|r| r.has_accepted_improvement())
        .cloned()
        .collect();
    assert_eq!(accepted.len(), 2);
    while runs.iter().filter(|r| r.has_accepted_improvement()).count() * 2 <= runs.len() {
        runs.push(accepted[0].clone());
    }
    let lifted = DiscoveryRunBatch {
        description: "hypothetical lifted-yield batch".to_string(),
        runs,
    };
    let report = compute_batch_report(&lifted);
    assert!(
        report.meets_success_threshold(),
        "a majority-accepted batch must satisfy the parent success criterion",
    );
}

/// A yield collapse (a regression that stops discovery accepting anything) is
/// detected as a drop below the plateau baseline.
#[test]
fn yield_collapse_is_flagged_below_baseline() {
    let batch = load_batch();
    // Keep only the empty runs — models a regression that breaks acceptance.
    let runs: Vec<_> = batch
        .runs
        .into_iter()
        .filter(|r| !r.has_accepted_improvement())
        .collect();
    let collapsed = DiscoveryRunBatch {
        description: "collapsed".to_string(),
        runs,
    };
    let report = compute_batch_report(&collapsed);
    assert_eq!(report.accepted_runs, 0);
    assert!(report.regressed_below_baseline(PLATEAU_ACCEPTED_RUN_RATE));
}

/// Build a small production-representative creature that exercises an
/// aggregation squash (MAXIMUM) — the sub-graph shape the milestone lead
/// hypothesis (#1738) centred on.
fn representative_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "agg-max".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "MAXIMUM".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "agg-max".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "agg-max".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "agg-max".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 2.0,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

fn representative_records() -> Vec<DiscoverRecord> {
    vec![
        DiscoverRecord::new(0, "input-0".to_string(), Some(1.0), 1.0, vec![0.0]),
        DiscoverRecord::new(0, "input-1".to_string(), Some(2.0), 2.0, vec![0.0]),
        // MAXIMUM(1.0, 2.0) = 2.0
        DiscoverRecord::new(0, "agg-max".to_string(), Some(2.0), 2.0, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(4.0), 4.0, vec![0.05]),
        DiscoverRecord::new(1, "input-0".to_string(), Some(0.5), 0.5, vec![0.0]),
        DiscoverRecord::new(1, "input-1".to_string(), Some(0.25), 0.25, vec![0.0]),
        // MAXIMUM(0.5, 0.25) = 0.5
        DiscoverRecord::new(1, "agg-max".to_string(), Some(0.5), 0.5, vec![0.2]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
    ]
}

/// The production-representative snapshot fixture loads through the real
/// snapshot pipeline (`src/export/snapshot.rs`). This is the harness's
/// snapshot-loading backstop from the issue's Failure Detection section: a
/// broken fixture or a panicking snapshot path fails here on every PR.
#[test]
fn snapshot_fixture_loads_through_export_pipeline() {
    let dir = tempdir().expect("temp dir");
    let parquet_path = dir.path().join("records.parquet");
    let snapshot_path = dir.path().join("snapshot.json");

    write_records_to_parquet(parquet_path.to_str().unwrap(), &representative_records())
        .expect("write parquet");

    let creature = representative_creature();
    let stats = export_visualisation_snapshot(
        parquet_path.to_str().unwrap(),
        &creature,
        snapshot_path.to_str().unwrap(),
        &ExportOptions::default(),
    )
    .expect("snapshot export over the production-representative fixture must succeed");

    assert_eq!(stats.obs_count, 2);
    // Distinct neuron uuids recorded: input-0, input-1, agg-max, output-0.
    assert_eq!(stats.neuron_count, 4);

    // Deserialise the emitted snapshot to prove it round-trips.
    let file = File::open(&snapshot_path).expect("open snapshot");
    let snapshot: VisualisationSnapshot =
        serde_json::from_reader(BufReader::new(file)).expect("snapshot JSON must deserialise");
    assert!(
        snapshot
            .creature
            .neurons
            .iter()
            .any(|n| n.uuid == "agg-max"),
        "snapshot must retain the aggregation-squash neuron",
    );
    assert!(
        snapshot.recording.neurons.contains_key("agg-max"),
        "snapshot recording must include the aggregation-squash neuron",
    );
}
