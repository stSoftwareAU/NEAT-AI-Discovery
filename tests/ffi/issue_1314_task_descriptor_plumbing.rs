//! Tests for Issue #1314: optional `task_descriptor` plumbing on the
//! discovery FFI inputs.
//!
//! The three FFI request structs — [`RecordDiscoveryInput`],
//! [`AnalyzeParallelInput`], and [`RankFocusNeuronsInput`] — accept an
//! optional `task_descriptor` field. The field is pure plumbing in this
//! issue: no recommendation generator reads it yet.
//!
//! Acceptance criteria checked here:
//!
//! - A payload that omits `task_descriptor` deserialises cleanly and the
//!   resulting `Option<TaskDescriptor>` is `None` — consumers treat that
//!   as [`TaskDescriptor::neutral`].
//! - A payload that supplies a structured `task_descriptor` round-trips
//!   the value verbatim.
//! - Adding the field does not break any existing payload shape (the
//!   regression guard).

use neat_ai_discovery::analysis::task_descriptor::{
    OutputSquashFamily, TargetRange, TargetTopology, TaskDescriptor,
};
use neat_ai_discovery::{AnalyzeParallelInput, RankFocusNeuronsInput, RecordDiscoveryInput};

// =============================================================================
// RecordDiscoveryInput
// =============================================================================

#[test]
fn record_discovery_input_omits_task_descriptor_to_none() {
    // RecordDiscoveryInput uses snake_case at the top level.
    let payload = r#"{
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "training_data": [],
        "temp_dir": "/tmp/x"
    }"#;
    let parsed: RecordDiscoveryInput = serde_json::from_str(payload).expect("parse");
    assert!(
        parsed.task_descriptor.is_none(),
        "absent task_descriptor must deserialise to None",
    );
    // None must collapse to neutral() at the consumer site.
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
    );
}

#[test]
fn record_discovery_input_supplied_task_descriptor_round_trips() {
    let payload = r#"{
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "training_data": [],
        "temp_dir": "/tmp/x",
        "task_descriptor": {
            "targetTopology": "OneHot",
            "targetRange": "Unit",
            "outputSquashFamily": "BoundedUnipolar",
            "numOutputs": 5
        }
    }"#;
    let parsed: RecordDiscoveryInput = serde_json::from_str(payload).expect("parse");
    let descriptor = parsed.task_descriptor.expect("task_descriptor present");
    assert_eq!(descriptor.target_topology, TargetTopology::OneHot);
    assert_eq!(descriptor.target_range, TargetRange::Unit);
    assert_eq!(
        descriptor.output_squash_family,
        OutputSquashFamily::BoundedUnipolar
    );
    assert_eq!(descriptor.num_outputs, 5);
}

// =============================================================================
// AnalyzeParallelInput
// =============================================================================

#[test]
fn analyze_parallel_input_omits_task_descriptor_to_none() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusNeurons": []
    }"#;
    let parsed: AnalyzeParallelInput = serde_json::from_str(payload).expect("parse");
    assert!(
        parsed.task_descriptor.is_none(),
        "absent task_descriptor must deserialise to None",
    );
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
    );
}

#[test]
fn analyze_parallel_input_supplied_task_descriptor_round_trips() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusNeurons": [],
        "taskDescriptor": {
            "targetTopology": "Margin",
            "targetRange": "SignedUnit",
            "outputSquashFamily": "BoundedBipolar",
            "numOutputs": 1
        }
    }"#;
    let parsed: AnalyzeParallelInput = serde_json::from_str(payload).expect("parse");
    let descriptor = parsed.task_descriptor.expect("task_descriptor present");
    assert_eq!(descriptor.target_topology, TargetTopology::Margin);
    assert_eq!(descriptor.target_range, TargetRange::SignedUnit);
    assert_eq!(
        descriptor.output_squash_family,
        OutputSquashFamily::BoundedBipolar
    );
    assert_eq!(descriptor.num_outputs, 1);
}

#[test]
fn analyze_parallel_input_accepts_neutral_descriptor() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusNeurons": [],
        "taskDescriptor": {
            "targetTopology": "Unknown",
            "targetRange": "Unbounded",
            "outputSquashFamily": "Any",
            "numOutputs": 0
        }
    }"#;
    let parsed: AnalyzeParallelInput = serde_json::from_str(payload).expect("parse");
    let descriptor = parsed.task_descriptor.expect("task_descriptor present");
    assert_eq!(descriptor, TaskDescriptor::neutral());
}

// =============================================================================
// RankFocusNeuronsInput
// =============================================================================

#[test]
fn rank_focus_neurons_input_omits_task_descriptor_to_none() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1}
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    assert!(
        parsed.task_descriptor.is_none(),
        "absent task_descriptor must deserialise to None",
    );
    assert_eq!(
        parsed.task_descriptor.unwrap_or_default(),
        TaskDescriptor::neutral(),
    );
}

#[test]
fn rank_focus_neurons_input_supplied_task_descriptor_round_trips() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "taskDescriptor": {
            "targetTopology": "Independent",
            "targetRange": "Unbounded",
            "outputSquashFamily": "Unbounded",
            "numOutputs": 3
        }
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    let descriptor = parsed.task_descriptor.expect("task_descriptor present");
    assert_eq!(descriptor.target_topology, TargetTopology::Independent);
    assert_eq!(descriptor.target_range, TargetRange::Unbounded);
    assert_eq!(
        descriptor.output_squash_family,
        OutputSquashFamily::Unbounded
    );
    assert_eq!(descriptor.num_outputs, 3);
}

// =============================================================================
// Back-compat / regression guard
// =============================================================================

#[test]
fn payload_without_task_descriptor_still_parses_all_existing_fields() {
    // Verify that adding the optional field has not perturbed how an
    // existing payload (carrying every other previously-supported field)
    // is parsed. This is the "no behaviour change" guard from the issue.
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusNeurons": ["a", "b"],
        "maxSynapseCandidates": 16,
        "maxNeuronCandidates": 8,
        "analysisDeadlineMs": 1000,
        "randomSeed": 42,
        "temperature": 1.25,
        "maxAnalysisMemoryMb": 512,
        "maxDiscoveryWallClockMinutes": 5,
        "costName": "MSE"
    }"#;
    let parsed: AnalyzeParallelInput = serde_json::from_str(payload).expect("parse");
    assert_eq!(parsed.focus_neurons, vec!["a".to_string(), "b".to_string()]);
    assert_eq!(parsed.max_synapse_candidates, Some(16));
    assert_eq!(parsed.max_neuron_candidates, Some(8));
    assert_eq!(parsed.analysis_deadline_ms, Some(1000));
    assert_eq!(parsed.random_seed, Some(42));
    assert!((parsed.temperature - 1.25).abs() < 1e-6);
    assert_eq!(parsed.max_analysis_memory_mb, Some(512));
    assert_eq!(parsed.max_discovery_wall_clock_minutes, Some(5));
    assert_eq!(parsed.cost_name.as_deref(), Some("MSE"));
    assert!(parsed.task_descriptor.is_none());
}
