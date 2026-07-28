//! Issue #1767: removal triage must run on the **near-opposite** axis to focus
//! (low structural contribution vs complexity savings) and must NOT open or
//! decode the discovery parquet during focus selection.
//!
//! Focus draws HIGH structural impact (#1766); removal flags hidden neurons
//! whose LOW structural contribution is outweighed by pruning savings. These
//! tests lock the fix at the FFI boundary: a *non-existent* parquet path still
//! yields removal candidates, proving triage is structure-only.

use neat_ai_discovery::rank_focus_neurons_internal;
use serde_json::{Value, json};
use std::time::Instant;

/// A creature with one high-impact hidden neuron driving the output plus a dead
/// hidden chain that never reaches the output. The dead chain has zero
/// structural impact, so those neurons are the low-contribution removal
/// candidates; the output and the driving hidden neuron are not.
fn creature_with_dead_branch() -> Value {
    json!({
        "neurons": [
            {"uuid": "i0", "type": "input", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "h_main", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "o0", "type": "output", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "d0", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "d1", "type": "hidden", "squash": "IDENTITY", "bias": 0.0},
            {"uuid": "d2", "type": "hidden", "squash": "IDENTITY", "bias": 0.0}
        ],
        "synapses": [
            {"from_uuid": "i0", "to_uuid": "h_main", "weight": 1.0},
            {"from_uuid": "h_main", "to_uuid": "o0", "weight": 1.0},
            {"from_uuid": "i0", "to_uuid": "d0", "weight": 0.5},
            {"from_uuid": "d0", "to_uuid": "d1", "weight": 0.5},
            {"from_uuid": "d1", "to_uuid": "d2", "weight": 0.5}
        ],
        "input": 1,
        "output": 1
    })
}

fn rank(creature: &Value, parquet_file: &str) -> (Value, std::time::Duration) {
    // costOfGrowth 1e-4 so the dead branch's boosted savings clear the default
    // 1e-5 remove-low-impact noise floor.
    let rank_input = json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "maxResults": 32,
        "focusSetSize": 4,
        "focusSelectionCursor": 0,
        "costOfGrowth": 1e-4
    })
    .to_string();
    let started = Instant::now();
    let result_json = rank_focus_neurons_internal(&rank_input).unwrap();
    let elapsed = started.elapsed();
    let result: Value = serde_json::from_str(&result_json).unwrap();
    (result, elapsed)
}

fn removal_uuids(result: &Value) -> Vec<String> {
    result["removalCandidates"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .map(|c| c["neuronUuid"].as_str().unwrap().to_string())
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn removal_triage_runs_structure_only_with_a_missing_parquet() {
    let creature = creature_with_dead_branch();
    // A path that does NOT exist: if triage decoded parquet it would fail here.
    let missing = "/nonexistent/does-not-exist-1767.parquet";
    let (result, elapsed) = rank(&creature, missing);

    assert_eq!(
        result["success"], true,
        "structure-only removal triage must succeed with a missing parquet: {result:?}"
    );
    // Seconds bar: aim milliseconds, hard-fail past 5s.
    assert!(
        elapsed.as_secs() < 5,
        "removal triage must finish under the seconds bar, took {elapsed:?}"
    );

    let removals = removal_uuids(&result);
    assert!(
        !removals.is_empty(),
        "the dead low-impact hidden branch must yield removal candidates: {result:?}"
    );
    // Exactly the dead-branch hidden neurons — near-opposite of the high-impact
    // focus draw.
    for uuid in ["d0", "d1", "d2"] {
        assert!(
            removals.contains(&uuid.to_string()),
            "dead hidden neuron {uuid} must be a removal candidate, got {removals:?}"
        );
    }
}

#[test]
fn high_impact_and_output_neurons_are_never_removal_candidates() {
    let creature = creature_with_dead_branch();
    let (result, _) = rank(&creature, "/nonexistent/x.parquet");
    let removals = removal_uuids(&result);

    assert!(
        !removals.contains(&"o0".to_string()),
        "the output neuron must never be a removal candidate: {removals:?}"
    );
    assert!(
        !removals.contains(&"h_main".to_string()),
        "the high-impact driving neuron must never be a removal candidate: {removals:?}"
    );
}

#[test]
fn removal_candidates_defer_activation_weighted_fields_to_analysis() {
    let creature = creature_with_dead_branch();
    let (result, _) = rank(&creature, "/nonexistent/x.parquet");
    let candidates = result["removalCandidates"]
        .as_array()
        .expect("removalCandidates array");

    for c in candidates {
        // Record-derived gates run later in the analysis phase, not at focus time.
        assert_eq!(
            c["meanActivation"].as_f64().unwrap(),
            0.0,
            "meanActivation is record-derived and must be unmeasured on the focus path"
        );
        assert_eq!(
            c["activationWeightedImpact"].as_f64().unwrap(),
            0.0,
            "activationWeightedImpact is deferred to analysis"
        );
        // Structural contribution (impact) is low — the removal axis.
        assert!(
            c["impact"].as_f64().unwrap() < 1e-4,
            "removal candidates carry LOW structural impact"
        );
        assert!(c["removalSavings"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn constant_neuron_removals_stay_off_the_focus_path() {
    // Constant-neuron folding needs recorded activation variance, so it must not
    // ride the parquet-free focus path.
    let creature = creature_with_dead_branch();
    let (result, _) = rank(&creature, "/nonexistent/x.parquet");
    assert!(
        result.get("constantNeuronRemovals").is_none()
            || result["constantNeuronRemovals"].is_null(),
        "constantNeuronRemovals must be absent on the focus path: {result:?}"
    );
}
