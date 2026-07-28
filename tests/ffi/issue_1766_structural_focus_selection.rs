//! Issue #1766: focus selection must be a structure-only weighted-random draw
//! by structural impact — it must NOT open or decode the discovery parquet, and
//! it must finish well under the "seconds bar" whether multi-GB parquet is
//! present or absent.
//!
//! The reference stall incident burned ~2h because focus choice was coupled to a
//! full-file parquet decode (~12.5 GB projected). These tests lock the fix at
//! the FFI boundary: the parquet path is never touched when choosing the focus
//! set, so a *non-existent* parquet path still yields a successful, instant
//! selection.

use neat_ai_discovery::rank_focus_neurons_internal;
use serde_json::{Value, json};
use std::time::Instant;

/// A reference-stall-shaped creature: `n_hidden` hidden neurons all feeding a single
/// output, so there are `n_hidden + 1` selectable neurons.
fn stalled_shape_creature(n_hidden: usize) -> Value {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    for i in 0..n_hidden {
        let uuid = format!("hidden-{i}");
        neurons.push(json!({
            "uuid": uuid,
            "type": "hidden",
            "squash": "IDENTITY",
            "bias": 0.0
        }));
        // Ascending weights so impacts differ across hidden neurons.
        let weight = 1.0 + f64::from(u32::try_from(i).unwrap());
        synapses.push(json!({
            "from_uuid": uuid,
            "to_uuid": "output-0",
            "weight": weight
        }));
    }
    neurons.push(json!({
        "uuid": "output-0",
        "type": "output",
        "squash": "IDENTITY",
        "bias": 0.0
    }));
    json!({
        "neurons": neurons,
        "synapses": synapses,
        "input": 0,
        "output": 1
    })
}

fn rank(creature: &Value, parquet_file: &str, cursor: u64) -> (Value, std::time::Duration) {
    let rank_input = json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "maxResults": 32,
        "focusSetSize": 6,
        "focusSelectionCursor": cursor
    })
    .to_string();
    let started = Instant::now();
    let result_json = rank_focus_neurons_internal(&rank_input).unwrap();
    let elapsed = started.elapsed();
    let result: Value = serde_json::from_str(&result_json).unwrap();
    (result, elapsed)
}

#[test]
fn focus_selection_never_opens_parquet_and_stays_under_the_seconds_bar() {
    // 16 hidden + 1 output == 17 selectable, matching the reference stall shape.
    let creature = stalled_shape_creature(16);

    // A path that does NOT exist. If the focus path decoded parquet it would
    // fail here; instead it must succeed, proving no parquet is opened.
    let missing = "/nonexistent/does-not-exist-1766.parquet";
    let (result, elapsed) = rank(&creature, missing, 0);

    assert_eq!(
        result["success"], true,
        "structure-only focus must succeed with a missing parquet path: {result:?}"
    );

    // Seconds bar: aim milliseconds, hard-fail past 5s.
    assert!(
        elapsed.as_secs() < 5,
        "focus selection must finish under the seconds bar, took {elapsed:?}"
    );

    let fs = &result["focusSelection"];
    assert!(fs.is_object(), "focusSelection must be present: {result:?}");
    let selected = fs["selected"].as_array().expect("selected array");
    assert!(!selected.is_empty(), "a focus set must be drawn");
    assert_eq!(selected.len(), 6, "focus set should honour focusSetSize");
    assert_eq!(
        fs["eligiblePoolSize"].as_u64().unwrap(),
        17,
        "17 selectable neurons (16 hidden + output)"
    );

    // No parquet was loaded, so the loading-mode observability fields are absent
    // and no record-derived removal detection runs on the focus path.
    assert!(
        result.get("loadingMode").is_none() || result["loadingMode"].is_null(),
        "loadingMode must be absent on the structure-only path: {result:?}"
    );
    assert!(
        result.get("removalCandidates").is_none() || result["removalCandidates"].is_null(),
        "removalCandidates must be absent on the focus path: {result:?}"
    );
}

#[test]
fn output_neuron_seeds_at_impact_one_and_leads_the_ranked_pool() {
    let creature = stalled_shape_creature(8);
    let (result, _elapsed) = rank(&creature, "/nonexistent/x.parquet", 0);
    assert_eq!(result["success"], true, "{result:?}");

    let neurons = result["neurons"].as_array().expect("neurons array");
    // Ranked impact-descending: output-0 (impact 1.0) leads.
    assert_eq!(
        neurons[0]["neuronUuid"], "output-0",
        "output-0 is the highest-impact target by definition"
    );
    let output_impact = neurons[0]["impact"].as_f64().unwrap();
    assert!(
        (output_impact - 1.0).abs() < 1e-6,
        "output impact seeds at 1.0, got {output_impact}"
    );
    // weightedScore mirrors the structural impact draw weight.
    let ws = neurons[0]["weightedScore"].as_f64().unwrap();
    assert!((ws - output_impact).abs() < 1e-6);

    // The dominant output neuron is drawn into the focus set.
    let selected: Vec<String> = result["focusSelection"]["selected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert!(
        selected.contains(&"output-0".to_string()),
        "highest-impact neuron must be selected, got {selected:?}"
    );
}

#[test]
fn structural_selection_is_reproducible_for_a_fixed_cursor() {
    let creature = stalled_shape_creature(16);
    let (a, _) = rank(&creature, "/nonexistent/x.parquet", 5);
    let (b, _) = rank(&creature, "/nonexistent/x.parquet", 5);
    assert_eq!(
        a["focusSelection"]["selected"], b["focusSelection"]["selected"],
        "a fixed cursor must reproduce the same focus set"
    );
}
