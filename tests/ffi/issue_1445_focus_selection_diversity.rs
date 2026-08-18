//! Tests for Issue #1445: focus-selection roulette collapses to a single
//! neuron on plateaued mature networks.
//!
//! On a mature, plateaued creature one dominant neuron can hold ~98% of the
//! focus-selection roulette weight, so the weighted roulette becomes
//! single-target and discovery revisits the same neighbourhood every pass.
//! These tests guard the FFI contract for the diversity-aware focus selection:
//!
//! - The new optional `epochsSinceLastAcceptedCandidate` / `focusSetSize`
//!   fields deserialise (backwards compatible when omitted).
//! - `rank_focus_neurons_internal` surfaces a `focusSelection` block with the
//!   concentration ratios and a diverse selected set, and each ranked neuron
//!   carries its `weightedScore`.
//!
//! Issue #1766 supersedes the record-derived exploit/explore selection these
//! tests originally guarded: the FFI focus path is now a structure-only
//! weighted-random draw by structural impact that never opens the parquet. The
//! behavioural assertions below have been updated to that contract; the input
//! deserialisation tests are unchanged.

use neat_ai_discovery::{
    RankFocusNeuronsInput, rank_focus_neurons_internal, record_discovery_internal,
};
use serde_json::{Value, json};
use tempfile::TempDir;

#[test]
fn rank_focus_input_new_fields_default_to_none() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1}
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    assert!(parsed.epochs_since_last_accepted_candidate.is_none());
    assert!(parsed.focus_set_size.is_none());
}

#[test]
fn rank_focus_input_new_fields_round_trip() {
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "epochsSinceLastAcceptedCandidate": 42,
        "focusSetSize": 6
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    assert_eq!(parsed.epochs_since_last_accepted_candidate, Some(42));
    assert_eq!(parsed.focus_set_size, Some(6));
}

/// Build a plateau-shaped creature: `n_hidden` hidden neurons all feeding a
/// single output, where `hidden-0` dominates via a far larger weight and error.
/// Returns `(creature_json, parquet_file_path, temp_dir_guard)`.
fn build_plateau_parquet(n_hidden: usize) -> (Value, String, TempDir) {
    let mut neurons = Vec::new();
    let mut synapses = Vec::new();
    let mut neuron_data = Vec::new();

    for i in 0..n_hidden {
        let uuid = format!("hidden-{i}");
        neurons.push(json!({
            "uuid": uuid,
            "type": "hidden",
            "squash": "IDENTITY",
            "bias": 0.0
        }));
        // hidden-0 dominates: huge weight to the output and a large error.
        let weight = if i == 0 { 1.0e6 } else { 1.0 };
        synapses.push(json!({
            "from_uuid": uuid,
            "to_uuid": "output-0",
            "weight": weight
        }));
        // hidden-0 carries essentially all the recorded error; the rest are
        // near-zero so the raw roulette weight collapses onto one neuron.
        let error = if i == 0 { 0.9 } else { 0.0 };
        let _ = i;
        let activation = 0.5;
        neuron_data.push(json!({
            "neuron_uuid": uuid,
            "activation": activation,
            "value": activation,
            "errors": [error]
        }));
    }
    neurons.push(json!({
        "uuid": "output-0",
        "type": "output",
        "squash": "IDENTITY",
        "bias": 0.0
    }));
    neuron_data.push(json!({
        "neuron_uuid": "output-0",
        "activation": 0.5,
        "value": 0.5,
        "errors": [0.2]
    }));

    let creature = json!({
        "neurons": neurons,
        "synapses": synapses,
        "input": 1,
        "output": 1
    });

    let temp_dir = TempDir::new().unwrap();
    let temp_path = temp_dir.path();
    let record_input = json!({
        "creature": creature,
        "training_data": [{
            "input": [0.0],
            "output": [0.5],
            "neuron_data": neuron_data
        }],
        "temp_dir": temp_path.to_str().unwrap()
    })
    .to_string();

    let record_output_json = record_discovery_internal(&record_input).unwrap();
    let record_output: Value = serde_json::from_str(&record_output_json).unwrap();
    assert_eq!(
        record_output["success"], true,
        "failed to record discovery data: {record_output:?}"
    );
    let parquet_file = temp_path
        .join(record_output["file"].as_str().unwrap())
        .to_str()
        .unwrap()
        .to_string();

    (creature, parquet_file, temp_dir)
}

#[test]
fn focus_selection_is_surfaced_with_concentration_metrics() {
    let (creature, parquet_file, _guard) = build_plateau_parquet(12);

    let rank_input = json!({
        "parquetFile": parquet_file,
        "creature": creature,
        "maxResults": 12,
        "focusSetSize": 6
    })
    .to_string();

    let result_json = rank_focus_neurons_internal(&rank_input).unwrap();
    let result: Value = serde_json::from_str(&result_json).unwrap();
    assert_eq!(result["success"], true, "rank failed: {result:?}");

    // Each ranked neuron now carries its weighted score.
    let neurons = result["neurons"].as_array().expect("neurons array");
    assert!(!neurons.is_empty());
    for n in neurons {
        assert!(
            n["weightedScore"].is_number(),
            "every ranked neuron must carry weightedScore: {n:?}"
        );
    }

    let fs = &result["focusSelection"];
    assert!(fs.is_object(), "focusSelection must be present: {result:?}");

    let raw = fs["rawWeightConcentrationRatio"].as_f64().unwrap();
    let eff = fs["weightConcentrationRatio"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&raw));
    assert!((0.0..=1.0).contains(&eff));

    // Issue #1766: the raw ratio is now the concentration of the *structural
    // impact* weights (spread between the dominant hidden neuron and the
    // output-0 seed at 1.0), not the record-derived roulette. It stays a
    // well-formed positive ratio.
    assert!(
        raw > 0.0,
        "expected a positive raw structural-impact concentration, got {raw}"
    );

    // Issue #1766: every focus slot is an impact-weighted draw, so all picks
    // report as exploitation and the exploit/explore split of #1662 no longer
    // applies. The diagnostics stay populated and internally consistent.
    let exploitation = fs["exploitationCount"].as_u64().unwrap();
    let exploration = fs["explorationCount"].as_u64().unwrap();
    let selected: Vec<String> = fs["selected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(exploitation + exploration, selected.len() as u64);
    assert!(fs["eligiblePoolSize"].as_u64().unwrap() >= selected.len() as u64);
    assert!(fs["explorationCursor"].is_number());
    assert!(fs["cumulativeCoverage"].is_number());
    let _ = eff;

    // The selected set is diverse: >=3 distinct focus targets.
    let distinct: std::collections::HashSet<&String> = selected.iter().collect();
    assert!(
        distinct.len() >= 3,
        "expected >=3 distinct focus targets, got {selected:?}"
    );
}

#[test]
fn cursor_reseeds_structural_weighted_random_draw() {
    // Issue #1766: supersedes the #1662 drought/exploit-explore rotation. Focus
    // selection is now a structure-only weighted-random draw seeded by the
    // monotonic per-creature cursor, so advancing the cursor reseeds the draw
    // and sweeps a fresh tail while still landing on the high-impact neurons.
    let (creature, parquet_file, _guard) = build_plateau_parquet(40);

    let select_for = |cursor: u64| -> Value {
        let rank_input = json!({
            "parquetFile": parquet_file,
            "creature": creature,
            "maxResults": 40,
            "focusSetSize": 6,
            "focusSelectionCursor": cursor
        })
        .to_string();
        let result_json = rank_focus_neurons_internal(&rank_input).unwrap();
        let result: Value = serde_json::from_str(&result_json).unwrap();
        assert_eq!(result["success"], true, "rank failed: {result:?}");
        result["focusSelection"].clone()
    };

    let selected = |fs: &Value| -> Vec<String> {
        fs["selected"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect()
    };

    let fs_a = select_for(0);
    let fs_b = select_for(1);
    let fs_c = select_for(2);

    // A fixed cursor is reproducible; different cursors reseed the draw.
    let fs_a_again = select_for(0);
    assert_eq!(
        selected(&fs_a),
        selected(&fs_a_again),
        "a fixed cursor must reproduce the same focus set"
    );

    let pass_a = selected(&fs_a);
    let pass_b = selected(&fs_b);
    let pass_c = selected(&fs_c);
    assert_ne!(pass_a, pass_b, "advancing the cursor must change the set");
    assert_ne!(pass_b, pass_c, "advancing the cursor must change the set");

    let union: std::collections::HashSet<String> =
        pass_a.into_iter().chain(pass_b).chain(pass_c).collect();
    assert!(
        union.len() >= 3,
        "three cursor positions must cover >=3 distinct targets, got {union:?}"
    );
}

#[test]
fn rank_focus_input_focus_selection_cursor_round_trips() {
    // Issue #1662: the monotonic exploration cursor deserialises and defaults to
    // None (backwards compatible).
    let without = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1}
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(without).expect("parse");
    assert!(parsed.focus_selection_cursor.is_none());

    let with = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusSelectionCursor": 123
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(with).expect("parse");
    assert_eq!(parsed.focus_selection_cursor, Some(123));
}
