//! Tests for Issue #1407: focus selection bills against the shared discovery
//! deadline instead of opening a fresh independent window.
//!
//! The FFI contract change is the new optional `analysisDeadlineMs` field on
//! [`RankFocusNeuronsInput`] — the same absolute discovery deadline already
//! accepted by `analyzeParallel`. These tests guard the deserialisation
//! contract:
//!
//! - An omitted `analysisDeadlineMs` deserialises to `None` (backwards
//!   compatible — legacy budget-only behaviour).
//! - A supplied `analysisDeadlineMs` round-trips verbatim so the Rust layer can
//!   thread it into `FocusDeadline`.

use neat_ai_discovery::RankFocusNeuronsInput;

#[test]
fn rank_focus_input_omits_analysis_deadline_to_none() {
    // RankFocusNeuronsInput uses camelCase field names.
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 0, "output": 0}
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    assert!(
        parsed.analysis_deadline_ms.is_none(),
        "absent analysisDeadlineMs must deserialise to None (budget-only behaviour)",
    );
}

#[test]
fn rank_focus_input_supplied_analysis_deadline_round_trips() {
    // An absolute ms-since-epoch deadline well above the year-2000 threshold.
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 0, "output": 0},
        "analysisDeadlineMs": 1893456000000
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    assert_eq!(
        parsed.analysis_deadline_ms,
        Some(1_893_456_000_000),
        "analysisDeadlineMs must round-trip so it can be shared with analysis",
    );
}

#[test]
fn rank_focus_input_accepts_relative_deadline_duration() {
    // A small value (below year-2000-in-ms) is a relative duration; the field
    // still round-trips and the Rust layer interprets it via the shared epoch
    // heuristic.
    let payload = r#"{
        "parquetFile": "/tmp/x.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 0, "output": 0},
        "analysisDeadlineMs": 120000
    }"#;
    let parsed: RankFocusNeuronsInput = serde_json::from_str(payload).expect("parse");
    assert_eq!(parsed.analysis_deadline_ms, Some(120_000));
}
