//! Regression test for Issue #2222 (part of #2170): a deserialised
//! `ModuleOutcomeTracker` bypasses the invariants `record()`,
//! `record_soft_failures()` and `apply_decay()` keep, so `success_rate()` must
//! be total and bounded in [0, 1] for any `ModuleStats` value.
//!
//! Against the pre-fix code, `successes > attempts` panicked with
//! `attempt to subtract with overflow` in a debug build (and wrapped to ~4.29e9
//! failures in release), and a negative `soft_failures` made the rate `+inf`.

use neat_ai_discovery::analysis::module_weights::{ModuleOutcomeTracker, ModuleStats};
use neat_ai_discovery::analysis::synapse::add_synapse_gating::{
    ADD_SYNAPSE_MODULE_NAME, DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
    should_skip_add_synapse_by_outcome,
};

/// Deserialises a tracker holding a single `add-synapse` entry, as a host
/// would hand it across the FFI boundary.
fn tracker_from_json(attempts: u32, successes: u32, soft_failures: &str) -> ModuleOutcomeTracker {
    let json = format!(
        r#"{{"modules":{{"{ADD_SYNAPSE_MODULE_NAME}":{{"attempts":{attempts},"successes":{successes},"candidatesProduced":0,"softFailures":{soft_failures}}}}}}}"#
    );
    serde_json::from_str(&json).expect("tracker JSON must deserialise")
}

fn assert_bounded(rate: f64, context: &str) {
    assert!(
        rate.is_finite(),
        "{context}: rate must be finite, got {rate}"
    );
    assert!(
        (0.0..=1.0).contains(&rate),
        "{context}: rate must lie in [0, 1], got {rate}"
    );
}

const THRESHOLDS: [f64; 5] = [
    0.001,
    0.01,
    DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD,
    0.5,
    0.99,
];

#[test]
fn successes_above_attempts_matches_the_well_formed_record() {
    let corrupt = tracker_from_json(10, 20, "0.0");
    let well_formed = tracker_from_json(10, 10, "0.0");

    let corrupt_rate = corrupt.stats(ADD_SYNAPSE_MODULE_NAME).success_rate();
    let well_formed_rate = well_formed.stats(ADD_SYNAPSE_MODULE_NAME).success_rate();
    assert_bounded(corrupt_rate, "attempts 10, successes 20");
    assert_eq!(corrupt_rate, well_formed_rate);

    for threshold in THRESHOLDS {
        assert_eq!(
            should_skip_add_synapse_by_outcome(&corrupt, threshold),
            should_skip_add_synapse_by_outcome(&well_formed, threshold),
            "gate decision must match the well-formed record at threshold {threshold}"
        );
    }
    // A perfect record must never close the gate at the default threshold.
    assert!(!should_skip_add_synapse_by_outcome(&corrupt, 0.01));
}

#[test]
fn negative_soft_failures_are_ignored() {
    let corrupt = tracker_from_json(10, 10, "-12.0");
    let well_formed = tracker_from_json(10, 10, "0.0");

    let rate = corrupt.stats(ADD_SYNAPSE_MODULE_NAME).success_rate();
    assert_bounded(rate, "soft_failures -12.0");
    assert_eq!(
        rate,
        well_formed.stats(ADD_SYNAPSE_MODULE_NAME).success_rate()
    );
    for threshold in THRESHOLDS {
        assert_eq!(
            should_skip_add_synapse_by_outcome(&corrupt, threshold),
            should_skip_add_synapse_by_outcome(&well_formed, threshold),
            "gate decision must match the well-formed record at threshold {threshold}"
        );
    }
}

#[test]
fn huge_soft_failures_stay_finite_and_close_the_gate() {
    let tracker = tracker_from_json(10, 10, "1.7976931348623157e308");
    let stats = tracker.stats(ADD_SYNAPSE_MODULE_NAME);
    assert_eq!(stats.soft_failures, f64::MAX);

    let rate = stats.success_rate();
    assert_bounded(rate, "soft_failures f64::MAX");
    // f64::MAX soft failures is a (finite) overwhelming failure count, so the
    // rate collapses towards 0 and the outcome gate closes.
    assert!(rate < 0.01, "got {rate}");
    assert!(should_skip_add_synapse_by_outcome(&tracker, 0.01));
}

#[test]
fn non_finite_soft_failures_are_ignored() {
    let well_formed = ModuleStats {
        attempts: 10,
        successes: 10,
        ..ModuleStats::default()
    };
    for soft_failures in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let stats = ModuleStats {
            soft_failures,
            ..well_formed.clone()
        };
        let rate = stats.success_rate();
        assert_bounded(rate, &format!("soft_failures {soft_failures}"));
        assert_eq!(
            rate,
            well_formed.success_rate(),
            "soft_failures {soft_failures}"
        );
    }

    // With no attempts, a NaN must not skip the 0.5 no-data prior.
    let empty_nan = ModuleStats {
        soft_failures: f64::NAN,
        ..ModuleStats::default()
    };
    assert_eq!(empty_nan.success_rate(), 0.5);
}

// ---------------------------------------------------------------------------
// Issue #2223 (part of #2170): the FFI boundary rejects a corrupt tracker as
// `data_validation` before any analysis runs, rather than computing a gate.
// ---------------------------------------------------------------------------

use neat_ai_discovery::{
    DiscoveryErrorKind, MODULE_NAME_DETAIL_MAX_CHARS, analyze_parallel_internal,
    validate_module_outcome_tracker, validate_module_stats,
};

/// An `analyze_parallel` payload for a minimal forward-only creature. Both
/// phases are disabled so a well-formed payload completes without Parquet data.
fn analyze_parallel_payload(tracker_json: Option<&str>) -> String {
    let tracker = tracker_json.map_or(String::new(), |t| format!(r#","moduleOutcomeTracker":{t}"#));
    format!(
        r#"{{
            "parquetFile": "/tmp/issue-2223-does-not-exist.parquet",
            "creature": {{
                "neurons": [
                    {{"uuid": "input-0", "type": "input", "squash": "IDENTITY"}},
                    {{"uuid": "output-0", "type": "output", "squash": "LOGISTIC", "bias": 0.0}}
                ],
                "synapses": [{{"fromUUID": "input-0", "toUUID": "output-0", "weight": 0.5}}],
                "input": 1,
                "output": 1
            }},
            "focusNeurons": ["output-0"],
            "includeSynapseAnalysis": false,
            "includeNeuronAnalysis": false{tracker}
        }}"#
    )
}

fn tracker_json(name: &str, attempts: u32, successes: u32, soft_failures: &str) -> String {
    format!(
        r#"{{"modules":{{"{name}":{{"attempts":{attempts},"successes":{successes},"candidatesProduced":0,"softFailures":{soft_failures}}}}}}}"#
    )
}

fn analyze(tracker_json: Option<&str>) -> serde_json::Value {
    let result = analyze_parallel_internal(&analyze_parallel_payload(tracker_json))
        .expect("analyze_parallel_internal must return a JSON payload");
    serde_json::from_str(&result).expect("response must be valid JSON")
}

fn assert_rejected(tracker: &str, field: &str) -> String {
    let parsed = analyze(Some(tracker));
    assert_eq!(parsed["success"], false, "must reject {tracker}: {parsed}");
    assert_eq!(
        parsed["errorKind"], "data_validation",
        "must classify {tracker} as data_validation: {parsed}"
    );
    assert_eq!(parsed["retryable"], false, "{parsed}");
    let error = parsed["error"].as_str().expect("error must be present");
    assert!(error.contains(field), "error must name `{field}`: {error}");
    assert!(
        error.contains("Issue #2170"),
        "error must cite the issue: {error}"
    );
    error.to_string()
}

#[test]
fn analyze_parallel_rejects_successes_above_attempts() {
    assert_rejected(
        &tracker_json(ADD_SYNAPSE_MODULE_NAME, 10, 20, "0.0"),
        "successes",
    );
}

#[test]
fn analyze_parallel_rejects_negative_soft_failures() {
    assert_rejected(
        &tracker_json(ADD_SYNAPSE_MODULE_NAME, 10, 10, "-12.0"),
        "softFailures",
    );
}

#[test]
fn analyze_parallel_rejects_soft_failures_above_u32_max() {
    assert_rejected(
        &tracker_json(ADD_SYNAPSE_MODULE_NAME, 10, 10, "1.7976931348623157e308"),
        "softFailures",
    );
}

#[test]
fn analyze_parallel_accepts_a_well_formed_tracker() {
    let parsed = analyze(Some(&tracker_json(ADD_SYNAPSE_MODULE_NAME, 10, 3, "4.5")));
    assert_eq!(parsed["success"], true, "well-formed tracker: {parsed}");
    // The caller's tracker is handed back for persistence.
    assert_eq!(
        parsed["moduleOutcomeTracker"]["modules"][ADD_SYNAPSE_MODULE_NAME]["successes"], 3,
        "{parsed}"
    );
}

#[test]
fn analyze_parallel_is_unaffected_by_a_missing_tracker() {
    let parsed = analyze(None);
    assert_eq!(parsed["success"], true, "absent tracker: {parsed}");
}

#[test]
fn validate_module_stats_accepts_the_boundaries() {
    for (attempts, successes, soft_failures) in [
        (0, 0, 0.0),
        (10, 10, 0.0),
        (u32::MAX, u32::MAX, f64::from(u32::MAX)),
    ] {
        let stats = ModuleStats {
            attempts,
            successes,
            soft_failures,
            ..ModuleStats::default()
        };
        validate_module_stats("m", &stats).unwrap_or_else(|e| {
            panic!("({attempts}, {successes}, {soft_failures}) must be accepted: {e}")
        });
    }
}

#[test]
fn validate_module_stats_rejects_non_finite_soft_failures() {
    for soft_failures in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let stats = ModuleStats {
            attempts: 10,
            successes: 10,
            soft_failures,
            ..ModuleStats::default()
        };
        let err = validate_module_stats("add-synapse", &stats)
            .expect_err("a non-finite soft_failures must be rejected");
        assert_eq!(
            err.error_kind(),
            DiscoveryErrorKind::DataValidation,
            "{soft_failures}"
        );
        assert!(err.to_string().contains("softFailures"), "{err}");
    }
}

#[test]
fn validate_module_stats_rejects_soft_failures_just_above_u32_max() {
    let stats = ModuleStats {
        soft_failures: f64::from(u32::MAX) + 1.0,
        ..ModuleStats::default()
    };
    assert!(validate_module_stats("m", &stats).is_err());
}

#[test]
fn validate_module_outcome_tracker_checks_every_module() {
    let json = r#"{"modules":{"good":{"attempts":5,"successes":1,"candidatesProduced":0},"bad":{"attempts":1,"successes":2,"candidatesProduced":0}}}"#;
    let tracker: ModuleOutcomeTracker = serde_json::from_str(json).expect("tracker JSON");
    let err = validate_module_outcome_tracker(&tracker).expect_err("`bad` must be rejected");
    assert!(err.to_string().contains("\"bad\""), "{err}");

    validate_module_outcome_tracker(&ModuleOutcomeTracker::new())
        .expect("an empty tracker must be accepted");
}

#[test]
fn error_detail_truncates_a_long_module_name() {
    // Multi-byte characters prove the cut lands on a `char` boundary.
    let name = "é".repeat(MODULE_NAME_DETAIL_MAX_CHARS * 4);
    let error = assert_rejected(&tracker_json(&name, 1, 2, "0.0"), "successes");
    assert!(!error.contains(&name), "the full name must not be echoed");
    assert!(
        error.contains(&"é".repeat(MODULE_NAME_DETAIL_MAX_CHARS)),
        "the truncated prefix must identify the module: {error}"
    );
    assert!(
        !error.contains(&"é".repeat(MODULE_NAME_DETAIL_MAX_CHARS + 1)),
        "no more than the limit may be echoed: {error}"
    );
}
