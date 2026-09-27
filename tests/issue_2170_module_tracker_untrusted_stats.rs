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
