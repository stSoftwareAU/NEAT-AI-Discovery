//! Regression tests for Issue #2391 (CWE-190): `NeuronDiscoveryHistory::record_attempt`
//! must saturate its counters at `u32::MAX` rather than overflow.
//!
//! Before the fix, `attempts += 1` panicked in debug and wrapped in release at the
//! ceiling, leaving `successes <= attempts` (the #1906 invariant) broken and the
//! saved history unloadable. Each test drives the real API; none greps source.

use neat_ai_discovery::discovery_history::{DiscoveryHistory, NeuronDiscoveryHistory};

/// Builds a history entry through the validated deserialise path.
fn history_at(attempts: u32, successes: u32) -> NeuronDiscoveryHistory {
    let json = format!(r#"{{"uuid":"n1","attempts":{attempts},"successes":{successes}}}"#);
    serde_json::from_str(&json).expect("fixture counts must satisfy successes <= attempts")
}

#[test]
fn record_attempt_failed_at_u32_max_saturates() {
    let mut h: NeuronDiscoveryHistory =
        serde_json::from_str(r#"{"uuid":"n1","attempts":4294967295,"successes":4294967295}"#)
            .expect("successes == attempts at u32::MAX is valid");

    h.record_attempt(false, None);

    assert_eq!(h.attempts(), u32::MAX, "attempts must pin at u32::MAX");
    assert_eq!(h.successes(), u32::MAX, "successes must pin at u32::MAX");
    assert!(
        h.successes() <= h.attempts(),
        "invariant broken: successes ({}) exceeds attempts ({})",
        h.successes(),
        h.attempts()
    );

    let serialised = serde_json::to_string(&h).expect("history must serialise");
    serde_json::from_str::<NeuronDiscoveryHistory>(&serialised)
        .expect("history must remain loadable after a failed attempt at the ceiling");
}

#[test]
fn record_attempt_successful_at_u32_max_saturates() {
    let mut h: NeuronDiscoveryHistory =
        serde_json::from_str(r#"{"uuid":"n1","attempts":4294967295,"successes":4294967295}"#)
            .expect("successes == attempts at u32::MAX is valid");

    h.record_attempt(true, Some(42));

    assert_eq!(h.attempts(), u32::MAX, "attempts must pin at u32::MAX");
    assert_eq!(h.successes(), u32::MAX, "successes must pin at u32::MAX");
    assert!(
        h.successes() <= h.attempts(),
        "invariant broken: successes ({}) exceeds attempts ({})",
        h.successes(),
        h.attempts()
    );
    assert_eq!(
        h.last_success_epoch(),
        Some(42),
        "a successful attempt must still update last_success_epoch"
    );

    let serialised = serde_json::to_string(&h).expect("history must serialise");
    serde_json::from_str::<NeuronDiscoveryHistory>(&serialised)
        .expect("history must remain loadable after a successful attempt at the ceiling");
}

#[test]
fn record_attempt_failed_at_ceiling_with_fewer_successes() {
    let mut h = history_at(u32::MAX, 5);

    h.record_attempt(false, None);
    assert_eq!(h.attempts(), u32::MAX);
    assert_eq!(h.successes(), 5);

    h.record_attempt(true, None);
    assert_eq!(h.attempts(), u32::MAX);
    assert_eq!(h.successes(), 6);
}

#[test]
fn discovery_history_record_at_ceiling_keeps_whole_history_loadable() {
    let mut history: DiscoveryHistory = serde_json::from_str(
        r#"{"neurons":{"n1":{"uuid":"n1","attempts":4294967295,"successes":4294967295}}}"#,
    )
    .expect("a history at the u32::MAX ceiling is valid");

    history.record("n1", false, None);

    let entry = history.get("n1").expect("n1 must still be present");
    assert_eq!(entry.attempts(), u32::MAX);
    assert!(entry.successes() <= entry.attempts());

    let serialised = serde_json::to_string(&history).expect("history must serialise");
    serde_json::from_str::<DiscoveryHistory>(&serialised)
        .expect("whole history must remain loadable after recording at the ceiling");
}

/// Control: below the ceiling, counters increment normally.
#[test]
fn record_attempt_below_ceiling_increments_normally() {
    let mut h = history_at(3, 1);

    h.record_attempt(false, None);
    assert_eq!(h.attempts(), 4);
    assert_eq!(h.successes(), 1);

    h.record_attempt(true, None);
    assert_eq!(h.attempts(), 5);
    assert_eq!(h.successes(), 2);
}
