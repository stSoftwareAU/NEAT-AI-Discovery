//! Issue #1098: Verify that the overall wall-clock cap limits total discovery time.
//!
//! The `maxDiscoveryWallClockMinutes` configuration caps the total elapsed time
//! from discovery start, regardless of how recording and analysis budgets are
//! split. This prevents total wall time from exceeding the configured limit.

#![allow(clippy::cast_possible_truncation)]

use neat_ai_discovery::analysis::utils::{
    YEAR_2000_MS, build_deadline, cap_deadline_to_wall_clock, deadline_to_absolute_ms,
};
use std::time::{Duration, SystemTime};

/// Verify that `cap_deadline_to_wall_clock` clamps the analysis deadline
/// to the overall wall-clock cap when the analysis deadline exceeds it.
#[test]
fn wall_clock_cap_clamps_analysis_deadline_that_exceeds_it() {
    let discovery_start = SystemTime::now();

    // Analysis deadline is 30 minutes from now
    let analysis_deadline = Some(discovery_start + Duration::from_secs(30 * 60));

    // Wall-clock cap is 20 minutes from discovery start
    let wall_clock_cap_minutes = Some(20u64);

    let capped =
        cap_deadline_to_wall_clock(analysis_deadline, discovery_start, wall_clock_cap_minutes);

    assert!(capped.is_some(), "Capped deadline should be Some");
    let capped_time = capped.unwrap();

    // The capped deadline should be approximately 20 minutes from now
    // (the wall-clock cap), not 30 minutes
    let expected = discovery_start + Duration::from_secs(20 * 60);
    let drift = if capped_time > expected {
        capped_time
            .duration_since(expected)
            .unwrap_or(Duration::ZERO)
    } else {
        expected
            .duration_since(capped_time)
            .unwrap_or(Duration::ZERO)
    };

    assert!(
        drift < Duration::from_secs(2),
        "Capped deadline should be at the wall-clock cap (20 min), not the analysis deadline (30 min). Drift: {drift:?}"
    );
}

/// Verify that `cap_deadline_to_wall_clock` preserves the analysis deadline
/// when it is within the wall-clock cap.
#[test]
fn wall_clock_cap_preserves_deadline_within_cap() {
    let discovery_start = SystemTime::now();

    // Analysis deadline is 10 minutes from now
    let analysis_deadline = Some(discovery_start + Duration::from_secs(10 * 60));

    // Wall-clock cap is 20 minutes from discovery start
    let wall_clock_cap_minutes = Some(20u64);

    let capped =
        cap_deadline_to_wall_clock(analysis_deadline, discovery_start, wall_clock_cap_minutes);

    assert!(capped.is_some(), "Capped deadline should be Some");
    let capped_time = capped.unwrap();

    // The analysis deadline (10 min) is within the wall-clock cap (20 min),
    // so it should be preserved as-is
    let expected = discovery_start + Duration::from_secs(10 * 60);
    let drift = if capped_time > expected {
        capped_time
            .duration_since(expected)
            .unwrap_or(Duration::ZERO)
    } else {
        expected
            .duration_since(capped_time)
            .unwrap_or(Duration::ZERO)
    };

    assert!(
        drift < Duration::from_secs(2),
        "Analysis deadline within cap should be preserved. Drift: {drift:?}"
    );
}

/// Verify that `cap_deadline_to_wall_clock` returns the analysis deadline
/// unchanged when no wall-clock cap is configured.
#[test]
fn wall_clock_cap_none_leaves_deadline_unchanged() {
    let discovery_start = SystemTime::now();

    let analysis_deadline = Some(discovery_start + Duration::from_secs(30 * 60));

    // No wall-clock cap
    let capped = cap_deadline_to_wall_clock(analysis_deadline, discovery_start, None);

    assert!(capped.is_some(), "Deadline should be preserved");
    let capped_time = capped.unwrap();

    let expected = discovery_start + Duration::from_secs(30 * 60);
    let drift = if capped_time > expected {
        capped_time
            .duration_since(expected)
            .unwrap_or(Duration::ZERO)
    } else {
        expected
            .duration_since(capped_time)
            .unwrap_or(Duration::ZERO)
    };

    assert!(
        drift < Duration::from_secs(2),
        "Without cap, deadline should be unchanged. Drift: {drift:?}"
    );
}

/// Verify that `cap_deadline_to_wall_clock` handles None analysis deadline
/// by applying just the wall-clock cap.
#[test]
fn wall_clock_cap_with_no_analysis_deadline_applies_cap() {
    let discovery_start = SystemTime::now();

    // No analysis deadline, but wall-clock cap is 20 minutes
    let capped = cap_deadline_to_wall_clock(None, discovery_start, Some(20));

    assert!(
        capped.is_some(),
        "Wall-clock cap should produce a deadline even with no analysis deadline"
    );
    let capped_time = capped.unwrap();

    let expected = discovery_start + Duration::from_secs(20 * 60);
    let drift = if capped_time > expected {
        capped_time
            .duration_since(expected)
            .unwrap_or(Duration::ZERO)
    } else {
        expected
            .duration_since(capped_time)
            .unwrap_or(Duration::ZERO)
    };

    assert!(
        drift < Duration::from_secs(2),
        "With no analysis deadline, cap should be applied. Drift: {drift:?}"
    );
}

/// Verify that both None analysis deadline and None wall-clock cap returns None.
#[test]
fn wall_clock_cap_both_none_returns_none() {
    let discovery_start = SystemTime::now();
    let capped = cap_deadline_to_wall_clock(None, discovery_start, None);
    assert!(capped.is_none(), "Both None should return None");
}

/// Verify that the wall-clock cap is converted to an absolute timestamp
/// that works correctly with `deadline_to_absolute_ms`.
#[test]
fn wall_clock_capped_deadline_round_trips_via_absolute_ms() {
    let discovery_start = SystemTime::now();

    let analysis_deadline = Some(discovery_start + Duration::from_secs(30 * 60));
    let capped = cap_deadline_to_wall_clock(analysis_deadline, discovery_start, Some(15));

    assert!(capped.is_some());

    // Convert to absolute ms
    let abs_ms = deadline_to_absolute_ms(&capped);
    assert!(abs_ms.is_some());
    let abs_ms_value = abs_ms.unwrap();

    // Must be above YEAR_2000_MS so build_deadline treats it as absolute
    assert!(
        abs_ms_value >= YEAR_2000_MS,
        "Capped deadline absolute ms ({abs_ms_value}) should be >= YEAR_2000_MS"
    );

    // Rebuild from absolute ms should not drift
    let rebuilt = build_deadline(Some(abs_ms_value)).expect("should rebuild");
    let capped_epoch_ms = capped
        .unwrap()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let rebuilt_epoch_ms = rebuilt
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let drift = capped_epoch_ms.abs_diff(rebuilt_epoch_ms);
    assert!(
        drift < 2_000,
        "Rebuilt deadline drifted {drift}ms from the capped deadline"
    );
}

/// Verify that the FFI input accepts the maxDiscoveryWallClockMinutes field.
#[test]
fn ffi_input_deserialises_wall_clock_cap() {
    let json = r#"{
        "parquetFile": "/tmp/test.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusNeurons": ["n1"],
        "maxDiscoveryWallClockMinutes": 20
    }"#;

    let input: neat_ai_discovery::AnalyzeParallelInput =
        serde_json::from_str(json).expect("should deserialise with wall clock cap");
    assert_eq!(input.max_discovery_wall_clock_minutes, Some(20));
}

/// Verify that the FFI input defaults to None when wall clock cap is absent.
#[test]
fn ffi_input_defaults_wall_clock_cap_to_none() {
    let json = r#"{
        "parquetFile": "/tmp/test.parquet",
        "creature": {"neurons": [], "synapses": [], "input": 1, "output": 1},
        "focusNeurons": ["n1"]
    }"#;

    let input: neat_ai_discovery::AnalyzeParallelInput =
        serde_json::from_str(json).expect("should deserialise without wall clock cap");
    assert_eq!(input.max_discovery_wall_clock_minutes, None);
}
