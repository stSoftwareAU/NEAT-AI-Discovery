//! Issue #1097: Verify that the analysis deadline is computed once and shared
//! across all sub-phases.
//!
//! Before this fix, each sub-phase called `build_deadline(analysis_deadline_ms)`
//! independently. When `analysis_deadline_ms` was a relative duration (e.g.,
//! `600_000ms` = 10 minutes), each call created a fresh deadline from "now",
//! allowing total analysis to exceed the intended timeout (24+ minutes on a
//! 10-minute budget).
//!
//! The fix converts the relative duration to an absolute timestamp once at the
//! top of `analyze_all()` and passes that to all sub-phases.

#![allow(clippy::cast_possible_truncation)]

use neat_ai_discovery::analysis::utils::{YEAR_2000_MS, build_deadline, deadline_to_absolute_ms};

/// Verify the round-trip: relative ms → `build_deadline` → absolute ms → `build_deadline`
/// produces a consistent deadline that does not drift forward.
#[test]
fn relative_deadline_round_trip_does_not_drift() {
    let relative_ms = 300_000u64; // 5 minutes

    // Build initial deadline from relative duration
    let initial = build_deadline(Some(relative_ms)).expect("should produce a deadline");

    // Convert to absolute ms
    let abs_ms = deadline_to_absolute_ms(&Some(initial)).expect("should produce absolute ms");

    // Verify it's treated as an absolute timestamp
    assert!(
        abs_ms >= YEAR_2000_MS,
        "Absolute timestamp should be >= YEAR_2000_MS"
    );

    // Rebuild from absolute ms — should NOT add another 5 minutes
    let rebuilt = build_deadline(Some(abs_ms)).expect("should produce a deadline from absolute ms");

    let initial_epoch_ms = initial
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let rebuilt_epoch_ms = rebuilt
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let drift = initial_epoch_ms.abs_diff(rebuilt_epoch_ms);
    assert!(
        drift < 2_000,
        "Rebuilt deadline drifted {drift}ms — should be near-identical to original. \
         A large drift means the absolute timestamp was re-interpreted as relative."
    );
}

/// Verify that `deadline_to_absolute_ms` always produces values above
/// the `YEAR_2000_MS` threshold so they are treated as absolute timestamps
/// by `calculate_effective_timeout_ms`.
#[test]
fn deadline_to_absolute_ms_always_above_year_2000_threshold() {
    // Even a very short relative deadline should produce an absolute timestamp
    // well above YEAR_2000_MS when converted.
    let short_deadline = build_deadline(Some(3_000)); // 3 seconds (minimum)
    let abs_ms =
        deadline_to_absolute_ms(&short_deadline).expect("short deadline should be convertible");
    assert!(
        abs_ms >= YEAR_2000_MS,
        "Even a 3-second deadline should produce an absolute ms ({abs_ms}) >= YEAR_2000_MS ({YEAR_2000_MS})"
    );

    // A long relative deadline should also be above the threshold
    let long_deadline = build_deadline(Some(3_600_000)); // 1 hour (maximum)
    let abs_ms =
        deadline_to_absolute_ms(&long_deadline).expect("long deadline should be convertible");
    assert!(
        abs_ms >= YEAR_2000_MS,
        "A 1-hour deadline should produce an absolute ms ({abs_ms}) >= YEAR_2000_MS ({YEAR_2000_MS})"
    );
}
