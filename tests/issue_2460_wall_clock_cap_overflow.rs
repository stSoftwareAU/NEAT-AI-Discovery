//! Issue #2460: the wall-clock cap multiplication must not wrap or panic.
//!
//! `cap_deadline_to_wall_clock` converts minutes to seconds. An unbounded
//! `u64` minutes value (from the FFI `max_discovery_wall_clock_minutes`) used to
//! overflow in `minutes * 60`: it wrapped to a short cap in release builds and
//! panicked in debug builds. An unrepresentable cap now yields no wall-clock
//! deadline. (The 1–120 clamp of the FFI value is unit-tested in `config`.)

use neat_ai_discovery::analysis::utils::cap_deadline_to_wall_clock;
use std::time::{Duration, SystemTime};

/// `307445734561825861 * 60` is `u64::MAX + 45`, so the unchecked product wraps
/// to 44 seconds (a 44 s cap rather than an overflow).
const WRAPPING_MINUTES: u64 = 307_445_734_561_825_861;

fn fixed_start() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
}

#[test]
fn wrapping_multiplier_does_not_produce_short_cap() {
    let start = fixed_start();

    // Must not panic (debug) and must not yield the wrapped `start + 44s` cap.
    let result = cap_deadline_to_wall_clock(None, start, Some(WRAPPING_MINUTES));

    assert_ne!(result, Some(start + Duration::from_secs(44)));
    assert_eq!(result, None, "unrepresentable cap must yield no deadline");
}

#[test]
fn u64_max_minutes_yields_no_wall_clock_deadline() {
    let start = fixed_start();

    assert_eq!(
        cap_deadline_to_wall_clock(None, start, Some(u64::MAX)),
        None
    );

    let analysis_deadline = start + Duration::from_secs(600);
    assert_eq!(
        cap_deadline_to_wall_clock(Some(analysis_deadline), start, Some(u64::MAX)),
        Some(analysis_deadline)
    );
}

#[test]
fn unrepresentable_cap_preserves_analysis_deadline() {
    let start = fixed_start();
    let analysis_deadline = start + Duration::from_secs(10 * 60);

    let result = cap_deadline_to_wall_clock(Some(analysis_deadline), start, Some(WRAPPING_MINUTES));

    assert_eq!(result, Some(analysis_deadline));
    assert_ne!(result, Some(start + Duration::from_secs(44)));
}

#[test]
fn boundary_minutes_still_cap_normally() {
    let start = fixed_start();

    // 120 minutes, the maximum valid cap, with no analysis deadline.
    assert_eq!(
        cap_deadline_to_wall_clock(None, start, Some(120)),
        Some(start + Duration::from_secs(7_200))
    );

    // 1 minute, the minimum valid cap, tighter than a 30 minute analysis deadline.
    let analysis_deadline = start + Duration::from_secs(30 * 60);
    assert_eq!(
        cap_deadline_to_wall_clock(Some(analysis_deadline), start, Some(1)),
        Some(start + Duration::from_secs(60))
    );
}
