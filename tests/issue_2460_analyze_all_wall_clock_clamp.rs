//! Issue #2460: `analyze_all` must clamp the FFI `max_discovery_wall_clock_minutes`.
//!
//! `analyze_all` resolves the cap through `resolve_wall_clock_deadline`, so
//! asserting on that helper pins the production wiring: with the call site
//! reverted to the unclamped read, an out-of-range FFI value reaches
//! `cap_deadline_to_wall_clock` raw and no effective cap of at most 120 minutes
//! is produced.

use neat_ai_discovery::analysis::resolve_wall_clock_deadline;
use std::time::{Duration, SystemTime};

const WRAPPING_MINUTES: u64 = 307_445_734_561_825_861;
const MAX_CAP_MINUTES: u64 = 120;

fn fixed_start() -> SystemTime {
    SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000)
}

#[test]
fn oversized_ffi_minutes_are_clamped_to_max_cap() {
    let start = fixed_start();

    for requested in [u64::MAX, WRAPPING_MINUTES] {
        let (deadline, minutes) = resolve_wall_clock_deadline(None, start, Some(requested));

        assert!(minutes <= MAX_CAP_MINUTES, "effective cap {minutes} > 120");
        assert_eq!(minutes, MAX_CAP_MINUTES);
        assert_eq!(
            deadline,
            Some(start + Duration::from_secs(MAX_CAP_MINUTES * 60)),
            "requested {requested} must yield a 120 minute deadline"
        );
    }
}

#[test]
fn oversized_ffi_minutes_still_respect_tighter_analysis_deadline() {
    let start = fixed_start();
    let analysis_deadline = start + Duration::from_secs(10 * 60);

    let (deadline, minutes) =
        resolve_wall_clock_deadline(Some(analysis_deadline), start, Some(u64::MAX));

    assert_eq!(minutes, MAX_CAP_MINUTES);
    assert_eq!(deadline, Some(analysis_deadline));
}

#[test]
fn zero_ffi_minutes_clamp_up_to_one_minute() {
    let start = fixed_start();

    let (deadline, minutes) = resolve_wall_clock_deadline(None, start, Some(0));

    assert_eq!(minutes, 1);
    assert_eq!(deadline, Some(start + Duration::from_secs(60)));
}
