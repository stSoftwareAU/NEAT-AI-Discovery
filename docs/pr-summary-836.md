## Summary

Replace `panic!()` with `std::process::abort()` in the deadlock detector thread so that the process terminates immediately when a deadlock is found, rather than having the panic caught by `catch_unwind()` at the FFI boundary. Diagnostic output is flushed to stderr before aborting. The check interval is reduced from 10 to 5 seconds for faster detection. Closes #836.

## Evidence

- The deadlock detector in `src/debug.rs` now calls `std::process::abort()` after printing diagnostics and flushing stderr, matching the pattern used by the watchdog in `src/watchdog.rs`.
- `panic!()` has been removed from the detection path entirely.

## Test Plan

- Added unit test `test_deadlock_check_interval_is_short` verifying the interval is between 1–5 seconds
- Added integration test `deadlock_detector_reports_no_deadlocks_in_clean_state` verifying no false positives
- Added integration test `deadlock_detector_initialisation_is_idempotent` verifying safe repeated calls
- All existing tests pass; `quality.sh` passes cleanly
