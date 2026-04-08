## Summary

Add exponential backoff and observability improvements to GPU retry recovery. Closes #1038.

**Changes:**
- Added exponential backoff between GPU retry attempts (10ms -> 20ms -> 40ms, capped at 1s) to give the GPU driver time to recover before re-initialisation
- Enhanced tracing instrumentation on each retry attempt to include backoff delay, attempt number, and error details via `tracing::warn!`
- Fixed typo in error pattern string: `"a]location failed"` -> `"allocation failed"`
- Added `backoff_delay_ms()` function with configurable initial delay and maximum cap
- Exported new backoff constants (`DEFAULT_BACKOFF_INITIAL_MS`, `DEFAULT_BACKOFF_MAX_MS`) and function for external use

Part of #1035.

## Evidence

All quality checks pass (`./quality.sh`), including clippy, tests, and documentation build. The backoff function is pure and deterministic, so correctness is verified entirely via unit and integration tests.

## Test Plan

- Added 7 unit tests in `src/analysis/gpu/queue/recovery.rs`:
  - `test_backoff_delay_initial_attempt` — verifies first attempt returns initial delay
  - `test_backoff_delay_doubles_each_attempt` — verifies exponential doubling across 7 attempts
  - `test_backoff_delay_respects_cap` — verifies delay is capped at maximum
  - `test_backoff_delay_zero_attempt_returns_initial` — verifies edge case handling
  - `test_backoff_delay_with_default_constants` — verifies behaviour with production constants
  - `test_backoff_delay_large_attempt_does_not_overflow` — verifies no panic on large attempt numbers
  - `test_default_backoff_constants` — verifies constant values
  - `test_is_device_lost_error_detects_allocation_failed` — verifies corrected typo pattern
- Added 5 integration tests in `tests/gpu/issue_647_gpu_device_lost_recovery.rs`:
  - `test_device_lost_error_detects_allocation_failed_pattern` — verifies corrected pattern via public API
  - `test_backoff_delay_doubles_exponentially` — end-to-end exponential doubling check
  - `test_backoff_delay_caps_at_max` — end-to-end cap verification
  - `test_backoff_delay_does_not_overflow_on_large_attempt` — overflow safety
  - `test_backoff_constants_are_sensible` — sanity checks on exported constants
- All 171 existing integration tests pass
- All existing GPU recovery tests continue to pass
