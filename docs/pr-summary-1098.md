## Summary

Add overall wall-clock cap for total discovery time via `maxDiscoveryWallClockMinutes` FFI input field and `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` environment variable (default 20 min). This caps the total elapsed time from discovery start (recording + analysis), preventing runaway discovery sessions that exceed `max-task-hours`. The analysis deadline is clamped to `min(analysis_deadline, discovery_start + wall_clock_cap)`. Closes #1098.

## Changes

- **`src/analysis/utils/deadline.rs`**: Add `cap_deadline_to_wall_clock()` function that computes `min(analysis_deadline, discovery_start + wall_clock_cap_minutes)`
- **`src/config/user_facing.rs`**: Add `NEAT_AI_DISCOVERY_MAX_WALL_CLOCK_MINUTES` environment variable (default 20, range 1–120 minutes) with `max_wall_clock_minutes()` accessor
- **`src/ffi_types/requests.rs`**: Add `max_discovery_wall_clock_minutes: Option<u64>` to `AnalyzeParallelInput` and `AnalyzeAllInput`
- **`src/ffi_internal/analysis.rs`**: Pass through `max_discovery_wall_clock_minutes` in `build_analyze_all_input_from_parallel()`
- **`src/analysis/orchestration.rs`**: Apply wall-clock cap to the overall deadline in `analyze_all()` — when FFI field is absent, falls back to the env var (default 20 min)
- **Documentation**: Update `AGENTS.md`, `docs/FFI_API.md`, and `src/config/mod.rs` with the new configuration

## Evidence

This is a backend/FFI change with no visual output. Evidence is provided by passing tests:

- 5 unit tests in `deadline_tests.rs` verify `cap_deadline_to_wall_clock()` behaviour
- 8 integration tests in `tests/analysis/issue_1098_wall_clock_cap.rs` verify end-to-end behaviour including FFI deserialisation
- 2 config tests verify default values and range validation
- All existing tests pass cleanly; full quality gate (`./quality.sh`) passes

## Test Plan

- Added `tests/analysis/issue_1098_wall_clock_cap.rs` with 8 integration tests:
  - `wall_clock_cap_clamps_analysis_deadline_that_exceeds_it`
  - `wall_clock_cap_preserves_deadline_within_cap`
  - `wall_clock_cap_none_leaves_deadline_unchanged`
  - `wall_clock_cap_with_no_analysis_deadline_applies_cap`
  - `wall_clock_cap_both_none_returns_none`
  - `wall_clock_capped_deadline_round_trips_via_absolute_ms`
  - `ffi_input_deserialises_wall_clock_cap`
  - `ffi_input_defaults_wall_clock_cap_to_none`
- Added 5 unit tests to `src/analysis/utils/deadline_tests.rs` for `cap_deadline_to_wall_clock()`
- Added 2 config tests: `wall_clock_minutes_default_values`, `wall_clock_minutes_returns_valid_value`
