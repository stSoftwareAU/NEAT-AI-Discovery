## Summary

Cancel in-flight discovery when memory reaches CRITICAL level. Closes #1099.

When the system has less than 5% available memory (CRITICAL pressure), in-flight analysis is now cancelled gracefully, returning partial results. This prevents OOM crashes that occurred when discovery accumulated large record buffers without any recovery mechanism.

### Changes

- **`cancellation.rs`**: Added `MEMORY_PRESSURE_CANCELLED` atomic flag, `request_cancellation_memory_pressure()`, and `is_memory_pressure_cancelled()` to distinguish memory-triggered cancellation from host SIGTERM cancellation.
- **`ffi/analysis.rs`**: Added `cancel_analysis_memory_pressure()` FFI export so the TypeScript host's memory monitor can signal CRITICAL pressure across thread boundaries to the Rust analysis pipeline.
- **`analysis/utils/memory.rs`**: Added `check_memory_pressure_and_cancel()` for Rust-side self-monitoring — checks system memory pressure and triggers cancellation if CRITICAL. Also added `would_cancel_for_memory_pressure()` pure function for testing.
- **`analysis/orchestration.rs`**: Integrated memory pressure checks at three key phase boundaries: before parquet loading, after parquet loading, and after GPU analysis dispatch.
- **`AnalyzeAllResult` / `AnalyzeParallelOutput`**: Added `memory_pressure_cancelled` field so the host can distinguish memory-triggered cancellation and take additional recovery actions (e.g., clearing WASM caches).

### Signal Path

The cancellation works via two complementary mechanisms:
1. **Host-initiated**: TypeScript memory monitor calls `cancel_analysis_memory_pressure()` FFI → sets both `CANCELLED` and `MEMORY_PRESSURE_CANCELLED` atomic flags → checked at all existing `deadline_passed()` points, parquet batch boundaries, and GPU queue idle loops.
2. **Self-monitoring**: The Rust analysis pipeline calls `check_memory_pressure_and_cancel()` at phase boundaries, checking actual system memory via `vm_stat` (macOS) or `/proc/meminfo` (Linux).

## Evidence

This is a backend/library change with no web interface. Evidence is provided by the test suite:
- 12 new integration tests verify flag lifecycle, memory pressure detection, deadline integration, and FFI output serialisation
- 6 unit tests in `cancellation.rs` verify flag behaviour including the new memory pressure tests
- All existing cancellation and memory budget tests continue to pass

## Test Plan

- Added `tests/infrastructure/issue_1099_memory_pressure_cancellation.rs` with 12 tests:
  - `memory_pressure_cancellation_sets_both_flags` — verifies both flags are set
  - `reset_clears_memory_pressure_flag` — verifies reset clears both flags
  - `normal_cancellation_does_not_set_memory_pressure_flag` — verifies isolation
  - `deadline_passed_respects_memory_pressure_cancellation` — verifies integration with deadline checks
  - `critical_pressure_triggers_cancellation_check` — pure function test for 3% available
  - `high_pressure_does_not_trigger_cancellation` — 10% available should not cancel
  - `moderate_pressure_does_not_trigger_cancellation` — 20% available should not cancel
  - `no_pressure_does_not_trigger_cancellation` — 50% available should not cancel
  - `boundary_at_5_percent_is_critical` — verifies the 5% boundary
  - `zero_total_memory_is_critical` — edge case
  - `analyze_parallel_output_includes_memory_pressure_cancelled` — FFI JSON serialisation
  - `analyze_parallel_output_omits_memory_pressure_cancelled_when_none` — JSON omission
- Added 2 unit tests in `cancellation.rs`:
  - `test_memory_pressure_cancellation_lifecycle` — flag set/reset cycle
  - `test_normal_cancellation_does_not_set_memory_pressure` — isolation test
