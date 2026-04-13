## Summary

Fixed the analysis timeout guard being bypassed when `analysis_deadline_ms` is a relative duration. Each analysis sub-phase (parquet loading, synapse analysis, neuron analysis, discovery modules) was calling `build_deadline()` independently. When the deadline was a relative duration (e.g., `600_000ms` = 10 minutes), each call created a fresh deadline from "now", allowing total analysis to exceed 24 minutes on a 10-minute budget. Closes #1097.

### Root Cause

`build_deadline(Some(600_000))` creates a `SystemTime` 10 minutes in the future. When called at the start of parquet loading, synapse analysis, neuron analysis, and discovery modules independently, each phase got its own 10-minute window rather than sharing a single deadline.

### Fix

- Added `deadline_to_absolute_ms()` utility to convert a `SystemTime` deadline back to an absolute millisecond timestamp (always `>= YEAR_2000_MS`, so `build_deadline` treats it as absolute rather than relative).
- In `analyze_all()`, the deadline is now built **once** at the start and converted to an absolute timestamp. All sub-phases (synapse input, neuron input, GPU queue, discovery modules) receive this shared absolute deadline.
- Added a deadline check before the post-processing phase (compression, discovery modules, reranking) so that when the analysis phases consume the entire time budget, heavyweight post-processing is skipped.

## Evidence

This is a backend/CLI fix with no visual output. Evidence is provided by the test suite:

- Unit tests verify `deadline_to_absolute_ms` correctness and round-trip consistency
- Integration test `issue_1097_shared_deadline` verifies that converting a relative deadline to absolute and rebuilding produces a consistent point in time (no drift)
- All 783 existing unit tests continue to pass
- `quality.sh` passes cleanly

## Test Plan

- Added unit tests in `src/analysis/utils/deadline_tests.rs`:
  - `deadline_to_absolute_ms_returns_none_for_none` — None input returns None
  - `deadline_to_absolute_ms_returns_absolute_timestamp` — valid deadline returns correct absolute ms
  - `shared_deadline_does_not_reset_on_rebuild` — round-trip relative → absolute → rebuild produces consistent deadline (no drift)
- Added integration test `tests/analysis/issue_1097_shared_deadline.rs`:
  - `relative_deadline_round_trip_does_not_drift` — verifies the fix prevents deadline reset
  - `deadline_to_absolute_ms_always_above_year_2000_threshold` — ensures absolute values are always treated as absolute timestamps
