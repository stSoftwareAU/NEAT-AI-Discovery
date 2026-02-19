## Summary

Add deadline-aware parquet loading with early abort so that the parquet loading
phase respects the analysis deadline and aborts early if insufficient time
remains, returning a clear error rather than starting a doomed analysis run.
Closes #648.

### Changes

1. **`src/parquet_format/reader.rs`** — Added `read_all_records_grouped_by_neuron_with_deadline()`
   which checks the deadline at each batch boundary and beats the watchdog every 10 batches.
   The original `read_all_records_grouped_by_neuron()` delegates to this with `None` deadline.

2. **`src/analysis/cache/mod.rs`** — Added `RecordCache::new_adaptive_with_deadline()` which
   threads the deadline through to the preloaded reader. The original `new_adaptive()` delegates
   to this with `None` deadline.

3. **`src/analysis/orchestration.rs`** — `analyze_all()` now builds a `SystemTime` deadline from
   `analysis_deadline_ms` and passes it to `new_adaptive_with_deadline()`, so the loading phase
   is deadline-aware.

4. **`src/parquet_format/mod.rs`** — Re-exports the new function for public API access.

### Key behaviours

- **Deadline check before loading**: If the deadline has already passed, loading aborts immediately
  with a clear error message.
- **Deadline check at batch boundaries**: During loading, the deadline is checked at each Parquet
  batch boundary. If exceeded, a warning is logged with batch count and neuron count, and an
  error is returned.
- **Watchdog beats**: Every 10 batches, a watchdog beat is emitted so the watchdog doesn't
  trigger for legitimately slow (but progressing) loads.
- **No overhead for normal loads**: When no deadline is set, the code path is identical to the
  original (a single `None` check per batch, which the branch predictor handles trivially).

## Evidence

This is a backend/library change with no visual output. Evidence is provided by the test suite:

- 5 new integration tests verify deadline-aware loading behaviour
- All existing tests continue to pass
- `./quality.sh` passes cleanly

## Test Plan

- `tests/issue_648_deadline_aware_parquet_loading.rs`:
  - `deadline_aware_loading_succeeds_with_generous_deadline` — verifies loading succeeds with a far-future deadline
  - `deadline_aware_loading_aborts_when_deadline_already_passed` — verifies error when deadline is in the past
  - `deadline_aware_loading_with_no_deadline_matches_adaptive` — verifies no-deadline path matches original behaviour
  - `deadline_abort_error_message_is_clear` — verifies the error message mentions "deadline"
  - `deadline_aware_loading_checks_before_load_starts` — verifies pre-load deadline check
