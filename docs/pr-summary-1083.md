## Summary

Reduce GPU batch size on memory exhaustion before retry. When an OOM error occurs during GPU execution, the retry loop now halves the batch size (down to a minimum floor of 64 samples) before re-initialising the `GpuAnalyzer`. This makes GPU recovery more likely to succeed on memory-constrained systems. Closes #1083.

## Changes

- **`src/analysis/gpu/queue/recovery.rs`**: Added `is_memory_exhaustion_error()` to detect OOM-specific errors (subset of `is_device_lost_error`), and `MINIMUM_GPU_BATCH_SIZE` constant (64) as the reduction floor
- **`src/analysis/gpu/analyzer.rs`**: Added `new_with_batch_size()` to create a `GpuAnalyzer` with an overridden batch size, and `batch_size()` accessor
- **`src/analysis/gpu/queue/execution.rs`**: Modified the retry loop to halve batch size on OOM, enforce the minimum floor, and use `new_with_batch_size()` for recovery. The reduced batch size persists for subsequent requests
- **`src/observability/gpu_metrics.rs`**: Added `effective_batch_size` and `batch_size_reductions` tracking, exposed in the metrics report
- **`src/analysis/gpu/mod.rs`**: Re-exported new public symbols

## Evidence

This is a backend change with no UI. Verified via:
- 9 new integration tests all passing
- 17 unit tests in recovery module all passing
- Full quality gate (`./quality.sh`) passes cleanly

## Test Plan

- Added `tests/gpu/issue_1083_gpu_batch_size_reduction.rs` with 9 tests:
  - `test_memory_exhaustion_detects_oom_patterns` — OOM pattern detection
  - `test_memory_exhaustion_ignores_non_memory_errors` — non-OOM errors not matched
  - `test_memory_exhaustion_detection_is_case_insensitive` — case insensitivity
  - `test_minimum_batch_size_is_64` — constant verification
  - `test_batch_size_halving_above_minimum` — halving stays above floor
  - `test_batch_size_halving_reaches_minimum` — iterative halving reaches floor
  - `test_batch_size_below_minimum_cannot_reduce` — below-floor detection
  - `test_batch_size_reduction_from_large_value` — reduction from 1024
  - `test_gpu_metrics_tracks_batch_size_reduction` — metrics tracking
- Added 4 unit tests in `recovery.rs` for `is_memory_exhaustion_error` and `MINIMUM_GPU_BATCH_SIZE`
- Existing GPU recovery tests (issue #647) continue to pass
