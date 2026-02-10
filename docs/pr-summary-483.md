## Summary

NaN-safe floating-point sorting across all analysis modules (#483).

Replaced the last remaining `partial_cmp().unwrap_or()` pattern with `total_cmp()` in
`tests/issue_419_parallel_discovery_execution.rs`. All `src/` code already used `total_cmp()`
(addressed by earlier work in #481). Added 12 comprehensive tests verifying NaN-safe
sorting behaviour across analysis functions including `ErrorDistribution`, `compute_sample_weights`,
`stratify_samples`, and `detect_high_error_neurons`.

### Changes

- **`tests/issue_419_parallel_discovery_execution.rs`**: Replaced `partial_cmp(b).unwrap_or(Equal)`
  with `total_cmp(b)` — the last remaining unsafe floating-point comparison in the codebase.
- **`tests/issue_483_nan_safe_sorting.rs`** (new): 12 tests covering:
  - `cmp_f64_desc` NaN and infinity handling
  - `ErrorDistribution::from_errors` with NaN-containing and all-NaN input
  - `compute_sample_weights` with NaN errors (verifies NaN records get zero weight)
  - `stratify_samples` with NaN errors
  - `detect_high_error_neurons` with NaN in error vectors
  - Sort stability: NaN does not corrupt ordering of finite values
  - Edge cases: subnormal values, mixed special values (NaN, infinity, subnormal, -0.0)

## Evidence

Backend-only change — no UI or performance impact. All tests exercise real library functions with
NaN-containing data and assert on observable outcomes (no source-code grep tests).

Zero `partial_cmp` usages remain in executable code (only in doc comments).

## Test Plan

- Added `tests/issue_483_nan_safe_sorting.rs` with 12 tests
- All 12 new tests pass
- All existing tests continue to pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)
