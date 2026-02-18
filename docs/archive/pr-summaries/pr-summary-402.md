## Summary

Range-aware weight optimisation — compute weights that respect bounded observation ranges (Issue #402).

When observations contain sentinel values (e.g., -1.0 meaning "no data"), the standard weight
calculation includes these meaningless samples, leading to suboptimal weight proposals. This PR
adds a pre-filter wrapper that excludes sentinel samples before computing optimal weights.

### Changes

- **`src/analysis/weights.rs`**: Added two new public functions:
  - `compute_range_aware_sums()` — filters samples by sentinel values from `ObservationRangeResult` (#398) and computes `sum_error_activation` / `sum_activation_sq` on the effective set only.
  - `calculate_range_aware_weight()` — wrapper that calls `compute_range_aware_sums()` then delegates to the existing `calculate_optimal_outgoing_weight()` (DRY — core function unchanged).
  - Added `DEFAULT_SENTINEL_TOLERANCE` constant (0.02, matching the observation range module).

- **`src/analysis/mod.rs`**: Re-exported the new functions and constant.

### Design Decisions

- The existing `calculate_optimal_outgoing_weight()` is **not modified** — the new functions wrap it with pre-filtering (DRY principle per acceptance criteria).
- Sentinel matching uses the same tolerance (0.02) as the observation range detection module (#398).
- Uses `ObservationRangeResult` metadata directly from #398 for sentinel values.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

Added 8 integration tests in `tests/issue_402_range_aware_weight_optimisation.rs`:

1. `test_filtered_sums_exclude_sentinel_samples` — verifies sentinel samples are excluded from accumulators
2. `test_range_aware_weight_differs_from_full_range` — proves range-aware weight differs when sentinels skew full-range
3. `test_no_sentinels_matches_full_range_weight` — confirms identical results when no sentinels present
4. `test_all_samples_sentinel_returns_none` — returns None when all samples are at sentinel values
5. `test_insufficient_effective_samples_returns_none` — handles gracefully with minimal effective samples
6. `test_multiple_sentinels_filtered` — filters multiple sentinel values simultaneously
7. `test_uses_observation_range_metadata` — uses `ObservationRangeResult` from #398 including tolerance
8. `test_existing_function_unchanged_dry_wrapper` — confirms core function unchanged (DRY)

All existing tests continue to pass. `./quality.sh` passes cleanly.
