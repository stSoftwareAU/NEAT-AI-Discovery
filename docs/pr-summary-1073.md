## Summary

Add Parquet column pruning via `ProjectionMask` to skip the `errors` `ListArray` column
when it is not needed by the caller. Introduces a `ColumnProfile` enum with `Full` and
`WithoutErrors` variants, and new `_with_profile` reader functions that apply column
projection at the Parquet I/O level. Existing reader functions are unchanged and delegate
to the profile-aware versions with `ColumnProfile::Full`. Closes #1073.

## Evidence

Benchmark results from `cargo bench --bench parquet_loading_comparison -- parquet_column_pruning`:

| Dataset | Full Read | Without Errors | Speedup |
|---------|----------|----------------|---------|
| 50n x 200r x 5 errors (10K records) | 952 us | 488 us | **1.95x (49% faster)** |
| 100n x 500r x 10 errors (50K records) | 5.27 ms | 2.23 ms | **2.36x (58% faster)** |
| 200n x 500r x 20 errors (100K records) | 13.46 ms | 4.50 ms | **2.99x (67% faster)** |

The improvement scales with the number of error values per record, confirming that
`ListArray` deserialisation is the dominant cost being avoided.

## Test Plan

- Added `tests/parquet_column_pruning.rs` with 8 integration tests:
  - `test_full_profile_returns_errors` — full profile preserves errors
  - `test_without_errors_profile_returns_empty_errors` — pruned profile returns empty errors
  - `test_without_errors_preserves_other_fields` — obs_index, value, activation preserved
  - `test_without_errors_preserves_nullable_value` — nullable value field handled correctly
  - `test_grouped_read_without_errors` — grouped read with pruning works
  - `test_grouped_read_full_profile_matches_original` — full profile matches original behaviour
  - `test_limit_read_without_errors` — limit + pruning combination works
  - `test_column_profile_debug_display` — enum variant distinction
- All existing tests pass unchanged
- `./quality.sh` passes cleanly
