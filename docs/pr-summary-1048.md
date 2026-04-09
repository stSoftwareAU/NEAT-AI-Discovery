## Summary

Prevent the parquet temp directory from being deleted while the Rust analysis pipeline is still reading from it. This race condition caused `DiscoveryError: Parquet file does not exist` crashes when the TypeScript host cleaned up `.discovery/<uuid>/` directories during SIGTERM shutdown before the Rust FFI call returned. Closes #1048.

### Changes

1. **File handle guard (defence-in-depth)**: `RecordCache`, `LruRecordCache`, and `CompressedLruRecordCache` now hold an open `File` handle to the parquet file for their entire lifetime. On Unix, this keeps the inode alive even if the TypeScript host unlinks the path, so in-flight reads still succeed.

2. **Analysis-active tracking**: Added an atomic counter (`ANALYSIS_ACTIVE`) in the cancellation module that tracks how many analysis invocations are in-flight. The counter is incremented at the start of `analyze_parallel_internal` and `rank_focus_neurons_internal`, and decremented when they return (including error/cancellation paths).

3. **`is_analysis_active` FFI export**: New FFI function the host can call to check whether analysis is still running before deleting temp directories. Returns `1` if active, `0` if idle.

4. **Recommended shutdown sequence** (documented in FFI_API.md):
   - Call `cancel_analysis()` on SIGTERM
   - Wait for the analysis FFI call to return
   - Optionally poll `is_analysis_active()` until `0`
   - Then delete the temp directory

## Evidence

- 8 new integration tests verify file guard behaviour and analysis-active counter lifecycle
- All 811+ existing tests continue to pass
- Full `quality.sh` passes cleanly (clippy, fmt, tests, doc build, release build)

## Test Plan

- `issue_1048_parquet_file_guard::record_cache_file_guard_survives_path_deletion` — verifies records accessible after path deletion
- `issue_1048_parquet_file_guard::lru_cache_holds_file_guard` — verifies LRU cache holds file guard
- `issue_1048_parquet_file_guard::compressed_cache_holds_file_guard` — verifies compressed cache holds file guard
- `issue_1048_parquet_file_guard::analysis_active_starts_at_zero` — counter starts at zero
- `issue_1048_parquet_file_guard::analysis_active_increments_and_decrements` — counter tracks correctly
- `issue_1048_parquet_file_guard::is_analysis_active_reflects_counter` — boolean API works
- `issue_1048_parquet_file_guard::analysis_active_does_not_underflow` — saturating decrement
- `issue_1048_parquet_file_guard::cancellation_and_analysis_active_are_independent` — orthogonal signals
