## Summary

Implement automatic cleanup of incomplete parquet files when a recording session is dropped without calling `finish_session()`. Uses an atomic write pattern (`.parquet.tmp` → `.parquet`) so consumers never see partially-written files. Closes #1039.

### Changes

- Added `parquet_path` and `finished` fields to `RecordingSession` to track file location and completion state
- Wrapped `ParquetRecordWriter` in `Option` to allow moving it out of the `Drop`-implementing struct for `finish()`
- Writes now go to a `.parquet.tmp` temporary file; `finish_session()` atomically renames it to `.parquet`
- Implemented `Drop` for `RecordingSession` that removes the temporary file if the session was not finished normally
- `cancel_session()` now automatically cleans up via the `Drop` implementation

## Evidence

All 13 streaming tests pass (10 existing + 3 new), and all 171 integration tests pass. The `./quality.sh` gate passes cleanly.

## Test Plan

- `test_cancel_session_cleans_up_incomplete_file` — verifies that cancelling a session with written records removes the temporary file
- `test_drop_without_finish_cleans_up_file` — verifies that dropping a session (simulating panic) removes the temporary file
- `test_finished_session_preserves_file` — verifies that successfully finished sessions have their final parquet file preserved and the tmp file is gone
