## Summary

Fixed the discovery cleanup race condition between async cleanup and the orphan scanner. Closes #1100.

The root cause was that `removeDiscoveryLockFile()` was called before `Deno.remove(tempDir)`, creating a window where the orphan scanner could see a directory without a lock file, classify it as orphaned, and delete it. The original async cleanup would then fail with `NotFound`.

The fix implements Option A from the issue: `cleanup_discovery_dir()` removes the entire directory tree in a single recursive `remove_dir_all()` call, so the lock file is never absent while the directory still exists. `NotFound` errors are suppressed as defence in depth, since another actor may have already cleaned the directory.

### New FFI functions

- **`cleanup_discovery_dir`** - Atomically removes a discovery temp directory (including lock file). Returns `alreadyGone: true` if another actor already removed it.
- **`clean_orphaned_discovery_dirs`** - Scans a base directory for orphaned discovery directories (no lock file) and removes them safely. Suppresses `NotFound` from race conditions.

### New files

- `src/discovery_cleanup.rs` - Core cleanup logic with `CleanupOutcome`, `OrphanCleanupResult`, and helper functions
- `src/ffi_types/cleanup.rs` - FFI input/output types for cleanup functions

## Evidence

This is a backend/library change with no UI. Verified via unit tests that exercise the exact race condition scenario described in the issue.

Key test scenarios:
- `test_concurrent_cleanup_no_not_found_error` - Two actors clean the same directory; neither gets an error
- `test_concurrent_orphan_scan_with_async_cleanup` - Atomic cleanup followed by orphan scan produces no errors
- `test_lock_file_lifecycle_correct` - Lock file exists while active, removed only when directory is gone
- `test_orphan_scan_removes_only_orphaned_dirs` - Only directories without lock files are removed

## Test Plan

- Added 12 unit tests in `src/discovery_cleanup.rs`:
  - `test_cleanup_removes_directory` - Basic directory removal
  - `test_cleanup_already_gone_returns_ok` - NotFound suppression
  - `test_is_directory_orphaned_no_lock_file` - Orphan detection without lock
  - `test_is_directory_orphaned_with_lock_file` - Active directory detection
  - `test_orphan_scan_removes_only_orphaned_dirs` - Selective orphan removal
  - `test_orphan_scan_nonexistent_base_dir` - Handles missing base dir
  - `test_orphan_scan_empty_base_dir` - Handles empty base dir
  - `test_orphan_scan_skips_files` - Only processes directories
  - `test_concurrent_cleanup_no_not_found_error` - Race condition safety
  - `test_concurrent_orphan_scan_with_async_cleanup` - End-to-end race scenario
  - `test_lock_file_lifecycle_correct` - Lock file lifecycle verification
  - `test_multiple_orphaned_dirs_cleaned` - Multiple orphan cleanup
- All existing tests continue to pass
- `./quality.sh` passes cleanly (fmt, clippy, tests, doc, release build)
