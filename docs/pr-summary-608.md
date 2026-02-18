## Summary

Split `src/analysis/gpu/queue.rs` (897 lines) into focused sub-modules under `queue/` directory for better separation of concerns. Closes #608.

The split follows the structure requested in the issue:
- `queue/mod.rs` — Public API, re-exports, queue types (`GpuWorkQueue`, `GpuWorkRequest`, `GpuFuture`)
- `queue/submission.rs` — Work item submission and batching (all `evaluate_*` and `submit_*` methods)
- `queue/execution.rs` — GPU thread main loop and `GpuEvaluator` trait implementation
- `queue/scheduling.rs` — Initialisation (`new()`), shutdown, and `Drop` implementation

## Evidence

This is a pure refactoring with no UI or performance changes. The public API remains unchanged — all types are re-exported at the same paths. All 509 unit tests and 97 integration test files pass without modification. `cargo clippy` and `./quality.sh` pass cleanly.

## Test Plan

- All existing tests pass without modification (509 unit + integration tests)
- `cargo clippy` passes with no warnings
- `./quality.sh` passes all checks (fmt, clippy, check, test, release build)
- No public API changes — `GpuWorkQueue`, `GpuFuture`, and `GpuWorkRequest` remain accessible at the same crate paths
