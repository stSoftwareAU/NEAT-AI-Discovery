## Summary

Replace blocking `recv()` with `recv_timeout()` in `gpu_thread_loop()` at
`src/analysis/gpu/queue/execution.rs` so the GPU thread no longer hangs
indefinitely when the sender is dropped without sending a `Shutdown` request.
Closes #1082.

On each 30-second timeout the loop now:
- Beats the watchdog heartbeat (`watchdog::beat("gpu-queue-idle")`)
- Checks the cancellation flag (`is_cancelled()`)
- Logs a debug message for observability
- Detects a disconnected channel and exits cleanly with a warning

## Evidence

This is a backend/thread-safety change with no UI component. Evidence is
provided by the unit test below.

## Test Plan

- Added `test_gpu_thread_exits_when_sender_dropped` — creates a `GpuAnalyzer`,
  drops the channel sender without sending `Shutdown`, spawns the GPU thread
  loop, and asserts it exits cleanly (skipped on machines without a GPU)
- All 9 existing tests in `execution::tests` continue to pass
- Full `quality.sh` gate passes (clippy, fmt, tests, doc build, release build)
