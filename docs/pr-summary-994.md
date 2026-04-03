## Summary

Fix process hang after discovery completes by adding graceful shutdown for debug background threads. Closes #994.

The root cause was that `init_debug_handlers()` spawned two background threads (`deadlock-detector` and `signal-handler`) with infinite loops and no shutdown mechanism. Their `JoinHandle`s were discarded, so there was no way to stop them. After discovery work finished, these threads continued running and — combined with `signal_hook`'s registered signal handlers — could prevent the host process from exiting cleanly.

### Changes

- **`src/debug.rs`**: Added `shutdown_debug_handlers()` function that:
  - Sets an `AtomicBool` shutdown flag checked by the deadlock-detector loop
  - Calls `signal_hook::Handle::close()` to unblock the signal-handler's `forever()` iterator
  - Joins both threads with a bounded timeout (10s) to avoid blocking shutdown
  - Stores `JoinHandle`s and signal closer in a global `DebugThreadState` struct

- **`src/ffi/utilities.rs`**: Added `cleanup_discovery_lib()` FFI function that Deno/Node callers can invoke before process exit to cleanly shut down library background threads.

## Evidence

The deadlock-detector and signal-handler threads now respect shutdown requests. The `cleanup_discovery_lib()` FFI function provides a clean shutdown path for host processes, preventing the hang described in issue #994.

## Test Plan

- Added `tests/infrastructure/issue_994_fix_deadlock.rs` with 4 integration tests:
  - `shutdown_debug_handlers_completes_promptly` — verifies shutdown finishes within 15s (no hang)
  - `shutdown_debug_handlers_is_idempotent` — multiple shutdown calls are safe
  - `cleanup_discovery_lib_ffi_does_not_hang` — FFI cleanup function works correctly
  - `cleanup_before_init_does_not_panic` — shutdown before init is safe
- Added 2 unit tests in `src/debug.rs`:
  - `test_shutdown_is_idempotent` — repeated calls do not panic
  - `test_shutdown_stops_deadlock_detector` — shutdown flag is set and read correctly
- All existing tests continue to pass (158 integration + unit tests)
