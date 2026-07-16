## Summary

Add a cancellation signal to the FFI analysis pipeline so the TypeScript host
can request graceful shutdown when SIGTERM arrives (e.g. `max-task-hours
exceeded`). Previously the Rust analysis continued running after the host began
cleanup, creating a race condition where the parquet file was deleted while
analysis was still reading it. Closes #1047.

### What changed

1. **New `cancellation` module** (`src/cancellation.rs`) — a global `AtomicBool`
   flag with `request_cancellation()`, `reset_cancellation()`, and
   `is_cancelled()` functions.

2. **New FFI exports** (`cancel_analysis`, `reset_cancellation`) — the host calls
   `cancel_analysis()` on SIGTERM; the flag is automatically reset at the start
   of each `analyze_all()` invocation.

3. **Integrated into `deadline_passed()`** — all existing deadline-check points
   now also check the cancellation flag, so synapse analysis, neuron analysis,
   and detection modules all stop promptly.

4. **Parquet reader batch boundary check** — cancellation is checked at each
   batch boundary during parquet loading, preventing the race condition from
   the original bug report.

5. **Distinguishable result** — cancelled analysis returns
   `{ success: true, cancelled: true }` with `error_kind: "cancelled"` (not
   retryable), rather than crashing or returning an error.

6. **New `DiscoveryErrorKind::Cancelled` variant** — allows the host to
   distinguish cancellation from timeouts, GPU errors, etc.

## Evidence

- All 12 new tests pass (cancellation flag lifecycle, deadline integration,
  error classification, FFI output serialisation)
- All existing tests pass (713 lib + 286 integration)
- `quality.sh` passes cleanly

## Test Plan

- `tests/infrastructure/issue_1047_cancellation_signal.rs` — 12 tests covering:
  - Cancellation flag starts false, can be set and cleared
  - `deadline_passed()` returns `true` when cancelled (with and without a real deadline)
  - `DiscoveryErrorKind::Cancelled` is not retryable and serialises correctly
  - `DiscoveryError::Cancelled` has the correct kind and message
  - String-based error classification detects cancellation messages
  - `AnalyzeParallelOutput` includes `cancelled` field when set, omits when `None`
- `src/cancellation.rs` — 1 unit test for flag lifecycle
