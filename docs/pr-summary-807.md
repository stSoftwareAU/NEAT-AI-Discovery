## Summary

Add tracing instrumentation to GPU queue scheduling for observability. All
`let _ = channel.send(...)` patterns in `scheduling.rs` and `execution.rs` now
log at `trace!` level when the receiver has been dropped, providing diagnostic
visibility into GPU pipeline send failures. Debug-level lifecycle spans are
added for GPU thread init, shutdown, work item enqueue/dequeue/completion
events. No behaviour change — only observability is added. Addresses #807.

## Changes

### scheduling.rs
- Replaced 3 bare `let _ = tx.send(...)` patterns (init success, init failure,
  exit signal) with `if tx.send(...).is_err() { tracing::trace!(...) }`
- Replaced 1 bare `let _ = send_timeout(...)` (shutdown request) with trace
  logging on failure
- Added `tracing::debug!` spans for GPU thread start, init success/failure,
  shutdown request, and thread exit

### execution.rs
- Replaced 5 `let _ = response_tx.send(result)` patterns in `execute_request`
  with trace-level logging (helpful, harmful, ReLU, activation, activation batch)
- Replaced 5 `let _ = response_tx.send(Err(...))` patterns in
  `send_error_to_request` with trace-level logging
- Added `request_label()` helper for structured log fields
- Added `tracing::debug!` spans for work item dequeue, completion, and shutdown
  receipt in `gpu_thread_loop`

### submission.rs
- Replaced 1 bare `let _ = tx.send(...)` pattern (empty helpful batch
  pre-resolved future) with trace-level logging
- Added `tracing::debug!` enqueue spans for all 6 submission methods (helpful
  async, helpful blocking, harmful, ReLU, activation, activation batch) with
  sample/batch count fields

## Evidence
- All `let _ = channel.send(...)` patterns in GPU queue now have trace-level
  logging (verified by inspection — no bare `let _ =` send patterns remain)
- Uses `trace!` for high-frequency send-failure events and `debug!` for
  lifecycle events — no performance regression (trace is compiled out unless
  enabled)
- `quality.sh` passes (fmt, clippy, check, test, doc, release build)

## Test Plan
- Added 7 unit tests in `execution.rs` for `send_error_to_request`:
  - `test_send_error_to_request_handles_dropped_helpful_receiver`
  - `test_send_error_to_request_handles_dropped_harmful_receiver`
  - `test_send_error_to_request_handles_dropped_relu_receiver`
  - `test_send_error_to_request_handles_dropped_activation_receiver`
  - `test_send_error_to_request_handles_dropped_activation_batch_receiver`
  - `test_send_error_to_request_delivers_error_when_receiver_alive`
  - `test_send_error_to_request_noop_for_shutdown`
- Added 1 unit test for `request_label`:
  - `test_request_label_returns_correct_labels`
- Added 2 integration tests in `tests/issue_807_gpu_queue_tracing.rs`:
  - `test_shutdown_handles_disconnected_channel_gracefully`
  - `test_gpu_queue_module_exports_intact`
- All existing tests continue to pass
