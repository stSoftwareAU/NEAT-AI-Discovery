## Summary

Replace `unwrap()`/`expect()` calls with proper error handling in production code
paths to prevent panics. Closes #872.

### Changes

**`src/focus/layers.rs`** — 4 `unwrap()` calls on `neuron_index.get_index()` replaced
with `filter_map` / `if let Some` / `let Some … else { continue }` patterns that
gracefully skip neurons with unknown UUIDs instead of panicking.

**`src/debug.rs`** — 2 `expect()` calls on `thread::Builder::spawn()` replaced with
`if let Err` + `tracing::warn!` so debug utilities log a warning and continue
rather than crashing the process on thread spawn failure.

**`src/ffi_internal/`** — Already uses proper `?` operator and `match`/`if let`
error handling throughout all production code. The `unwrap()`/`expect()` calls in
this module are exclusively within `#[cfg(test)]` blocks, which is acceptable.

## Evidence

- All 124 existing tests pass (including GPU-skipped tests)
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
- No new `unwrap()`/`expect()` introduced in production code

## Test Plan

- Added `synapse_referencing_unknown_uuid_does_not_panic` in `tests/focus/focus_layers.rs`
  — verifies `compute_network_layers` handles synapses referencing UUIDs not in the
  neuron list without panicking
- Added `neuron_with_uuid_missing_from_index_is_skipped_gracefully` in
  `tests/focus/focus_layers.rs` — verifies disconnected neurons with no synapses are
  handled gracefully
- All existing tests continue to pass unchanged
