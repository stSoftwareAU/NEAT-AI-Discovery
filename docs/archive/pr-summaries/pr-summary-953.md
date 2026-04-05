## Summary

Fix liveness stall in the GPU work queue that caused long-running discovery jobs
to hang during neuron analysis until killed by the outer task timeout. Closes #953.

### Root cause

The `GpuEvaluator` trait implementation for `GpuWorkQueue` passed `None` for the
deadline parameter on every GPU operation, giving each call the maximum 5-minute
timeout. During neuron analysis, multiple rayon threads submit work through the
shared queue via this trait. If the GPU becomes slow (e.g. Metal driver contention,
buffer mapping delays), all threads block simultaneously for up to 5 minutes per
call. The cumulative stall far exceeds the outer task timeout, causing the process
to appear hung with no useful progress.

### Fix

1. **Deadline propagation** -- Added a `deadline` field to `GpuWorkQueue` and a
   `with_deadline()` builder method. The `GpuEvaluator` trait implementation now
   uses this deadline to calculate adaptive timeouts (via the existing
   `calculate_gpu_batch_timeout()`) instead of always using the 5-minute maximum.

2. **Wired deadlines into analysis** -- Both `analyze_neurons_with_cache` and
   `analyze_synapses_with_cache` now set the analysis deadline on the GPU queue
   at creation time, ensuring all GPU operations respect the overall time budget.

3. **Stall detection logging** -- Added per-request timing and a 30-second stall
   warning in the GPU thread loop. Requests exceeding this threshold emit a
   `tracing::warn` with elapsed time and completion count, providing early
   visibility into GPU performance issues before the outer timeout fires.

## Evidence

- Unit tests verify `with_deadline()` correctly stores and overrides deadlines
- Integration tests verify both neuron and synapse analysis complete successfully
  with deadlines set, and that expired deadlines return promptly
- All 158 existing tests continue to pass
- `./quality.sh` passes cleanly (fmt, clippy, check, tests, docs, release build)

## Test Plan

- Added `tests/gpu/issue_953_gpu_queue_deadline.rs` with three integration tests:
  - `neuron_analysis_with_deadline_completes` -- verifies neuron analysis with a
    2-minute deadline runs correctly
  - `synapse_analysis_with_deadline_completes` -- same for synapse analysis
  - `analysis_with_expired_deadline_returns_promptly` -- verifies that an already-
    expired deadline causes fast return rather than stalling
- Added three unit tests in `src/analysis/gpu/queue/mod.rs`:
  - `test_with_deadline_none_leaves_deadline_unset`
  - `test_with_deadline_some_stores_deadline`
  - `test_with_deadline_overrides_previous`
