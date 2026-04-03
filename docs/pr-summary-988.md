## Summary

Replace `assert!` / `panic!` with graceful `DiscoveryError::GpuUnavailable` error
returns in both neuron and synapse analysis when GPU is unavailable. Previously,
calling `analyze_parallel` on a machine without a GPU caused a thread panic at
`src/analysis/neuron/mod.rs:99` and `src/analysis/synapse/orchestration.rs:69`.
Now these code paths return a structured JSON error with
`{ "success": false, "errorKind": "gpu_permanent" }` that the TypeScript layer
can handle gracefully. Closes #988.

## Changes

- `src/analysis/neuron/mod.rs`: Replaced `assert!(GpuAnalyzer::gpu_is_available(), ...)`
  with an `if` check that returns `DiscoveryError::GpuUnavailable` via `anyhow`.
- `src/analysis/synapse/orchestration.rs`: Same replacement for the synapse analysis
  path.
- Both errors propagate through the existing FFI error handling infrastructure
  (`error_fields_from_anyhow` → `classify_anyhow_error`) to produce the correct
  `gpu_permanent` error kind in the JSON response.

## Evidence

- On machines without a GPU, `analyze_parallel_internal` now returns structured
  JSON with `errorKind: "gpu_permanent"` instead of panicking.
- On machines with a GPU, behaviour is unchanged — the GPU availability check
  passes and analysis proceeds normally.

## Test Plan

- Added `tests/infrastructure/issue_988_graceful_gpu_error.rs` with 3 tests:
  - `gpu_unavailable_error_propagates_as_gpu_permanent_through_anyhow` — verifies
    `DiscoveryError::GpuUnavailable` classifies as `GpuPermanent` through anyhow.
  - `gpu_unavailable_error_fields_produce_correct_triple` — verifies the
    `(error_msg, error_kind, retryable)` tuple is correct.
  - `analyze_parallel_returns_structured_error_when_no_gpu` — end-to-end test
    that calls `analyze_parallel_internal` and verifies structured JSON error
    response on machines without GPU.
- All existing tests continue to pass (`./quality.sh` clean).
