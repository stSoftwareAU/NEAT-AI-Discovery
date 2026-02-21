## Summary

Fix test thread count inconsistency between `quality.sh` (`--test-threads=1`)
and CI (`--test-threads=2`) by marking tests that mutate shared global state
with `#[serial]` from the `serial_test` crate, and aligning both environments
to `--test-threads=2`. Closes #673.

### Root Cause

Tests that set environment variables (e.g. `NEAT_AI_DISCOVERY_GPU_TIMING`,
`NEAT_AI_DISCOVERY_ZERO_COPY`, `NEAT_AI_DISCOVERY_BLOCK_SIZE`) were relying
on `--test-threads=1` for correctness instead of explicit serialisation. This
meant CI (which used `--test-threads=2`) could see env-var races that local
`quality.sh` (which used `--test-threads=1`) would never trigger.

### Changes

1. **Added `#[serial]`** to all tests that mutate environment variables (22
   tests across 10 files — 7 integration test files and 3 inline `src/` test
   modules).
2. **Aligned `quality.sh`** from `--test-threads=1` to `--test-threads=2` to
   match CI.
3. **Updated SAFETY comments** from "Tests run single-threaded
   (`--test-threads=1`)" to "Serialised via `#[serial]`" across all affected
   files.
4. **Updated documentation** in `AGENTS.md`, `README.md`, `CONTRIBUTING.md`,
   and `benchmark.sh` to reflect the new `--test-threads=2` setting and
   `#[serial]` convention.

Tests using `lock_for_test_serialisation()` (watchdog/dispatch tests) already
had their own serialisation mechanism and did not need `#[serial]`.

## Evidence

This is a backend/infrastructure change with no visual output.

- `./quality.sh` passes with `--test-threads=2` (all tests pass, including the
  newly serialised env-var tests).
- CI already used `--test-threads=2`, so the same thread count is now
  consistent in both environments.

## Test Plan

- No new tests added — this change marks existing tests with `#[serial]` to
  make them safe under parallel execution.
- Verified all tests pass under `--test-threads=2` via `./quality.sh`.

### Files with `#[serial]` additions

| File | Tests marked `#[serial]` |
|------|--------------------------|
| `tests/observability.rs` | 3 tests (TIMING, PROFILE, GPU_METRICS env vars) |
| `tests/gpu_timing.rs` | 4 tests (GPU_TIMING env var) |
| `tests/issue_228_zero_copy_buffer.rs` | 3 tests (ZERO_COPY env var) |
| `tests/issue_193_streaming_parquet.rs` | 4 tests (BLOCK_SIZE env var) |
| `tests/issue_192_error_distribution_analysis.rs` | 3 tests (OUTLIER_ANALYSIS env var) |
| `tests/issue_527_diagnostic_tracking.rs` | 2 tests (NEURON_TARGETS_OUTPUT_ONLY env var) |
| `tests/issue_156_hidden_focus_neurons_filtered.rs` | 1 test (NEURON_TARGETS_OUTPUT_ONLY env var) |
| `src/analysis/streaming.rs` | 2 inline tests (BLOCK_SIZE, PRELOAD_ALL env vars) |
| `src/analysis/scoring/error_distribution.rs` | 1 inline test (OUTLIER env vars) |
| `src/analysis/samples/mod.rs` | 1 inline test (CONSTANT_SOURCE_EFFECT_THRESHOLD env var) |
