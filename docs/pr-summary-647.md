## Summary

Add GPU device-lost recovery with automatic retry to the GPU thread loop. When a GPU operation fails with a device-lost error (e.g., macOS Metal timeout, driver reset, resource exhaustion), the GPU thread now detects the error, re-initialises the `GpuAnalyzer`, and retries the failed work item. This allows unattended workers to recover from transient GPU issues without restarting the entire discovery process. Closes #647.

## Changes

- **`src/analysis/gpu/queue/recovery.rs`** (new) — Device-lost error detection (`is_device_lost_error`) and retry limit configuration (`get_gpu_retry_limit`), configurable via `NEAT_AI_DISCOVERY_GPU_RETRY_LIMIT` environment variable (default: 3)
- **`src/analysis/gpu/queue/execution.rs`** — Refactored `gpu_thread_loop` to detect device-lost errors from GPU operations, re-initialise the `GpuAnalyzer` on recovery, and retry the failed work item with configurable retry limits. Extracted `execute_request` helper for clean separation of execution and recovery logic
- **`src/analysis/gpu/queue/mod.rs`** — Added `recovery` sub-module
- **`src/analysis/gpu/mod.rs`** — Re-exported recovery types for public API access

## Evidence

This is a backend/infrastructure change with no visual UI. Evidence is provided by the passing test suite:

- All 8 new integration tests pass (`tests/issue_647_gpu_device_lost_recovery.rs`)
- All 6 unit tests pass in `recovery.rs`
- All existing GPU tests pass unchanged
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- **`tests/issue_647_gpu_device_lost_recovery.rs`** — 8 integration tests:
  - `test_device_lost_detection_recognises_wgpu_patterns` — Known device-lost error messages are detected
  - `test_device_lost_detection_ignores_normal_errors` — Normal operational errors are not misclassified
  - `test_device_lost_detection_is_case_insensitive` — Case-insensitive matching
  - `test_device_lost_detection_with_nested_errors` — Chained anyhow error detection
  - `test_default_retry_limit_is_three` — Default configuration value
  - `test_retry_limit_env_var_name_is_correct` — Env var name constant
  - `test_device_lost_error_with_buffer_mapping_context` — Buffer mapping device loss
  - `test_device_lost_error_with_allocation_failure` — Memory allocation device loss
- **`src/analysis/gpu/queue/recovery.rs`** — 6 unit tests covering error detection and configuration
