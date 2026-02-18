## Summary

Add calibration tracking to the discovery pipeline so it can self-correct prediction biases over time. The `CalibrationTracker` records predicted vs actual improvement per discovery module and candidate type, then computes calibration metrics (MAE, bias direction, per-module calibration factors). An FFI endpoint (`get_calibration_summary`) exposes the calibration data to NEAT-AI for debugging and logging. Closes #605.

### What changed

- **`src/discovery_history.rs`**: Added `CalibrationTracker` type with `record_prediction()`, `calibration_factor()`, and `calibration_summary()` methods. Integrated into `DiscoveryHistory` with `record_calibration()`, `calibration_factor()`, and `calibration_summary()` delegate methods. Backward-compatible: existing serialised `DiscoveryHistory` JSON deserialises cleanly via `#[serde(default)]`.
- **`src/ffi_types/requests.rs`**: Added `CalibrationSummaryInput` request type.
- **`src/ffi_types/responses.rs`**: Added `CalibrationSummaryOutput` response type.
- **`src/ffi_internal.rs`**: Added `get_calibration_summary_internal()` business logic function.
- **`src/ffi/utilities.rs`**: Added `get_calibration_summary()` FFI entry point with panic safety.
- **`tests/issue_605_calibration_tracking.rs`**: 13 integration tests covering all calibration functionality.

## Evidence

This is a backend/library change with no visual UI. Evidence is provided by the 13 passing integration tests covering:

- Recording predictions and computing metrics (MAE, bias, calibration factor)
- Per-module/candidate-type separation
- Serialisation roundtrip
- FFI endpoint returning valid JSON
- Edge cases (zero predicted, unknown modules, empty history)

## Test Plan

- `tests/issue_605_calibration_tracking.rs` — 13 new integration tests:
  - `calibration_tracker_records_predictions`
  - `calibration_tracker_computes_mean_absolute_error`
  - `calibration_tracker_computes_bias_direction`
  - `calibration_tracker_negative_bias_for_under_prediction`
  - `calibration_tracker_per_module_calibration_factor`
  - `calibration_factor_defaults_to_one_for_unknown_module`
  - `calibration_tracker_separates_modules_and_candidate_types`
  - `calibration_tracker_serialisation_roundtrip`
  - `discovery_history_includes_calibration_tracker`
  - `calibration_tracker_handles_zero_predicted`
  - `calibration_summary_sorted_by_sample_count`
  - `ffi_get_calibration_summary_returns_valid_json`
  - `ffi_get_calibration_summary_empty_history`
- All 509 existing unit tests pass
- All existing integration tests pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
