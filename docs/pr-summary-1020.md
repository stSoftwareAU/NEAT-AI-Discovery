## Summary

Add temperature scheduling for exploration-exploitation balance in the candidate selection pipeline. The temperature parameter controls how aggressively the system filters marginal candidates: high temperature (early in evolution) accepts more candidates for exploration, while low temperature (late in evolution) only accepts strong candidates for exploitation. A default temperature of 1.0 preserves existing behaviour (fully backward compatible). Closes #1020.

## Changes

### New: Temperature scheduling module (`src/analysis/constants/temperature.rs`)
- `CoolingSchedule` enum with `Linear` and `Exponential` variants
- `compute_scheduled_temperature()` — computes temperature from generation count using a cooling schedule
- `scale_threshold_by_temperature()` — scales acceptance threshold by temperature (higher temp = lower threshold = more permissive)
- `scale_ratio_by_temperature()` — scales `MIN_IMPROVED_RATIO` by temperature
- `scale_mh_temperature()` — scales Metropolis-Hastings temperature by schedule temperature
- Constants: `DEFAULT_TEMPERATURE` (1.0), `MIN_TEMPERATURE` (0.01), `MAX_TEMPERATURE` (5.0), `DEFAULT_EXPONENTIAL_DECAY_RATE` (0.995)

### Modified: FFI input structs (`src/ffi_types/requests.rs`)
- Added `temperature: f32` field (default 1.0) to `AnalyzeParallelInput`, `AnalyzeSynapsesInput`, `AnalyzeNeuronsInput`, and `AnalyzeAllInput`
- Serde default ensures backward compatibility with existing JSON callers

### Modified: Analysis pipeline
- `TargetAnalysisContext` gains `temperature` field threaded from input
- `evaluation.rs` applies temperature scaling to:
  - `MIN_IMPROVED_RATIO` threshold (via `scale_ratio_by_temperature`)
  - Acceptance threshold (via `scale_threshold_by_temperature`)
  - Metropolis-Hastings temperature (via `scale_mh_temperature`, when MH is enabled)

## Evidence

- 22 integration tests verify cooling schedules, threshold scaling, ratio scaling, MH scaling, boundary conditions, and backward compatibility
- 13 unit tests in the temperature module verify schedule functions and scaling helpers
- All existing tests pass unchanged (temperature defaults to 1.0, preserving identical behaviour)

## Test Plan

- `tests/analysis/issue_1020_temperature_scheduling.rs` — 22 tests covering:
  - Linear and exponential cooling schedule monotonicity
  - Both schedules start at initial temperature
  - Exponential cools faster initially than linear
  - Default temperature (1.0) preserves threshold, ratio, and MH temperature unchanged
  - High temperature lowers effective threshold and ratio (exploration)
  - Low temperature raises effective threshold and ratio (exploitation)
  - High/low schedule temperature scales MH acceptance accordingly
  - Temperature bounds are enforced (clamped to min/max)
  - Zero and negative temperature handled safely
  - End-to-end: cooling schedule + threshold scaling produces monotonically increasing effective threshold
  - End-to-end: cooling schedule + ratio scaling produces monotonically increasing effective ratio
- `src/analysis/constants/temperature.rs` — 13 unit tests for schedule functions and scaling helpers
