## Summary

Add integration tests for two infrastructure modules that previously lacked dedicated test coverage: confidence metrics (`src/analysis/confidence.rs`) and diagnostic tracking (`src/analysis/diagnostics/`). Closes #527.

## Changes

### `tests/issue_527_confidence_metrics.rs` (11 tests)
- Empty samples return zero confidence and zero-width interval
- Single sample produces low confidence
- Identical (zero-variance) samples produce low confidence
- High error variance produces wider confidence intervals than low variance
- More samples produce narrower confidence intervals
- Confidence interval always contains the point estimate
- High R² yields higher confidence than low R²
- Extreme outlier activations still produce valid (finite, clamped) metrics
- Prediction confidence is always clamped to [0, 1]
- Large varied dataset with good R² yields high confidence
- `None` R² defaults to neutral (not penalised)

### `tests/issue_527_diagnostic_tracking.rs` (7 tests)
- Hidden neurons reported as `HiddenNeuronFiltered` in neuron diagnostics
- Input neurons reported as `InputNeuronFiltered` in neuron diagnostics
- Synapse diagnostics include rejection reasons when no candidates found
- JSON output includes diagnostic fields (synapseDiagnostics / neuronDiagnostics)
- JSON diagnostic reasons serialised in snake_case format
- Multiple focus targets each get separate diagnostic entries
- Fully connected neuron reports `NoEligibleSources`

## Evidence

This is a test-only change with no UI or performance impact. All 18 new tests pass:

```
running 11 tests (issue_527_confidence_metrics)  ... ok
running 7 tests  (issue_527_diagnostic_tracking) ... ok
```

`./quality.sh` passes cleanly with all new tests.

## Test Plan

- Added `tests/issue_527_confidence_metrics.rs` — 11 integration tests for confidence interval calculations
- Added `tests/issue_527_diagnostic_tracking.rs` — 7 integration tests for diagnostic tracking pipeline
- All tests exercise real functions with test data (no source-code grepping)
- `./quality.sh` passes
