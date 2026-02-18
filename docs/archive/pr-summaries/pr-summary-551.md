## Summary

Add bias perturbation detection for activation regime shifts — a new detection
module that identifies hidden neurons operating in suboptimal activation regimes
and generates exploratory `setBias` candidates targeting specific regime shifts.
Closes #551.

Sub-issue of #545 (Suggestions for getting out of local minimums).

### What it does

A neuron's bias determines its operating regime on the activation function curve.
When a neuron operates in a saturated tail (e.g., TANH output ~1.0) or flat region,
small bias changes keep it in the same regime. This module detects such neurons and
generates a large bias shift that moves the neuron to the centre of the active zone,
effecting a qualitative regime change to escape the local minimum.

### Detection criteria

1. Hidden neuron uses a bounded squash with a known active zone
2. Dynamic range utilisation < 25% (operating in a suboptimal regime)
3. Mean absolute error ≥ 0.05 (non-negligible error — not already converged)
4. Sufficient samples (≥ 20)
5. Bias change would be meaningful (≥ 0.1)

### Candidate generation

Each detected neuron produces a coordinated `setBias` candidate targeting the
centre of the activation function's active zone.

## Evidence

This is a backend detection module with no UI changes. Evidence is provided by
the integration test suite.

## Test Plan

- `tests/issue_551_bias_perturbation_regime_shift.rs` — 7 integration tests:
  1. Detects neurons in saturated positive tail regime (TANH with bias=5.0)
  2. Detects neurons in saturated negative tail regime (LOGISTIC with bias=-6.0)
  3. `setBias` targets active zone centre (recommended bias closer to 0)
  4. No false positives for neurons in healthy operating regimes (low error)
  5. Coordinated candidates have positive gain and reference issue #551
  6. Multiple suboptimal neurons each produce candidates
  7. Insufficient samples returns empty
