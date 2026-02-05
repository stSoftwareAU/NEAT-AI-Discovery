# PR Summary: Issue #413 — Improve add-synapse prediction accuracy

## Problem

Add-synapse predictions were inverting: expected improvement of +0.00016
resulted in actual change of −0.00015. Production data showed only 10% success
rate (1 success from 10 samples), with predictions consistently in the wrong
direction.

## Root Cause

**Linear vs saturation model mismatch.** The synapse weight is computed using a
linear least-squares model (via GPU), but the predicted improvement is evaluated
using a saturation-aware simulation when the target neuron has a bounded
activation function (HARD_TANH, TANH, LOGISTIC, etc.).

When the target neuron operates near saturation, the linear-model weight
overshoots into the saturated region. The activation function clamps the output,
so the actual improvement is negative (inverted) despite the linear model
predicting positive improvement.

The add-neuron path already handled this by searching over 9 weight candidates,
but the add-synapse path used only the single linear-model weight.

## Fix

1. **Saturation-aware weight search** (`src/analysis/implementation.rs`): For
   targets with bounded/saturating activations, search over 9 weight candidates
   (scaled versions of the linear-model weight) and select the candidate with
   the best predicted improvement. This matches the approach used by add-neuron
   candidates.

2. **`is_saturating_target` helper** (`src/analysis/activation.rs`): A targeted
   check that returns `true` only for genuinely bounded activations (TANH,
   LOGISTIC, HARD_TANH, BIPOLAR, SOFTSIGN, ARCTAN, RELU6, etc.), not for
   nearly-linear functions like BENT_IDENTITY or IDENTITY that happen to have
   target simulation support.

3. **Consistent weight usage** (`src/analysis/implementation.rs`): The selected
   weight (`applied_weight`) is now used consistently in candidate emission —
   the `CandidateSynapseJson` weight field, constant source effect range, and
   setBias folding all use the weight that was actually evaluated.

## Files Changed

| File | Change |
|------|--------|
| `src/analysis/implementation.rs` | Weight search for saturating targets; consistent `applied_weight` usage |
| `src/analysis/activation.rs` | New `is_saturating_target()` helper |
| `src/analysis/synapse.rs` | 8 unit tests for prediction direction correctness |
| `tests/issue_413_add_synapse_prediction_accuracy.rs` | 2 integration tests via public API |
| `docs/DISCOVERY_TYPES.md` | Updated add-synapses status and documentation |

## Evidence

### Unit tests (8 tests in `src/analysis/synapse.rs`)

- `test_issue_413_positive_correlation_gives_positive_improvement` — Positive
  source–error correlation produces positive improvement
- `test_issue_413_negative_correlation_gives_positive_improvement` — Negative
  correlation with negative weight produces positive improvement
- `test_issue_413_hard_tanh_saturated_no_inversion` — HARD_TANH target near
  saturation does not produce inverted prediction
- `test_issue_413_deeply_saturated_not_inverted` — Deeply saturated target
  (activation at ±1.0) does not produce negative improvement
- `test_issue_413_prediction_sign_consistency_in_linear_region` — Linear region
  predictions remain consistent
- `test_issue_413_weight_search_finds_non_negative_for_saturated_target` —
  Weight search over candidates finds non-negative improvement
- `test_issue_413_tanh_saturation_not_inverted` — TANH target near saturation
  does not invert
- `test_issue_413_uncorrelated_source_near_zero_improvement` — Uncorrelated
  source produces near-zero improvement (no false positives)

### Integration tests (2 tests in `tests/issue_413_add_synapse_prediction_accuracy.rs`)

- `test_issue_413_add_synapse_hard_tanh_prediction_not_inverted` — End-to-end
  test with HARD_TANH output near saturation via public `analyze_synapses` API
- `test_issue_413_add_synapse_prediction_direction_correct` — End-to-end test
  with IDENTITY output confirming correct prediction direction

### Existing test preservation

- `issue_134_bent_identity_target_simulation_rejects_linear_false_positive` —
  Continues to pass: BENT_IDENTITY (unbounded) is correctly excluded from weight
  search, preserving the Issue #134 false-positive rejection behaviour

## Test Plan

- [x] All 8 issue #413 unit tests pass
- [x] Both issue #413 integration tests pass
- [x] Existing issue #134 direction flip tests pass
- [x] `./quality.sh` passes (lint, format, all 457+ tests, release build)
- [ ] Production validation: monitor add-synapse success rate for improvement
  from 10% toward 15–20% target
