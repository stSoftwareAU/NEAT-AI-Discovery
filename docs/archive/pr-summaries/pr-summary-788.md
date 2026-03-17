## Summary

Add a new **high-error squash exploration** detection module that proactively
suggests alternative activation functions for hidden neurons with high prediction
error. This increases change-squash candidate volume — the highest success-rate
candidate type (65%) — by triggering on error magnitude rather than waiting for
structural problems like saturation or mismatch. Closes #788.

## Approach

The module (`src/analysis/detection/high_error_squash_exploration.rs`) works by:

1. For each hidden neuron with pre-activation data and mean absolute error
   above a threshold (0.10), infer the target output as `activation + error`.
2. Simulate each candidate activation function (TANH, LOGISTIC, IDENTITY,
   SOFTSIGN, HARD_TANH, RELU, ELU, SELU, MISH, SWISH) on the pre-activation
   values.
3. If an alternative reduces mean absolute error by at least 15% versus the
   current activation, recommend it as a `ChangeSquash` coordinated candidate.

Conservative thresholds (15% error reduction minimum, 0.01 improvement scaling)
are designed to maintain a high success rate (target: >30%).

## Evidence

- 9 integration tests covering all edge cases (detection, non-detection,
  conversion, multi-neuron, skip conditions)
- 3 unit tests for internal logic
- Module count test updated from 43 to 44
- `quality.sh` passes cleanly (fmt, clippy, check, tests, docs, release build)

## Test Plan

- `tests/issue_788_high_error_squash_exploration.rs` (9 integration tests):
  - `high_error_neuron_with_pre_activation_data_triggers_exploration`
  - `low_error_neuron_not_detected`
  - `neuron_without_pre_activation_skipped`
  - `insufficient_samples_returns_empty`
  - `identity_neurons_skipped`
  - `does_not_recommend_same_squash`
  - `candidates_convert_to_coordinated_candidates`
  - `multiple_high_error_neurons_detected`
  - `neuron_without_records_returns_empty`
- `src/analysis/detection/high_error_squash_exploration.rs` (3 unit tests):
  - `test_is_skip_squash`
  - `test_high_error_triggers_detection`
  - `test_low_error_skips`
- `src/analysis/module_dispatch_specs/mod.rs` module count assertion updated
