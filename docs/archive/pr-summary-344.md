## Summary

Implements correlated error pattern detection for shared-cause identification (Issue #344).

When multiple output neurons consistently err in the same direction on the same samples, it suggests a missing input feature or hidden representation that would benefit all of them. This new analysis detects such correlated error groups and recommends adding a shared hidden neuron that connects predictive inputs to all affected outputs as a single coordinated structural change.

### Changes

- **New module**: `src/analysis/correlated_error.rs` — Detection logic including:
  - Pearson correlation computation between output neuron error vectors
  - Complete-linkage clustering to group correlated outputs (threshold > 0.7)
  - Shared error sample counting (same-direction errors)
  - Predictive input identification (which inputs predict the shared error)
  - Coordinated structural candidate generation (AddNeuron + AddSynapse operations)
- **Pipeline integration**: `src/analysis/mod.rs` — Correlated error detection runs after dead neuron detection in the analysis pipeline. Skipped when only one output neuron exists (nothing to correlate).
- **Documentation**: `README.md` — Added section describing the detection method, skip optimisation, and candidate output format.

### Key design decisions

- **Output neurons only**: Only output neuron errors are correlated (hidden/input neurons excluded from groups)
- **First error value**: When neurons have multiple error values, the primary (first) error is used
- **Complete-linkage clustering**: All pairwise correlations in a group must exceed 0.7 (prevents spurious groupings)
- **Skip optimisation**: Single output neuron networks skip this analysis entirely, as noted in the issue

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

Added 13 tests in `tests/issue_344_correlated_error_detection.rs`:

1. `test_detects_strongly_correlated_output_errors` — Verifies strongly correlated errors across three outputs are detected
2. `test_independent_errors_no_groups` — Independent error patterns do not form groups
3. `test_single_output_skips` — Single output neuron produces no groups
4. `test_insufficient_samples_no_detection` — Too few samples (< 20) do not trigger detection
5. `test_candidates_produce_coordinated_operations` — Groups produce AddNeuron + AddSynapse coordinated operations
6. `test_two_independent_correlated_groups` — Two separate correlated groups are detected independently
7. `test_negatively_correlated_errors_no_group` — Negative correlation does not form groups
8. `test_predictive_inputs_identified` — Input neurons predictive of the shared error are correctly identified
9. `test_shared_error_sample_count` — Shared error sample count is correctly computed
10. `test_estimated_improvement_positive` — Estimated improvement is always positive for detected groups
11. `test_empty_records_no_groups` — Empty records produce no groups
12. `test_only_output_neurons_in_groups` — Hidden neurons excluded from correlated groups
13. `test_multi_error_outputs_use_first_error` — Multi-error outputs use first (primary) error for correlation
