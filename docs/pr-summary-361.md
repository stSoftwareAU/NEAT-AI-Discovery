## Summary

Dedicated issue and comprehensive test suite for output bias drift detection as a
structural discovery method (#361). The output bias drift detection code was originally
implemented as part of PR #357 (Issue #356). This PR creates a focused test file for
issue #361, updates issue references in source and documentation, and verifies all
detection behaviour with 29 tests.

Output bias drift identifies output neurons with consistent error sign bias (> 70%
same sign), indicating a systematic prediction offset. The detection recommends
`setBias` coordinated structural candidates that adjust the neuron's bias by the
negative of the mean error to centre predictions.

### Changes

- **New test file**: `tests/issue_361_output_bias_drift_detection.rs` — 29 tests covering
  detection criteria, edge cases, boundary conditions, candidate conversion, sorting,
  diagnostic comments, multiple output neurons, and error handling.
- **Updated issue references**: Changed output bias drift references from Issue #356 to
  Issue #361 in `src/analysis/output_bias_drift.rs`, `src/analysis/mod.rs`, and `README.md`.
- **Version bump**: 0.9.25 → 0.9.26.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

All 29 tests in `tests/issue_361_output_bias_drift_detection.rs`:

1. `test_detects_positive_bias_drift` — Predominantly positive errors detected
2. `test_detects_negative_bias_drift` — Predominantly negative errors detected
3. `test_balanced_errors_not_flagged` — 50/50 positive/negative excluded
4. `test_hidden_neuron_excluded` — Hidden neurons excluded (only output)
5. `test_input_neuron_excluded` — Input neurons excluded
6. `test_insufficient_samples_not_flagged` — Below minimum sample count excluded
7. `test_noise_level_errors_not_flagged` — Tiny magnitude errors excluded
8. `test_coordinated_candidate_conversion` — Produces correct SetBias operations
9. `test_multiple_outputs_only_biased_detected` — Only biased outputs detected
10. `test_current_bias_recorded` — Current bias correctly recorded in candidate
11. `test_recommended_bias_delta_is_negative_mean_error` — Delta = -mean_error verified
12. `test_positive_error_fraction_computed_correctly` — Fraction matches expected value
13. `test_candidates_sorted_by_estimated_improvement` — Best improvement first
14. `test_estimated_improvement_always_positive` — All improvements positive
15. `test_coordinated_candidate_comment_includes_diagnostics` — Comment has UUID, error, bias info
16. `test_exactly_minimum_samples_accepted` — Exactly 20 samples accepted
17. `test_nineteen_samples_below_minimum_rejected` — 19 samples rejected (boundary)
18. `test_exactly_seventy_percent_at_threshold` — 70% same sign accepted (boundary)
19. `test_sixty_nine_percent_below_threshold_rejected` — 69% same sign rejected (boundary)
20. `test_mean_error_at_noise_threshold_boundary` — Above/below 0.01 noise threshold
21. `test_empty_records_no_candidates` — Empty records handled
22. `test_empty_creature_no_candidates` — Empty creature handled
23. `test_set_bias_value_is_current_plus_delta` — SetBias value = current_bias + delta
24. `test_each_candidate_has_one_operation` — Each candidate has exactly one operation
25. `test_empty_candidates_conversion` — Empty conversion handled
26. `test_coordinated_candidates_sorted_by_score_gain` — Coordinated output sorted
27. `test_all_positive_errors_statistics` — 100% positive fraction correct
28. `test_all_negative_errors_statistics` — 0% positive fraction correct
29. `test_records_with_empty_errors_handled` — Empty error vectors filtered gracefully
