## Summary

Dedicated issue and comprehensive test suite for oscillating neuron detection as a
structural discovery method (#358). The oscillating neuron detection code was originally
implemented as part of PR #357 (Issue #356). This PR creates a focused test file for
issue #358, updates issue references in source and documentation, and verifies all
detection behaviour with 19 tests.

Oscillating neurons have activations that frequently change sign across training samples,
indicating the neuron is fighting between contradictory functions. The detection recommends
`changeSquash` coordinated structural candidates (e.g., TANH → ABSOLUTE) to stabilise
the neuron's output.

### Changes

- **New test file**: `tests/issue_358_oscillating_neuron_detection.rs` — 19 tests covering
  detection criteria, edge cases, candidate conversion, sorting, and recommendations.
- **Updated issue references**: Changed oscillating neuron references from Issue #356 to
  Issue #358 in `src/analysis/oscillating_neuron.rs`, `src/analysis/mod.rs`, and `README.md`.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

All 19 tests in `tests/issue_358_oscillating_neuron_detection.rs`:

1. `test_perfect_alternation_detected` — Perfect alternating pattern is detected
2. `test_stable_positive_not_detected` — Consistently positive neuron excluded
3. `test_dead_neuron_excluded` — Dead neuron with sign changes excluded
4. `test_low_sign_change_frequency_excluded` — Low sign-change frequency excluded
5. `test_insufficient_samples_excluded` — Below minimum sample count excluded
6. `test_coordinated_candidate_has_change_squash` — Produces changeSquash operations
7. `test_unbalanced_sign_distribution_excluded` — >80% one sign excluded
8. `test_mixed_neurons_filters_correctly` — Only oscillating neurons detected
9. `test_recommended_squash_varies_by_activation` — TANH→ABSOLUTE, LOGISTIC→RELU, IDENTITY→ABSOLUTE
10. `test_bias_adjustment_for_imbalanced_oscillation` — Bias delta for imbalanced fractions
11. `test_candidates_sorted_by_improvement` — Best improvement first
12. `test_irregular_oscillation_detected` — Non-alternating irregular patterns detected
13. `test_empty_records_no_candidates` — Empty records handled
14. `test_missing_neuron_records_no_candidate` — Missing neuron records handled
15. `test_coordinated_candidate_includes_set_bias_when_imbalanced` — setBias included when needed
16. `test_balanced_oscillation_no_set_bias` — No setBias for balanced oscillation
17. `test_estimated_improvement_positive` — All improvements positive
18. `test_softsign_recommends_absolute` — SOFTSIGN→ABSOLUTE recommendation
19. `test_mean_abs_activation_correct` — Correct mean absolute activation computation
