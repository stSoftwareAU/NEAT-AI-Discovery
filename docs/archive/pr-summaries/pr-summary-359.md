## Summary

Dedicated issue and comprehensive test suite for dormant synapse detection as a
structural discovery method (#359). The dormant synapse detection code was originally
implemented as part of PR #357 (Issue #356). This PR creates a focused test file for
issue #359, updates issue references in source and documentation, and verifies all
detection behaviour with 26 tests.

Dormant synapses have near-zero weights (< 1e-4) that contribute negligible signal to
their target neurons. The detection recommends `removeSynapse` coordinated structural
candidates to reduce network complexity, provided the target neuron has other incoming
connections.

### Changes

- **New test file**: `tests/issue_359_dormant_synapse_detection.rs` — 26 tests covering
  detection criteria, edge cases, boundary conditions, candidate conversion, sorting,
  and diagnostic comments.
- **Updated issue references**: Changed dormant synapse references from Issue #356 to
  Issue #359 in `src/analysis/dormant_synapse.rs`, `src/analysis/mod.rs`, and `README.md`.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

All 26 tests in `tests/issue_359_dormant_synapse_detection.rs`:

1. `test_detects_near_zero_weight_synapse` — Near-zero weight synapse is detected
2. `test_active_synapse_not_flagged` — Active synapse with meaningful weight excluded
3. `test_sole_connection_not_flagged` — Sole connection protected even if dormant
4. `test_insufficient_samples_not_flagged` — Below minimum sample count excluded
5. `test_candidates_produce_coordinated_removal_operations` — Produces removeSynapse operations
6. `test_multiple_dormant_synapses_detected` — Multiple dormant synapses all detected
7. `test_other_fan_in_count_correct` — Fan-in metadata is accurate
8. `test_empty_synapses_no_candidates` — Empty synapse list handled
9. `test_missing_source_records_handled` — Missing source records handled gracefully
10. `test_weight_above_threshold_not_dormant` — Weight above threshold excluded
11. `test_weight_at_exact_threshold_passes_weight_check` — Boundary behaviour verified
12. `test_zero_weight_high_activation_is_dormant` — Zero-weight synapse detected regardless of activation
13. `test_low_activation_does_not_make_active_synapse_dormant` — Active weight not flagged with low activation
14. `test_candidates_sorted_by_estimated_improvement` — Best improvement first
15. `test_estimated_improvement_always_positive` — All improvements positive
16. `test_coordinated_candidate_comment_includes_diagnostics` — Comment has neuron UUIDs, weight, samples
17. `test_dormant_to_one_target_active_to_another` — Per-synapse detection, not per-source
18. `test_exactly_minimum_samples_accepted` — Exactly 20 samples accepted
19. `test_negative_near_zero_weight_detected` — Negative near-zero weight detected
20. `test_coordinated_candidates_sorted_by_score_gain` — Coordinated output sorted
21. `test_nineteen_samples_below_minimum_rejected` — 19 samples rejected (boundary)
22. `test_empty_records_no_candidates` — Empty records list handled
23. `test_mean_abs_contribution_correct` — Mean contribution computation verified
24. `test_varying_activations_correct_mean_contribution` — Varying activation mean verified
25. `test_empty_candidates_conversion` — Empty candidate conversion handled
26. `test_each_candidate_has_one_remove_synapse_op` — Each candidate has exactly one operation
