## Summary

Dedicated issue and comprehensive test suite for opposing synapse detection as a
structural discovery method (#360). The opposing synapse detection code was originally
implemented as part of PR #357 (Issue #356). This PR creates a focused test file for
issue #360, updates issue references in source and documentation, and verifies all
detection behaviour with 29 tests.

Opposing synapses have contributions (weight × source_activation) that correlate
positively with the target neuron's error (Pearson r ≥ 0.3), meaning they actively
worsen predictions. The detection recommends:
- `removeSynapse` coordinated structural candidates for strong opposition (r > 0.5)
- `setWeight` (negated weight) candidates for moderate opposition (0.3 ≤ r ≤ 0.5)

### Changes

- **New test file**: `tests/issue_360_opposing_synapse_detection.rs` — 29 tests covering
  detection criteria, edge cases, boundary conditions, candidate conversion, sorting,
  weight flip vs removal paths, diagnostic comments, and multiple output neurons.
- **Updated issue references**: Changed opposing synapse references from Issue #356 to
  Issue #360 in `src/analysis/opposing_synapse.rs`, `src/analysis/mod.rs`, and `README.md`.
- **Version bump**: 0.9.24 → 0.9.25.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

All 29 tests in `tests/issue_360_opposing_synapse_detection.rs`:

1. `test_detects_opposing_synapse` — Strong positive contribution–error correlation detected
2. `test_helpful_synapse_not_flagged` — Negative correlation (helpful) excluded
3. `test_hidden_target_synapses_not_analysed` — Only output neuron targets analysed
4. `test_insufficient_samples_not_flagged` — Below minimum sample count excluded
5. `test_strongly_opposing_recommends_removal` — Correlation > 0.5 recommends removal
6. `test_moderately_opposing_recommends_weight_flip` — Correlation 0.3–0.5 recommends flip
7. `test_coordinated_candidate_conversion` — Produces correct coordinated operations
8. `test_empty_synapses_no_candidates` — Empty synapse list handled
9. `test_missing_source_records_handled` — Missing source records handled gracefully
10. `test_missing_target_records_handled` — Missing target records handled gracefully
11. `test_low_correlation_not_flagged` — Correlation below threshold excluded
12. `test_dormant_synapse_excluded` — Low-contribution synapses excluded
13. `test_multiple_opposing_synapses_detected` — Multiple opposing synapses all detected
14. `test_candidates_sorted_by_estimated_improvement` — Best improvement first
15. `test_estimated_improvement_always_positive` — All improvements positive
16. `test_coordinated_candidate_comment_includes_diagnostics` — Comment has UUIDs, correlation, weight
17. `test_exactly_minimum_samples_accepted` — Exactly 20 samples accepted
18. `test_nineteen_samples_below_minimum_rejected` — 19 samples rejected (boundary)
19. `test_empty_records_no_candidates` — Empty records list handled
20. `test_coordinated_candidates_sorted_by_score_gain` — Coordinated output sorted
21. `test_negative_weight_opposing_synapse_detected` — Negative weight detected
22. `test_each_candidate_has_one_operation` — Each candidate has exactly one operation
23. `test_empty_candidates_conversion` — Empty candidate conversion handled
24. `test_weight_flip_uses_negated_weight` — SetWeight uses negated weight value
25. `test_removal_candidate_uses_remove_synapse` — RemoveSynapse operation correct
26. `test_perfect_correlation_detected` — Perfect linear correlation near 1.0
27. `test_candidate_fields_populated` — All candidate fields correctly populated
28. `test_multiple_output_neurons` — Opposing synapses detected for multiple output targets
29. `test_weight_flip_discount_factor` — Removal vs flip discount (0.7×) applied correctly
