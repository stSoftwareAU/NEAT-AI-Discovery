## Summary

Cross-module candidate deduplication for coordinated structural candidates (Issue #489).

When 25+ discovery modules run in parallel, different modules can independently propose
candidates targeting the same neuron with the same operation (e.g., both saturation and
oscillating detection propose `changeSquash` for neuron X). Previously, candidate
clustering (Issue #224) only reduced redundancy **within** synapse candidates but did
not deduplicate **across** discovery modules' coordinated structural candidates.

This PR adds a post-merge deduplication pass that:

1. **Groups candidates by operation signature** — target neuron + operation type + similar
   parameters (weights/biases bucketed within 5% tolerance)
2. **Keeps the best representative** — highest `expected_creature_score_gain` per group
3. **Detects conflicts** — flags when different operation types target the same neuron
   (e.g., `removeNeuron` vs `changeSquash`), reported via verbose logging for diagnostics

The deduplication runs after all discovery modules have contributed candidates but before
the existing within-module clustering (Issue #224), ensuring both mechanisms work together.

No changes to the NEAT-AI calling interface. Reuses existing candidate types.

## Evidence

This is a backend/library change with no UI. Evidence is provided by the test suite:

- 16 integration tests covering all deduplication scenarios
- All 463 unit tests + 97 integration test files pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

New test file: `tests/issue_489_cross_module_deduplication.rs` (16 tests)

| Test | Scenario |
|------|----------|
| `test_identical_change_squash_deduplicated` | Same ChangeSquash for same neuron from two modules |
| `test_best_candidate_kept` | Highest improvement is selected as representative |
| `test_different_op_types_not_deduplicated` | Different operations on same neuron both kept |
| `test_different_targets_not_deduplicated` | Different target neurons both kept |
| `test_empty_input` | Empty input produces empty output |
| `test_single_candidate_passthrough` | Single candidate unchanged |
| `test_multi_op_candidates_deduplicated` | Multi-operation candidates with identical ops |
| `test_conflicting_ops_flagged` | Remove vs modify conflicts detected |
| `test_deduplication_metadata_counts` | Correct duplicate/original counts |
| `test_large_candidate_set_deduplication` | 100 candidates across 5 neurons → 5 unique |
| `test_different_squash_values_not_deduplicated` | Different activation targets kept |
| `test_different_bias_values_not_deduplicated` | Different bias values kept |
| `test_similar_bias_values_deduplicated` | Similar bias values (within 5%) deduplicated |
| `test_add_synapse_different_weights_not_deduplicated` | Different weights kept |
| `test_add_synapse_similar_weights_deduplicated` | Similar weights deduplicated |
| `test_output_sorted_by_improvement` | Output sorted best-first |
