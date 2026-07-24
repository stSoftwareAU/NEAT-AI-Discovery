## Summary

Deduplicate coordinated-structural candidates that share a common dominant neuron. Closes #509.

In production discovery-cache commit `a1340f8d` (creature `b2ff6e45`), 10 out of 20 candidates (50% of the budget) were coordinated-structural pairs all sharing the same dominant neuron (`e8480883`). All 10 failed with nearly identical results. This change caps pairs per dominant neuron to 3, freeing candidate slots for alternative sources.

### Changes

- **`src/analysis/epistatic.rs`**: Added `deduplicate_by_dominant_neuron()` and `deduplicate_synergistic_by_dominant_neuron()` functions. These group pairs by the neuron with the larger individual improvement and keep at most 3 diverse pairs per group (sorted by combined improvement).
- **`src/analysis/synapse/target_analysis.rs`**: Applied deduplication in the pipeline after interference filtering and before conversion to coordinated candidates, for both epistatic and synergistic paths.

### Expected impact

- For the issue scenario: reduces coordinated-structural from 10 to 3 candidates, freeing 7 slots.
- No loss of coverage since all 10 were redundant (same dominant neuron, same outcome).
- Pairs with distinct dominant neurons are completely unaffected.

## Evidence

This is a backend logic change with no UI component. Verified by 9 new integration tests and all 474 existing unit tests passing.

## Test Plan

- Added `tests/issue_509_deduplicate_dominant_neuron.rs` with 9 tests:
  - `epistatic_pairs_with_shared_dominant_neuron_are_capped` — 10 pairs sharing one dominant capped to 3
  - `epistatic_pairs_with_distinct_dominant_neurons_unchanged` — distinct groups unaffected
  - `epistatic_deduplication_keeps_best_combined_improvement` — best pairs retained
  - `epistatic_deduplication_empty_input` — empty input returns empty
  - `epistatic_deduplication_single_pair` — single pair passes through
  - `epistatic_deduplication_mixed_groups` — mixed shared/unique groups handled correctly
  - `synergistic_with_shared_primary_are_capped` — synergistic candidates capped per primary
  - `synergistic_with_distinct_primaries_unchanged` — distinct primaries unaffected
  - `epistatic_dominant_is_higher_individual_improvement` — dominant determined by higher individual improvement, not position
- All existing tests pass (`./quality.sh` clean)
