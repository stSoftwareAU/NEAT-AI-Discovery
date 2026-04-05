## Summary

Interleave hidden sources among input sources during source ordering so that
hidden-to-hidden synapse candidates are evaluated even under deadline constraints.
Closes #907.

Previously, `order_eligible_sources()` always placed all input neurons before all
hidden neurons. Under tight deadlines, analysis time would run out before reaching
hidden sources, meaning hidden-to-hidden candidates were rarely or never evaluated.

This change introduces `interleave_sources()` which inserts one hidden source after
every `HIDDEN_SOURCE_INTERLEAVE_INTERVAL` (3) input sources. This gives hidden
sources approximately 25% of evaluation slots while still prioritising input sources
(which have a higher historical success rate).

### Changes

- **`src/analysis/constants.rs`**: Added `HIDDEN_SOURCE_INTERLEAVE_INTERVAL = 3`
  constant controlling how often hidden sources are inserted among input sources.
- **`src/analysis/utils/deadline.rs`**: Added `interleave_sources()` helper function
  and updated all three code paths in `order_eligible_sources()` (no-bias, weighted
  bias, and focus-unused-observations) to interleave rather than append hidden sources.
  Added diagnostic logging for input/hidden source ratios.
- **`tests/analysis/issue_907_hidden_source_interleaving.rs`**: 8 integration tests
  verifying interleaving behaviour, determinism, element preservation, and edge cases.

## Evidence

All existing tests pass (including the 11 issue #467 source-type prioritisation tests),
confirming no regression in input-source candidate quality. The new tests verify that
hidden sources appear in the first half of the evaluation order under various conditions.

## Test Plan

- `hidden_sources_are_interleaved_among_inputs` — hidden neurons appear in first half
- `all_elements_preserved_after_interleaving` — no neurons lost during reordering
- `interleaving_is_deterministic_with_seed` — same seed produces same order
- `hidden_sources_evaluated_under_simulated_deadline_pressure` — hidden sources present in first N evaluated
- `only_inputs_no_hidden_neurons_still_works` — edge case: no hidden neurons
- `only_hidden_no_input_neurons_still_works` — edge case: no input neurons
- `interleaving_consistent_across_seeds` — works across multiple seeds
- `focus_unused_observations_also_interleaves_hidden` — interleaving works with focus-unused path
- Compile-time assertions validate `HIDDEN_SOURCE_INTERLEAVE_INTERVAL` is in range [2, 5]
