## Summary

Implements source-type prioritisation for synapse candidate generation (Issue #467).

Production discovery-cache data shows input neurons as synapse sources have a **36.2% success rate** compared to only **2.8–3.3%** for hidden neurons. This PR adds two complementary mechanisms:

1. **Source evaluation ordering**: `order_eligible_sources()` now partitions input neurons before hidden neurons in all code paths (default shuffle, focus-unused-observations, and weighted-index-bias). Under deadline constraints, this ensures input-neuron sources are always evaluated first.

2. **Score gain boost**: The `INPUT_SOURCE_BOOST` multiplier (1.5×, configurable in `constants.rs`) is applied to `expected_creature_score_gain` for candidates whose source is an input neuron, via the new `apply_source_type_boost()` function.

### Changes

| File | Change |
|------|--------|
| `src/analysis/synapse.rs` | Added `apply_source_type_boost()` public function; applied boost in impact-discounting loop for helpful candidates |
| `src/analysis/utils/deadline.rs` | Modified `order_eligible_sources()` to partition input neurons before hidden neurons in all ordering paths |
| `tests/issue_467_source_type_prioritisation.rs` | 11 new tests covering ordering, boost application, and edge cases |

## Evidence

This is a backend/library change with no UI component. Evidence is provided via test results:

- All 11 new tests pass, verifying:
  - Input neurons always ordered before hidden neurons across multiple seeds
  - `apply_source_type_boost()` correctly boosts input sources and leaves hidden/output sources unchanged
  - Boost preserves zero-gain invariant
  - All neurons are preserved during reordering
- All existing tests continue to pass (including `order_eligible_sources` unit tests and issue #465 source-type scoring tests)
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- `tests/issue_467_source_type_prioritisation.rs` — 11 new tests:
  - `order_eligible_sources_places_input_neurons_before_hidden` — verifies input-first ordering
  - `order_eligible_sources_input_first_with_different_seeds` — consistency across 5 seeds
  - `order_eligible_sources_preserves_all_neurons_with_prioritisation` — no neuron loss
  - `order_eligible_sources_input_first_respects_focus_unused_override` — compatibility with Issue #182
  - `input_source_boost_value_is_reasonable` — constant range validation
  - `input_source_boost_amplifies_score_gain` — multiplier arithmetic
  - `input_source_boost_does_not_apply_to_hidden_sources` — differential boost
  - `apply_source_type_boost_boosts_input_source` — function correctness for input UUIDs
  - `apply_source_type_boost_neutral_for_hidden_source` — no boost for hidden neurons
  - `apply_source_type_boost_neutral_for_output_source` — no boost for output neurons
  - `apply_source_type_boost_zero_gain_stays_zero` — zero invariant
