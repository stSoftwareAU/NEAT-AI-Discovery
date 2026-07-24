## Summary

Target-type prioritisation for synapse candidate scoring (Issue #468). Production discovery-cache data shows existing hidden neurons as targets have a 31.4% success rate compared to 5.3-5.4% for output or discovery-hidden neurons. This change:

- Adds `EXISTING_HIDDEN_TARGET_BOOST` constant (1.5x) to `constants.rs` for configurable target-type weighting
- Adds `apply_target_type_boost()` function that boosts candidate score gains when the target is an existing hidden neuron
- Applies the boost in the candidate scoring pipeline (after impact discounting and source-type boost)
- Adds `order_focus_targets()` to prioritise existing hidden neurons in focus order, so they are evaluated first under deadline constraints

## Evidence

This is a backend/scoring change with no UI. Verified through:
- 13 new integration tests covering all acceptance criteria
- `./quality.sh` passes cleanly (fmt, clippy, check, 566 tests, release build)

## Test Plan

Added `tests/issue_468_target_type_prioritisation.rs` with 13 tests:

**Constant validation:**
- `existing_hidden_target_boost_value_is_reasonable` — verifies boost is bounded (1.0, 3.0]
- `existing_hidden_target_boost_amplifies_score_gain` — verifies multiplier effect

**Boost function (`apply_target_type_boost`):**
- `apply_target_type_boost_boosts_existing_hidden_target` — hidden target gets boosted
- `apply_target_type_boost_neutral_for_output_target` — output target is neutral
- `apply_target_type_boost_neutral_for_unknown_target` — unknown target is neutral
- `apply_target_type_boost_zero_gain_stays_zero` — zero gain stays zero
- `apply_target_type_boost_neutral_for_input_target` — input target is neutral
- `apply_target_type_boost_neutral_for_constant_target` — constant target is neutral

**Focus ordering (`order_focus_targets`):**
- `order_focus_targets_places_existing_hidden_before_output` — hidden targets ordered first
- `order_focus_targets_consistent_across_seeds` — ordering holds across random seeds
- `order_focus_targets_preserves_all_targets` — no targets lost during reordering
- `order_focus_targets_handles_single_target` — edge case: single element
- `order_focus_targets_handles_unknown_types_as_non_hidden` — unknown types treated as non-hidden
