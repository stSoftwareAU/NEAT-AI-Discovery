## Summary

Add 25 targeted integration tests across 3 new test files covering the synapse sub-modules that previously lacked dedicated test coverage: `scoring.rs`, `structural_patterns.rs`, and `post_processing.rs`. Closes #522.

### Changes

- **`tests/issue_522_synapse_scoring.rs`** (new): 15 tests exercising the public scoring functions directly — `apply_pessimism_discount`, `apply_source_type_boost`, and `apply_target_type_boost`. Covers edge cases (zero gain, zero total, negative gain), monotonicity of pessimism discount, boost stacking, and correct application order.
- **`tests/issue_522_synapse_structural_patterns.rs`** (new): 5 GPU-dependent pipeline tests for coordinated structural pattern discovery — noisy-vs-trusted folding (positive and negative cases) and hidden neuron collapse (positive case, multi-incoming guard, existing-bypass guard).
- **`tests/issue_522_synapse_post_processing.rs`** (new): 5 GPU-dependent pipeline tests for post-processing — impact discounting (output vs hidden), sort order verification, max_candidates truncation, pessimism discount integration, and metadata assembly.

### Approach

- Scoring functions are `pub` and tested directly with unit-style assertions against known constants.
- Structural patterns and post-processing functions are `pub(crate)`, so they are exercised through the full analysis pipeline (`analyze_synapses` / `analyze_parallel_internal`) with carefully constructed creature topologies that trigger specific code paths.
- GPU-dependent tests use `skip_without_gpu!()` guards for CI compatibility.

## Evidence

All 478 unit tests and 100 integration test files pass. `./quality.sh` passes cleanly including fmt, clippy, check, tests, and release build.

## Test Plan

### Scoring (direct function tests)
- `pessimism_discount_all_samples_improved_gives_full_gain` — 100% ratio yields gain unchanged
- `pessimism_discount_no_samples_improved_gives_floor` — 0% ratio yields FLOOR × gain
- `pessimism_discount_half_samples_improved` — 50% ratio yields correct interpolation
- `pessimism_discount_zero_total_gives_floor` — division-by-zero guard
- `pessimism_discount_preserves_sign_for_negative_gain` — negative input stays negative
- `pessimism_discount_zero_gain_stays_zero` — zero input stays zero
- `pessimism_discount_monotonically_increases_with_improved_ratio` — monotonicity
- `source_boost_applied_to_various_input_indices` — INPUT_SOURCE_BOOST for input-N UUIDs
- `source_boost_not_applied_to_hidden_uuids` — hidden UUIDs unaffected
- `source_boost_negative_gain_boosted_correctly` — boost applied to negative values
- `target_boost_applied_to_existing_hidden_neuron` — EXISTING_HIDDEN_TARGET_BOOST for hidden
- `target_boost_not_applied_to_output_neuron` — output neurons unaffected
- `target_boost_not_applied_to_missing_neuron` — unknown UUIDs unaffected
- `target_boost_multiple_neuron_types` — mixed type map correctness
- `combined_source_and_target_boost_stacks_multiplicatively` — 1.5 × 1.5 = 2.25

### Structural Patterns (pipeline tests)
- `noisy_vs_trusted_generates_coordinated_prune_candidate` — identical-weight pair triggers fold
- `noisy_vs_trusted_skipped_when_weights_differ` — different weights prevent fold
- `collapse_hidden_neuron_with_identity_squash` — 1-in/1-out identity chain collapses
- `no_collapse_when_hidden_has_multiple_incoming` — multi-input guard
- `no_collapse_when_bypass_synapse_already_exists` — existing bypass guard

### Post-Processing (pipeline tests)
- `output_targets_have_full_impact_hidden_targets_are_discounted` — impact discounting
- `candidates_sorted_by_expected_score_gain_descending` — sort order
- `max_candidates_limits_total_output` — truncation respects limit
- `pipeline_applies_pessimism_discount_to_candidates` — discount integration
- `metadata_contains_candidate_counts_and_timing` — metadata assembly
