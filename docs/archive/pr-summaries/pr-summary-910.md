## Summary

Add-synapse candidates between existing hidden neurons are now boosted and tracked
with diagnostic logging to ensure they compete fairly against input-sourced candidates.
Closes #910.

### Changes

1. **`HIDDEN_SOURCE_BOOST` constant** (`src/analysis/constants.rs`): Added a 1.2x boost
   multiplier for hidden-neuron sources. This is lower than `INPUT_SOURCE_BOOST` (1.5x)
   so input sources remain prioritised, but hidden-to-hidden candidates are no longer
   at a 1.5x disadvantage.

2. **`apply_source_type_boost` update** (`src/analysis/synapse/scoring.rs`): Hidden
   neuron sources now receive `HIDDEN_SOURCE_BOOST` instead of neutral (1.0x). Output
   sources remain unboosted.

3. **Source type distribution logging** (`src/analysis/synapse/post_processing.rs`):
   Added `log_source_type_distribution()` that logs the count of candidates by source
   type (input-to-hidden, input-to-output, hidden-to-hidden, hidden-to-output) after
   post-processing. This diagnostic output helps verify that hidden-to-hidden candidates
   are being generated.

4. **Test updates**: Updated existing tests in `issue_467_source_type_prioritisation.rs`
   and `issue_522_synapse_scoring.rs` to reflect the new hidden source boost behaviour.
   Business logic changed: hidden sources now intentionally receive a boost.

## Evidence

- Forward-only topology constraint (`neuron.index < target_index`) already correctly
  allows hidden-to-hidden connections — verified by tests
- Source ordering via `order_eligible_sources` with interleaving (#907) ensures hidden
  sources get evaluation time under deadline pressure — verified by existing tests
- `HIDDEN_SOURCE_BOOST` (1.2x) is bounded between 1.0 and `INPUT_SOURCE_BOOST` (1.5x)
  — enforced by compile-time assertions

## Test Plan

- Added 12 new tests in `tests/analysis/issue_910_hidden_to_hidden_synapse_candidates.rs`:
  - Compile-time validation of `HIDDEN_SOURCE_BOOST` bounds
  - `apply_source_type_boost` correctly applies hidden boost to hidden sources
  - Input sources still outrank hidden sources
  - Output sources receive no boost
  - Zero and negative gains handled correctly
  - Forward-only topology allows valid hidden-to-hidden connections
  - Hidden neurons present in source ordering for hidden targets
  - Source filtering does not exclude hidden-to-hidden candidates
- Updated `tests/analysis/issue_467_source_type_prioritisation.rs`:
  - `apply_source_type_boost_neutral_for_hidden_source` → `apply_source_type_boost_applies_hidden_boost_to_hidden_source`
  - `input_source_boost_does_not_apply_to_hidden_sources` → `input_source_boost_exceeds_hidden_source_boost`
- Updated `tests/synapse/issue_522_synapse_scoring.rs`:
  - `source_boost_not_applied_to_hidden_uuids` → `source_boost_applies_hidden_boost_to_hidden_uuids`
