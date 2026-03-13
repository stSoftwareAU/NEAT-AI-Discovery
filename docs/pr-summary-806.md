## Summary

Consolidated three near-identical variant generation functions (conservative, gentle nudge,
micro-nudge) into single parameterised functions driven by `NeuronVariantConfig` and
`SynapseVariantConfig` structs. Extracted all variant generation logic from `utils/mod.rs`
(818 lines) into a new `variant_generation.rs` sub-module, reducing `mod.rs` to 102 lines.
Addresses #806.

### Changes

- **`NeuronVariantConfig`** struct replaces `make_conservative_add_neuron_variant()`,
  `make_gentle_nudge_add_neuron_variant()`, and `make_micro_nudge_add_neuron_variant()`
  with a single `make_neuron_variant(candidate, config)` function
- **`SynapseVariantConfig`** struct replaces `make_conservative_synapse_variant()`,
  `make_gentle_nudge_synapse_variant()`, and `make_micro_nudge_synapse_variant()`
  with a single `make_synapse_variant(candidate, config)` function
- Synapse pairing function now iterates over a config array instead of repeating
  the same block three times
- Coordinated-structural pairing similarly uses a `COORDINATED_VARIANT_SPECS` array
- All public functions re-exported at `analysis::utils::` level for backward compatibility
- `utils/mod.rs`: 818 → 102 lines; new `variant_generation.rs`: 556 lines

## Evidence

All 40+ existing variant-related tests pass unchanged, confirming behavioural equivalence:
- `tests/extreme_candidate_pairing.rs` (4 tests)
- `tests/issue_507_micro_nudge_variant.rs` (10 tests)
- `tests/issue_513_synapse_weight_variants.rs` (11 tests)
- `tests/issue_510_coordinated_structural_weight_variants.rs` (11 tests)
- `tests/sensible_range_filtering.rs` (2 tests)
- `tests/neuron_metadata_candidates_found_includes_pairing.rs` (3 tests)

## Test Plan

- Added `tests/issue_806_dry_variant_generation.rs` (18 tests) verifying:
  - `NeuronVariantConfig` correctly clamps incoming weight, bias, and outgoing weight
  - Each preset config (conservative, gentle nudge, micro-nudge) produces expected values
  - Negative weight signs are preserved
  - Near-zero outgoing weights use the config-specific fallback
  - Expected improvement is scaled by the config multiplier
  - Comments are set correctly per config
  - `SynapseVariantConfig` correctly scales weight and expected gain
  - Custom configs with arbitrary parameters produce expected variants
- All existing tests pass without modification (`quality.sh` clean)
