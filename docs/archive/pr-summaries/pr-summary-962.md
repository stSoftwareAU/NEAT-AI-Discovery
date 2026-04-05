## Summary

Add two ultra-conservative weight variant tiers — **Feather-Touch** and **Whisper** — below the existing Micro-Nudge minimum for networks approaching equilibrium where even Micro-Nudge perturbations may be too aggressive. These variants are only generated when the base weight exceeds a configurable threshold (0.05) to avoid creating near-zero variants of already-small weights. Closes #962.

### Changes

- **Neuron variants**: Added `FEATHER_TOUCH_CONFIG` (`outgoing_scale` 0.01, `expected_multiplier` 0.25) and `WHISPER_CONFIG` (`outgoing_scale` 0.005, `expected_multiplier` 0.1) in `src/analysis/utils/variant_generation.rs`
- **Synapse variants**: Added `SYNAPSE_FEATHER_TOUCH_CONFIG` (`weight_scale` 0.05) and `SYNAPSE_WHISPER_CONFIG` (`weight_scale` 0.02)
- **Coordinated-structural variants**: Added `COORDINATED_ULTRA_VARIANT_SPECS` with Feather-Touch (0.01x) and Whisper (0.005x) scaling
- **Threshold gating**: `ULTRA_CONSERVATIVE_BASE_WEIGHT_THRESHOLD` (0.05) prevents generating ultra-conservative variants for already-small weights
- **Helper refactor**: Extracted `variant_name_from_comment()` to reduce duplication in variant name extraction

## Evidence

All 26 new tests pass, covering:
- Config value validation for both tiers
- Correct scaling behaviour for neuron, synapse, and coordinated-structural candidates
- Threshold gating (variants generated for large weights, skipped for small weights)
- Comment/label verification

`quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build).

## Test Plan

- Added `tests/analysis/issue_962_ultra_conservative_variants.rs` with 26 tests:
  - Feather-Touch and Whisper neuron config and variant tests
  - Feather-Touch and Whisper synapse config and variant tests
  - Ultra-conservative threshold gating tests for synapse, neuron, and coordinated-structural pairing
- Updated existing tests in:
  - `tests/analysis/issue_806_dry_variant_generation.rs` — updated max variant count
  - `tests/analysis/extreme_candidate_pairing.rs` — updated expected variant counts
  - `tests/neuron/neuron_metadata_candidates_found_includes_pairing.rs` — updated expected counts
  - `tests/recommendation/issue_507_micro_nudge_variant.rs` — updated expected counts
  - `tests/synapse/issue_513_synapse_weight_variants.rs` — updated expected counts and comment checks
  - `tests/synapse/issue_510_coordinated_structural_weight_variants.rs` — updated expected counts
