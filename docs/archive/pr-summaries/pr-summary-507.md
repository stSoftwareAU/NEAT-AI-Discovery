## Summary

Add ultra-conservative "Micro-Nudge" weight variant for add-neuron candidates. Closes #507.

Production evidence showed several conservative and gentle-nudge candidates narrowly missing success (within 0.0003 of zero score delta). The micro-nudge variant targets the sweet spot with outgoing weights in the ±0.002–0.005 range — half the conservative variant's scale.

### Key parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `MICRO_NUDGE_INCOMING_ABS_MAX` | 2.0 | Same as conservative |
| `MICRO_NUDGE_BIAS_ABS_MAX` | 1.0 | Same as conservative |
| `MICRO_NUDGE_OUTGOING_ABS_MAX` | 0.005 | Half of conservative (0.05 → 0.005) |
| `MICRO_NUDGE_OUTGOING_SCALE` | 0.05 | Half of conservative (0.2 → 0.05) |
| `MICRO_NUDGE_EXPECTED_MULTIPLIER` | 0.25 | Ranked below other variants |

### Mitigation
The micro-nudge variant is only generated when the conservative variant's outgoing weight exceeds 0.005 (the micro-nudge max). This avoids wasting the candidate budget on near-identical variants when the conservative outgoing weight is already small.

### Changes
- `src/analysis/utils/mod.rs`: Added micro-nudge constants, factory function (`make_micro_nudge_add_neuron_variant`), guard function (`should_generate_micro_nudge`), and integration into `pair_extreme_candidates_with_conservative_variants`. Updated comment generation to use dynamic formatting for 1–3 variant labels.
- `tests/coordinated_structural_replace_synapse_with_relu.rs`: Increased `maxNeuronCandidates` from 32 to 48 to accommodate the 4th variant per extreme candidate.
- `tests/extreme_candidate_pairing.rs`: Updated deduplication test limit from 6 to 8 (2 candidates × 4 variants).
- `tests/neuron_metadata_candidates_found_includes_pairing.rs`: Updated expected counts (3→4 per extreme, 9→12 for 3 extremes).

## Evidence

This is a backend/CLI change with no UI components. Evidence is provided by the test suite:
- 10 new tests in `tests/issue_507_micro_nudge_variant.rs` covering all micro-nudge behaviour
- All 472 unit tests + 97 integration tests pass via `./quality.sh`

## Test Plan

New test file `tests/issue_507_micro_nudge_variant.rs` with 10 tests:
- `micro_nudge_variant_is_generated_for_extreme_candidates` — verifies 4 variants are returned
- `micro_nudge_outgoing_weight_is_in_expected_range` — checks outgoing ≈ 0.005
- `micro_nudge_not_generated_when_conservative_outgoing_already_small` — mitigation guard
- `micro_nudge_expected_improvement_is_scaled_down` — 0.25× multiplier
- `micro_nudge_preserves_negative_outgoing_weight_sign` — sign preservation
- `extreme_candidate_comment_reflects_all_four_variants` — comment accuracy
- `micro_nudge_skipped_when_limit_too_small` — budget-constrained behaviour
- `micro_nudge_deduplication_across_different_neuron_pairs` — no cross-pair dedup
- `micro_nudge_minimum_outgoing_weight_when_scale_produces_near_zero` — guard mitigation
- `total_candidate_count_with_micro_nudge_is_four_per_extreme` — 3 × 4 = 12 total
