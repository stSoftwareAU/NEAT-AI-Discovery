## Summary

Apply conservative weight variants to coordinated-structural candidates, matching the
strategy already proven successful for add-neuron and synapse candidates. Closes #510.

Production evidence (creature b2ff6e45, production discovery-cache commit a1340f8d) showed that all 10
coordinated-structural candidates used a fixed weight of 0.1 for AddSynapse operations
and all failed. Meanwhile, conservative add-neuron variants with ~0.008 outgoing weight
succeeded. This change generates three additional weight variants for each coordinated-
structural candidate:

| Variant | Scale | Target weight (from 0.1) | Expected gain multiplier |
|---------|-------|--------------------------|--------------------------|
| Conservative | 0.2x | ~0.02 | 0.5x |
| Gentle Nudge | 0.1x | ~0.01 | 0.75x |
| Micro-Nudge | 0.05x | ~0.005 | 0.25x |

Each AddSynapse operation in the coordinated group is scaled independently. Non-AddSynapse
operations (RemoveSynapse, RemoveNeuron, etc.) are preserved unchanged. Candidates without
any AddSynapse operations are returned as-is with no variants.

## Evidence

This is a backend logic change with no UI component. Evidence is provided via comprehensive
tests that verify the weight scaling, sign preservation, comment handling, limit enforcement,
and correct handling of mixed operation types.

## Test Plan

- Added `tests/issue_510_coordinated_structural_weight_variants.rs` with 11 tests:
  - `coordinated_generates_four_variants` — original + 3 variants
  - `coordinated_conservative_variant_scales_add_synapse_weights` — 0.2x scaling
  - `coordinated_gentle_nudge_variant_scales_weights` — 0.1x scaling
  - `coordinated_micro_nudge_variant_scales_weights` — 0.05x scaling
  - `coordinated_variant_scales_expected_gain` — gain multipliers for each variant
  - `coordinated_variant_preserves_non_add_synapse_operations` — mixed ops preserved
  - `coordinated_variant_preserves_negative_weight_sign` — sign integrity
  - `coordinated_variant_respects_max_candidates_limit` — budget enforcement
  - `coordinated_variant_skips_candidates_without_add_synapse` — no spurious variants
  - `coordinated_variant_original_comment_lists_included_variants` — comment annotation
  - `coordinated_variant_multiple_candidates_each_get_variants` — per-candidate variants
- All existing tests continue to pass (474 unit tests + 97 integration test files)
