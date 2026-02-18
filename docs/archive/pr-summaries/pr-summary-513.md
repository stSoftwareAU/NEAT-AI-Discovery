## Summary

Generate multiple weight variants for synapse candidates, matching the existing
behaviour for add-neuron candidates. Closes #513.

The discovery process is expensive (~1 hour) but testing each candidate is cheap
(~1 minute). Previously, synapse candidates were returned with a single weight
value. Now each helpful synapse candidate is paired with three weight variants:

| Variant | Weight Scale | Expected Gain Multiplier |
|---------|-------------|-------------------------|
| Original | 1.0× | 1.0× |
| Conservative | 0.5× | 0.5× |
| Gentle Nudge | 0.25× | 0.75× |
| Micro-Nudge | 0.1× | 0.25× |

Variants are deduplicated (skipped when the scaled weight is too close to
the original or to another variant), and the total count respects the
existing `maxCandidates` limit.

A `comment` field was added to `CandidateSynapseJson` to label each variant,
consistent with the existing `comment` field on `CandidateNeuronJson`.

## Evidence

This is a backend/logic change with no visual output. Verified by:
- 11 new integration tests covering variant generation, weight scaling,
  sign preservation, deduplication, budget limits, and comment labelling
- All 472 unit tests + all integration tests pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- Added `tests/issue_513_synapse_weight_variants.rs` with 11 tests:
  - `synapse_candidate_generates_four_variants`
  - `synapse_conservative_variant_has_scaled_weight`
  - `synapse_gentle_nudge_variant_has_scaled_weight`
  - `synapse_micro_nudge_variant_has_scaled_weight`
  - `synapse_variant_preserves_negative_weight_sign`
  - `synapse_variant_expected_gain_is_scaled`
  - `synapse_variant_respects_max_candidates_limit`
  - `synapse_variant_skips_near_zero_weight_candidates`
  - `synapse_variant_original_comment_lists_included_variants`
  - `synapse_variant_multiple_candidates_each_get_variants`
  - `synapse_variant_from_to_neuron_uuids_match_original`
