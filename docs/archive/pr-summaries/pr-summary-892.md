## Summary

Boost `remove-low-impact` candidate generation and priority based on production
discovery cache evidence showing a 21.5% success rate (440/2,043) — the highest
of all candidate types and roughly double the overall 10.7% rate. Closes #892.

### Changes

1. **Stricter mean activation filtering** (`removal_candidates.rs`): Candidates with
   `mean_activation > 0.04` and non-zero structural impact are filtered out. Cache
   evidence shows failed removals consistently have high mean activation (up to 57.8).
   Disconnected neurons (zero impact) are always accepted regardless of activation.

2. **Scoring boost** (`removal_candidates.rs`): Accepted removal candidates receive
   a 1.5× boost to `removal_savings`, prioritising them over other candidate types
   in the ranking. Constant `REMOVAL_CANDIDATE_BOOST = 1.5` in `constants.rs`.

3. **Widened low-impact neuron detection** (`low_impact_neuron.rs`):
   - `LOW_IMPACT_CEILING` widened from `1e-3` to `0.04` to match cache evidence
   - `MAX_ABSOLUTE_STD_DEV` widened from `1e-3` to `0.02` proportionally
   - `BASE_IMPROVEMENT` boosted from `0.002` to `0.003` to reflect the higher success rate

4. **New constants** (`constants.rs`):
   - `REMOVAL_MEAN_ACTIVATION_THRESHOLD = 0.04`
   - `REMOVAL_IMPACT_THRESHOLD = 6e-5` (documents cache evidence)
   - `REMOVAL_CANDIDATE_BOOST = 1.5`

## Evidence

- Cache evidence: `remove-low-impact` at 21.5% success rate vs 10.7% baseline
- Successful removals: mean activation ~0 to 0.04, impact ~1e-6 to 6e-5
- Failed removals: mean activation up to 57.8 (neurons were actually contributing)
- All existing tests pass with the tighter filtering and boost applied

## Test Plan

- Added `tests/focus/issue_892_removal_candidate_boost.rs` with 10 tests:
  - `removal_boost_matches_expected_value` — constant value validation
  - `low_activation_neuron_is_removal_candidate` — near-zero activation accepted
  - `high_activation_neuron_filtered_from_removal` — high activation filtered
  - `removal_candidates_receive_scoring_boost` — 1.5× boost applied to savings
  - `widened_ceiling_detects_activation_at_001` — activation 0.01 detected (was outside old ceiling)
  - `widened_ceiling_detects_activation_at_003` — activation 0.03 detected
  - `activation_above_widened_ceiling_not_detected` — activation 0.05 rejected
  - `low_impact_candidates_have_boosted_improvement` — boosted base improvement
  - `widened_detection_generates_more_candidates` — more candidates from wider detection
- All 119 focus tests pass
- All 445 detection tests pass
- `quality.sh` passes cleanly
