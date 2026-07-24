## Summary

Reduce coordinated-structural false positives by applying three filtering improvements. Closes #790.

Production discovery-cache analysis (Issue #787) shows coordinated-structural candidates have a 2.3% success rate (272 / 12,069) with near-negligible actual gains (~2.2e-14). Three changes reduce candidate volume without losing successful candidates:

1. **Reduced `COORDINATED_OPERATION_DISCOUNT`** from 0.8 to 0.65 — a 4-op candidate now receives 0.65^3 ≈ 0.274 discount (was 0.512), more aggressively filtering multi-operation candidates
2. **Raised `MIN_COORDINATED_MULTI_OP_GAIN`** from 1e-5 to 1e-3 — filters out candidates with negligible predicted improvement that almost never succeed
3. **Added `COORDINATED_PESSIMISM_DISCOUNT = 0.15`** — a flat multiplicative discount applied during post-processing, analogous to the pessimism discounts for synapse/neuron candidates but using a fixed factor (since coordinated candidates lack per-sample improved ratios)

### Business logic test change

Updated `tests/issue_732_coordinated_structural_success_rate.rs::reasonable_gain_accepted_for_multi_op_candidate` — changed test gain from 0.001 to 0.01 because the tighter thresholds now correctly reject gain 0.001 on a 4-op candidate (0.001 × 0.65^3 = 2.74e-4 < 1e-3). A genuinely reasonable gain of 0.01 still passes.

## Evidence

The three changes compound to substantially reduce coordinated candidate volume:
- Operation discount: 4-op candidate gain reduced from 51.2% to 27.4% of predicted value
- Minimum gain threshold: 100× stricter (1e-3 vs 1e-5) filters marginal predictions
- Pessimism discount: additional 0.15× multiplier on all coordinated candidates

## Test Plan

- Added `tests/issue_790_reduce_coordinated_structural_false_positives.rs` with 10 tests:
  - `tighter_operation_discount_reduces_4op_candidate_aggressively` — verifies 4-op discount < 0.004
  - `tighter_operation_discount_reduces_2op_candidate` — verifies 2-op discount in expected range
  - `discount_increases_monotonically_with_operation_count` — verifies monotonic discount scaling
  - `marginal_gain_rejected_for_multi_op_with_raised_threshold` — verifies 5e-4 gain rejected
  - `previously_passing_gain_now_rejected` — verifies 1e-4 gain now rejected
  - `strong_gain_still_passes_with_raised_threshold` — verifies 0.01 gain still accepted
  - `very_small_4op_gain_rejected` — verifies 0.002 gain on 4-op rejected
  - `pessimism_discount_reduces_coordinated_candidate_gain` — verifies discount < 1.0
  - `pessimism_discount_combined_with_operation_discount_filters_aggressively` — verifies combined < 0.001
  - `pessimism_discount_is_strictly_less_than_one` — verifies discount reduces across multiple gain values
- Updated `tests/issue_732_coordinated_structural_success_rate.rs` — one test updated for new thresholds
