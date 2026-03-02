## Summary

Improve coordinated-structural success rate by addressing compounding prediction errors in multi-operation candidates. Closes #732.

The coordinated-structural module had a 1.9% success rate (232/12,005) because multi-operation candidates (e.g., RemoveSynapse + AddNeuron + AddSynapse × 2) combine several predictions, each with its own uncertainty. This change introduces three improvements:

1. **Operation-count discount**: Each additional operation beyond the first applies a 0.8× compounding discount (e.g., 4-op candidates receive 0.8³ ≈ 0.512 discount), reducing over-prediction.
2. **Minimum gain threshold**: Multi-operation candidates must exceed `MIN_COORDINATED_MULTI_OP_GAIN` (1e-5) after discounting, filtering near-zero predictions that almost never succeed.
3. **Capped ensemble boost**: The agreement boost from ensemble scoring is reduced proportionally to operation count, preventing amplification of weak multi-operation signals.

## Evidence

This is a backend-only change with no UI impact. Tests verify the mathematical correctness of the discount, threshold, and boost-capping logic.

## Test Plan

- Added `tests/issue_732_coordinated_structural_success_rate.rs` with 8 tests:
  - `single_operation_has_no_discount` — verifies single-op candidates are unaffected
  - `multi_operation_candidate_is_discounted` — verifies 4-op candidates receive discount
  - `more_operations_produce_larger_discount` — verifies discount increases with op count
  - `tiny_gain_rejected_for_multi_op_candidate` — verifies minimum gain threshold filters weak candidates
  - `reasonable_gain_accepted_for_multi_op_candidate` — verifies valid candidates pass threshold
  - `single_op_with_small_gain_accepted` — verifies single-op candidates have lower threshold
  - `ensemble_boost_capped_for_multi_operation_candidates` — verifies ensemble boost is reduced for complex candidates
  - `single_op_ensemble_boost_unchanged` — verifies single-op ensemble boost is unaffected
- All existing tests pass (including ensemble scoring tests from Issue #572)
- `quality.sh` passes cleanly
