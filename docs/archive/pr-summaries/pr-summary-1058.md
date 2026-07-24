## Summary

Simplify the coordinated-structural compound discounting pipeline to improve candidate success rate. The previous three-layer compound discount (per-operation exponential, flat pessimism discount, and calibration factor) was too aggressive, filtering out potentially viable candidates while remaining poorly calibrated. Replaces the compound model with a single empirical discount per operation count derived from production success rates, and lowers the minimum gain threshold since the calibration factor already accounts for overestimation. Closes #1058.

## Changes

### Simplified Discount Model
- **Replaced** `COORDINATED_OPERATION_DISCOUNT^(N-1) * COORDINATED_PESSIMISM_DISCOUNT` (three-layer compound) with per-op-count empirical factors:
  - 1 op: 1.0 (no discount)
  - 2 ops: 0.5 (was 0.65 * 0.15 = 0.0975 -- 5x less aggressive)
  - 3 ops: 0.2 (was 0.4225 * 0.15 = 0.0634 -- 3x less aggressive)
  - 4+ ops: 0.1 (was 0.274 * 0.15 = 0.0411 -- 2.4x less aggressive)
- **Added** `coordinated_empirical_discount(op_count)` function for clean lookup
- **Removed** separate `COORDINATED_PESSIMISM_DISCOUNT` application from post-processing (folded into empirical factors)
- **Lowered** `MIN_COORDINATED_MULTI_OP_GAIN` from 1e-3 to 1e-5 (calibration factor already handles overestimation)
- **Preserved** `COORDINATED_PREDICTION_CALIBRATION` (unchanged -- bridges prediction-to-reality magnitude gap)
- **Added** deprecated aliases for backward compatibility of old constant names

### Production Discovery-Cache Analysis
- Creature 0e18e62c: 6/389 successes (~1.5%), predominantly 2-op candidates
- Creature 066649c7: 0/519 successes
- Successful candidates were predominantly 2-operation, informing the empirical factors

## Evidence

All 171 tests pass, including new tests verifying:
- Empirical discount factors match expected values per operation count
- Monotonically increasing discount with operation count
- New factors are less aggressive than old compound
- Lowered threshold allows viable candidates through while rejecting truly tiny gains
- Production example candidates pass/fail correctly

## Test Plan

- **New test file**: `tests/synapse/issue_1058_reduce_coordinated_compound_discounting.rs` (17 tests)
  - Empirical discount constants are in valid range
  - Per-op-count factors decrease monotonically
  - 2-op, 3-op, 4-op, 5-op candidates use correct factors
  - Simplified model is less aggressive than old compound
  - Lowered threshold allows moderate gains while rejecting tiny ones
  - Production known example patterns
- **Updated tests** (business logic changed, tests modified to match new behaviour):
  - `tests/analysis/issue_938_constants_submodule_organisation.rs` -- updated constant values
  - `tests/synapse/issue_732_coordinated_structural_success_rate.rs` -- updated threshold comments
  - `tests/synapse/issue_790_reduce_coordinated_structural_false_positives.rs` -- replaced pessimism-specific tests with empirical discount tests
  - `tests/synapse/issue_1017_candidate_pipeline_mcmc_audit.rs` -- replaced pessimism constant check with empirical factors
  - `tests/analysis/issue_921_candidate_compression.rs` -- updated discount calculation and threshold values
  - `tests/analysis/issue_922_nonlinear_candidate_compression.rs` -- updated discount calculation
  - `src/analysis/candidate_compression/identity.rs` (inline tests) -- updated discount and threshold values
