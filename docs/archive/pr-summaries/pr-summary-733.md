## Summary

Tune pessimism discount and add neuron candidate filtering to improve add-neurons success rate above the 14.1% baseline. Closes #733.

### Changes

1. **Concave pessimism discount curve** (`src/analysis/constants.rs`, `src/analysis/synapse/scoring.rs`):
   - Replaced the linear pessimism discount formula with a concave (power) curve using `PESSIMISM_CURVE_EXPONENT = 0.6`
   - The concave curve is more forgiving at moderate improved ratios (30-60%), retaining add-neuron candidates with genuine signal
   - Still aggressive at very low ratios (<10%) to filter noise
   - Formula: `adjusted_ratio = ratio^0.6`, then `discount = FLOOR + (1 - FLOOR) × adjusted_ratio`

2. **Neuron candidate improved ratio filtering** (`src/analysis/neuron/evaluation.rs`, `src/analysis/constants.rs`):
   - Added `NEURON_MIN_IMPROVED_RATIO = 0.4` threshold for neuron candidates
   - Filters out neuron candidates where fewer than 40% of samples show improvement
   - Applied to both ReLU and activation spec evaluation paths
   - Reduces candidate volume (fewer low-quality candidates proposed) while retaining higher-quality ones

3. **Updated existing tests** (`tests/issue_506_*.rs`, `tests/issue_522_*.rs`):
   - Modified 3 tests that checked exact linear formula values to work with the new concave curve
   - Business logic change documented: Issue #733 changed pessimism discount from linear to concave

## Evidence

This is a backend/algorithm change with no visual output. Evidence is provided by unit tests:
- 8 new tests validate the concave curve behaviour, constant ranges, and monotonicity
- All 16 existing pessimism discount tests pass (3 updated for new formula)
- `quality.sh` passes cleanly

## Test Plan

- Added `tests/issue_733_add_neurons_pessimism_tuning.rs` with 8 tests:
  - `pessimism_discount_concave_curve_more_forgiving_at_moderate_ratios` — validates concave > linear at 40%
  - `pessimism_discount_still_aggressive_at_very_low_ratios` — validates aggression at 5%
  - `pessimism_discount_monotonically_increasing_with_concave_curve` — monotonicity check
  - `pessimism_discount_full_ratio_gives_full_gain` — ratio=1.0 edge case
  - `pessimism_discount_zero_total_gives_floor_gain` — total=0 edge case
  - `neuron_min_improved_ratio_is_sensible` — const range validation
  - `neuron_min_improved_ratio_not_more_strict_than_synapse` — cross-module consistency
  - `pessimism_discount_practical_add_neuron_improvement` — realistic scenario check
- Modified `tests/issue_506_score_prediction_pessimism_discount.rs` (2 tests updated for concave curve)
- Modified `tests/issue_522_synapse_scoring.rs` (1 test updated for concave curve)
