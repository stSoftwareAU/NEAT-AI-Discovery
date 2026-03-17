## Summary

Improve add-neurons prediction accuracy by implementing two complementary improvements from Issue #791:

1. **Neuron-specific pessimism calibration** — The generic pessimism discount (floor=0.15, exponent=0.6) treats neuron and synapse candidates identically, but GRQ-sampler data shows add-neurons has a 15% success rate (3,812/25,812) vs higher synapse rates. New neuron-specific constants (floor=0.10, exponent=0.75) apply more aggressive discounting: ~28% more aggressive at low ratios, ~13% at moderate ratios.

2. **Cross-validation brittleness filtering** — Integrates the existing cross-validation infrastructure (from `scoring/cross_validation.rs`) into the neuron evaluation pipeline. Candidates are now evaluated across k-fold sample subsets; those showing inconsistent improvement across folds receive a brittleness penalty that reduces their expected score gain. This filters candidates that overfit to specific data subsets.

Closes #791.

## Evidence

### Neuron-specific pessimism discount values

| Improved ratio | Synapse discount | Neuron discount | Reduction |
|---------------|-----------------|-----------------|-----------|
| 10%           | 0.363           | 0.260           | 28%       |
| 40%           | 0.639           | 0.555           | 13%       |
| 70%           | 0.832           | 0.770           | 7%        |
| 100%          | 1.000           | 1.000           | 0%        |

### Cross-validation integration

The cross-validation brittleness penalty is applied during GPU evaluation of neuron candidates (in `evaluation.rs`). Candidates with high variance across folds receive a penalty factor of up to 0.5 (50% reduction), filtering out overfitting candidates before they reach post-processing.

## Test Plan

Added `tests/issue_791_improve_add_neurons_prediction.rs` with 11 tests:
- `neuron_pessimism_floor_more_aggressive_than_synapse` — Validates neuron floor < synapse floor
- `neuron_pessimism_exponent_less_forgiving_than_synapse` — Validates neuron exponent > synapse exponent
- `neuron_pessimism_constants_in_valid_ranges` — Compile-time range validation
- `neuron_pessimism_more_aggressive_at_all_moderate_ratios` — Neuron discount <= synapse discount at all ratios 5–95%
- `neuron_pessimism_full_ratio_gives_full_gain` — 100% ratio gives full gain
- `neuron_pessimism_monotonically_increasing` — Monotonicity invariant
- `neuron_pessimism_zero_total_gives_floor` — Edge case: zero total
- `neuron_pessimism_practical_difference` — Meaningful reduction at typical 45% ratio
- `cross_validation_penalises_brittle_neuron_candidates` — Brittle samples get penalty > 0
- `cross_validation_does_not_penalise_consistent_candidates` — Consistent samples get low penalty
- `cross_validation_skips_with_insufficient_samples` — Graceful skip with too few samples
