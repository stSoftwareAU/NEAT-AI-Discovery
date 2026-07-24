## Summary

Improve add-synapses success rate by adding synapse-specific pessimism discounting and raising the MIN_IMPROVED_RATIO threshold. Closes #789.

### Root Cause Analysis

Production discovery cache (Issue #787) showed add-synapses had a **0% success rate** (0/31 candidates). Investigation revealed two contributing factors:

1. **Generic pessimism discount too generous**: Synapse candidates used the same generic pessimism parameters (floor=0.15, exponent=0.6) as the overall pool, despite having the worst success rate. Neuron candidates already received more aggressive discounting (floor=0.10, exponent=0.75) via Issue #791, but synapses still used generic parameters.

2. **MIN_IMPROVED_RATIO threshold too low**: All 31 candidates passed the 0.5 threshold but still failed ablation. The multi-weight search (9 weight variants) creates selection bias — picking the best weight for sample data that does not generalise to full evaluation.

### Changes

- **Added synapse-specific pessimism constants** (`SYNAPSE_PESSIMISM_DISCOUNT_FLOOR = 0.05`, `SYNAPSE_PESSIMISM_CURVE_EXPONENT = 0.85`) — the most aggressive of all candidate types
- **Added `apply_synapse_pessimism_discount()`** function using the synapse-calibrated parameters
- **Raised `MIN_IMPROVED_RATIO`** from 0.5 to 0.6 — requires 60% of samples to improve before a synapse candidate is accepted
- **Updated synapse post-processing** to use `apply_synapse_pessimism_discount()` instead of generic `apply_pessimism_discount()` for both helpful and harmful candidates
- **Updated two existing tests** to provide sufficient sample data for the raised threshold

### Discount Comparison

| Ratio | Generic (0.15/0.6) | Neuron (0.10/0.75) | **Synapse (0.05/0.85)** |
|-------|--------------------|--------------------|------------------------|
| 0.1   | 0.363              | 0.260              | **0.184**              |
| 0.4   | 0.639              | 0.555              | **0.480**              |
| 0.6   | 0.766              | 0.688              | **0.659**              |
| 0.7   | 0.832              | 0.770              | **0.754**              |
| 1.0   | 1.000              | 1.000              | **1.000**              |

## Evidence

- 12 new tests verify synapse pessimism constants, function behaviour, and MIN_IMPROVED_RATIO threshold
- All existing tests pass with the raised threshold (2 tests updated with additional sample data)
- `quality.sh` passes cleanly

## Test Plan

- Added `tests/issue_789_improve_add_synapses_success_rate.rs` with 11 tests:
  - Synapse pessimism floor more aggressive than generic and neuron
  - Synapse pessimism exponent less forgiving than generic and neuron
  - Constants within valid ranges
  - Discount more aggressive at all moderate ratios (5%–95%)
  - Full ratio gives full gain
  - Monotonically non-decreasing
  - Zero total gives floor discount
  - Practical difference at 60% ratio
  - MIN_IMPROVED_RATIO in valid range (> 0.5, <= 0.75)
- Updated `tests/issue_134_direction_flip.rs` — added 3 more observations for robust improved ratio
- Updated `tests/synapse_candidate_indices.rs` — added 4 more sample patterns for robust improved ratio
