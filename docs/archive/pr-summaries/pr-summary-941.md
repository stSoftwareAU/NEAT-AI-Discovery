## Summary

Extract common patterns from detection modules into shared helpers in
`src/analysis/detection/helpers.rs`, reducing duplication across the 42+
detection modules. Closes #941.

Five new shared helpers were added:

| Helper | Pattern | Modules refactored |
|--------|---------|-------------------|
| `compute_activation_stats` | Mean, variance, std dev from records | saturation, dead_neuron, low_impact_neuron |
| `compute_activation_range` | Min/max activation from records | restricted_range |
| `compute_mean_abs_activation` | Mean absolute activation from records | dead_neuron, low_impact_neuron, oscillating_neuron |
| `sort_candidates_by_score_gain` | Descending sort on score gain | saturation, dead_neuron, low_impact_neuron, oscillating_neuron, noise_signal, bottleneck, bounded_range, restricted_range |
| `weighted_confidence` | Multi-factor confidence scoring with floor/ceiling | dead_neuron, low_impact_neuron, bounded_range |

Eight detection modules were refactored to use the shared helpers:
1. `saturation.rs` -- `compute_activation_stats`, `sort_candidates_by_score_gain`
2. `dead_neuron.rs` -- `compute_activation_stats`, `compute_mean_abs_activation`, `sort_candidates_by_score_gain`, `weighted_confidence`
3. `low_impact_neuron.rs` -- `compute_activation_stats`, `compute_mean_abs_activation`, `sort_candidates_by_score_gain`, `weighted_confidence`
4. `oscillating_neuron.rs` -- `compute_mean_abs_activation`, `sort_candidates_by_score_gain`
5. `noise_signal.rs` -- `sort_candidates_by_score_gain`
6. `bottleneck.rs` -- `sort_candidates_by_score_gain`
7. `bounded_range.rs` -- `sort_candidates_by_score_gain`, `weighted_confidence`
8. `restricted_range.rs` -- `compute_activation_range`, `sort_candidates_by_score_gain`

No behavioural changes -- all 466 existing detection tests pass unchanged.

## Evidence

- `./quality.sh` passes with no warnings
- All 466 existing detection tests pass unchanged
- 21 new unit tests verify the helper functions

## Test Plan

- Added `tests/detection/issue_941_detection_helpers.rs` with 21 tests:
  - 5 tests for `compute_activation_stats` (typical, constant, empty, single, negative)
  - 3 tests for `compute_activation_range` (typical, constant, empty)
  - 3 tests for `compute_mean_abs_activation` (mixed, zeros, empty)
  - 3 tests for `sort_candidates_by_score_gain` (descending, empty, single)
  - 7 tests for `weighted_confidence` (single factor, multiple, all max, all zero, empty, clamped, custom range)
- Verified all 466 existing detection tests pass without modification
