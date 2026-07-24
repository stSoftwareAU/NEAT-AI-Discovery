## Summary

Fix the add-synapses module's 0% success rate in production cache (production discovery-cache data). Closes #730.

Three root causes were identified and addressed:

1. **Single-weight evaluation** — For non-saturating targets, only one computed weight was tried. If that weight was suboptimal (common with noisy data), there was no fallback. Now all new synapse candidates use multi-weight search (9 scaled variants), matching the approach already used for saturating targets (Issue #413).

2. **Neuron-level vs creature-level prediction mismatch** — Improvement was measured as a fraction of a single target neuron's error but used directly as the creature-level prediction. Added error fraction scaling: predictions are now multiplied by `target_error / total_creature_error` to produce calibrated creature-level estimates within an order of magnitude of actual results.

3. **Insufficient candidate filtering** — Candidates where more samples worsened than improved were still being proposed. Added `MIN_IMPROVED_RATIO` threshold (0.5) requiring at least 50% of samples to show improvement before a candidate is accepted.

## Changes

- `src/analysis/synapse/target_analysis/evaluation.rs` — Unified multi-weight search for ALL new synapse candidates (removed saturating-target-only conditional). Added `MIN_IMPROVED_RATIO` filter.
- `src/analysis/synapse/post_processing.rs` — Added `scale_by_error_fraction()` for creature-level calibration. Added `compute_neuron_error_sq_map()` to compute per-neuron error from cache. Applied error fraction scaling in `apply_impact_to_helpful()`.
- `src/analysis/constants.rs` — Added `MIN_IMPROVED_RATIO = 0.5` constant.
- `src/analysis/synapse/mod.rs` — Made `post_processing` module public. Added Issue #730 unit tests.
- `src/analysis/activation/simulation.rs` — Removed unused `is_saturating_target()` function (superseded by always-on multi-weight search).
- `tests/issue_134_direction_flip.rs` — Relaxed diagnostics assertion that tested implementation detail affected by multi-weight search change.

## Evidence

This is a backend/algorithmic change with no UI. Evidence:
- 8 new tests validate the improved prediction logic (3 unit + 5 integration)
- All existing tests pass (including the issue #134 regression test's core assertion)
- `quality.sh` passes cleanly

## Test Plan

- `tests/issue_730_synapse_prediction_calibration.rs` — 5 integration tests:
  - `test_issue_730_error_fraction_scaling_reduces_prediction` — Verifies scaling reduces raw predictions
  - `test_issue_730_error_fraction_preserves_dominant_target` — Verifies dominant targets preserve prediction
  - `test_issue_730_error_fraction_zero_total_returns_zero` — Edge case: zero total error
  - `test_issue_730_error_fraction_clamps_to_one` — Edge case: numerical overflow
  - `test_issue_730_min_improved_ratio_is_sensible` — Constant validation
- `src/analysis/synapse/mod.rs` — 3 unit tests:
  - `test_issue_730_multi_weight_search_finds_better_improvement` — Multi-weight >= single weight
  - `test_issue_730_multi_weight_improves_ratio` — Multi-weight improves improved/worsened ratio
  - `test_issue_730_worsened_exceeds_improved_poor_score` — Poor candidates get low scores
