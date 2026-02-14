## Summary

Enhance the `output_squash_mismatch` detection module with a new pre-activation squash comparison strategy (Strategy 4) that directly simulates alternative activation functions against observed pre-activation values to identify better-fitting squash functions. Closes #546.

The module now has four complementary detection strategies:

1. **Clipping analysis** — detects activations stuck at saturation bounds of hard-clipping functions (e.g., HARD_TANH)
2. **Range mismatch** — detects when activation range doesn't cover the target data range (e.g., LOGISTIC [0,1] with targets in [-1,1])
3. **Unbounded mismatch** — detects when unbounded activations (e.g., IDENTITY) produce high errors outside expected bounds
4. **Pre-activation squash comparison** (NEW) — when pre-activation values are available, evaluates 9 candidate squash functions by simulating their output and selecting the one that best reduces error against the observed target-to-activation mapping

This new strategy directly addresses the issue requirement to "compare pre-activation value distributions against what different squash functions would produce" and catches mismatches between same-range functions (e.g., SOFTSIGN vs TANH) that the clipping-based strategies cannot detect.

Also updated issue references from #545 to #546 throughout the module.

## Evidence

This is a backend detection module with no visual output. Evidence is provided via integration tests.

## Test Plan

- `test_hard_tanh_output_with_smooth_tanh_targets_detected` — verifies HARD_TANH→TANH detection
- `test_tanh_output_with_tanh_targets_no_detection` — verifies no false positive when squash matches
- `test_insufficient_samples_returns_empty` — verifies minimum sample threshold
- `test_hidden_neurons_are_ignored` — verifies only output neurons are analysed
- `test_coordinated_candidate_conversion` — verifies changeSquash candidate generation
- `test_multiple_output_neurons_detected` — verifies multiple neurons handled correctly
- `test_no_preactivation_data_still_analyses` — verifies detection without pre-activation data
- `test_logistic_output_with_symmetric_targets` — verifies LOGISTIC range mismatch detection
- `test_preactivation_comparison_finds_better_squash` (NEW) — verifies Strategy 4 detects SOFTSIGN→TANH mismatch via pre-activation comparison
- `test_preactivation_comparison_no_false_positive` (NEW) — verifies Strategy 4 doesn't fire when squash already fits target data

All 10 tests pass. Full quality gate (`./quality.sh`) passes cleanly.
