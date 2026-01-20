## Summary

Implements Issue #306: Constant-value neuron detection with bias adjustment for removal candidates.

When a hidden neuron has near-zero activation variance (constant output), it can be removed and its effect folded into bias adjustments for downstream neurons. This is a behaviour-preserving simplification that reduces network complexity.

### Key Changes

1. **Added `constant_neuron_removals` field** to `RankFocusStats` and `RankFocusNeuronsOutput` - returns coordinated structural candidates for removing constant neurons with bias adjustments.

2. **Added `activation_mean_and_variance_from_records()` function** in `focus.rs` - computes mean and variance of neuron activations to detect constant neurons.

3. **Constant neuron detection logic** - Hidden neurons with variance below `1e-10` (as proposed in Issue #217) are identified as constant and returned as removal candidates.

4. **Bias adjustment calculation** - For each downstream neuron, the bias is adjusted by `weight × mean_activation` to preserve network behaviour when the constant neuron is removed.

### How It Works

When `rank_focus_neurons` is called, it now:
1. Computes activation variance for each selectable neuron
2. Identifies hidden neurons with variance below `CONSTANT_VARIANCE_THRESHOLD` (1e-10)
3. For each constant neuron, creates a `CoordinatedStructuralCandidateJson` containing:
   - `SetBias` operations for all downstream neurons (with adjusted bias values)
   - `RemoveNeuron` operation for the constant neuron

This addresses Issue #217's proposal that "dead neurons" (neurons with near-zero variance) waste computation and should be candidates for removal with bias adjustments.

## Evidence

This is a backend change with no UI. The feature is verified through automated tests.

## Test Plan

Added 4 new tests in `tests/issue_306_constant_neuron_removal_with_bias_adjustment.rs`:

- `issue_306_constant_neuron_creates_removal_candidate_with_bias_adjustment` - Verifies constant neurons are detected and returned with correct bias adjustments
- `issue_306_varying_neuron_not_treated_as_constant` - Ensures neurons with normal variance are not incorrectly flagged
- `issue_306_constant_neuron_adjusts_multiple_downstream_biases` - Tests bias adjustments for neurons with multiple downstream connections
- `issue_306_near_zero_variance_treated_as_constant` - Verifies the 1e-10 threshold correctly identifies near-constant neurons

All existing tests continue to pass (576 tests total).
