## Summary

This PR implements gradient flow analysis for focus neuron selection (Issue #206), enhancing the discovery algorithm's ability to identify neurons with high learning potential.

### Problem

Previously, focus neurons were ranked solely by `error × impact`, which could miss neurons that are "stuck" in saturation or have "dead" ReLU neurons with zero gradient. These neurons cannot effectively learn, making them poor candidates for discovery analysis.

### Solution

Added gradient flow metrics to the focus neuron ranking:

1. **GradientFlowStats struct**: New struct containing three metrics:
   - `avg_gradient_magnitude`: Average |f'(value)| across samples - higher values indicate better error signal propagation
   - `saturation_ratio`: Fraction of samples in saturated region (gradient near zero)
   - `dead_ratio`: Fraction of samples with zero gradient (ReLU dead zones)

2. **Activation-specific gradient computation**: Implemented `compute_activation_gradient()` supporting 30+ activation functions including:
   - ReLU family: RELU, RELU6, LEAKYRELU, ELU, SELU
   - Sigmoid family: TANH, LOGISTIC, HARD_TANH, BIPOLAR_SIGMOID, SOFTSIGN
   - Other: GELU, SWISH, MISH, SOFTPLUS, IDENTITY, and more

3. **Saturation detection**: Function-specific thresholds:
   - TANH: |value| > 3 (gradient < 0.01)
   - LOGISTIC: |value| > 5
   - HARD_TANH: |value| >= 1 (clamped)
   - ARCTAN/SOFTSIGN: |value| > 10

4. **Dead neuron detection**: Identifies ReLU neurons with negative inputs that produce zero output and zero gradient.

5. **Ranking integration**: The new `compute_gradient_flow_factor()` adjusts ranking scores:
   ```
   factor = (1 - saturation_ratio) × (1 - dead_ratio) × gradient_boost
   ```
   Where `gradient_boost = 0.5 + 0.5 × clamp(avg_gradient_magnitude, 0, 1)`

### Key Changes

- `src/focus.rs`:
  - Added `GradientFlowStats` struct with documentation
  - Added `compute_activation_gradient()` for 30+ activation functions
  - Added `is_saturated()` with function-specific thresholds
  - Added `compute_gradient_flow_stats()` for parquet file analysis
  - Added `compute_gradient_flow_for_neuron()` for per-neuron stats
  - Added `compute_gradient_flow_factor()` for ranking adjustment
  - Updated `RankedNeuron` to include `gradient_flow` field
  - Updated `rank_focus_neurons()` and `rank_focus_neurons_with_history()` to use gradient flow in ranking

- `tests/issue_206_gradient_flow_analysis.rs`:
  - 12 comprehensive tests covering:
    - TANH saturation detection (high/low)
    - ReLU dead neuron detection
    - LeakyReLU never-dead behaviour
    - Gradient magnitude calculation
    - LOGISTIC saturation detection
    - Integration with focus ranking
    - Edge cases (IDENTITY, aggregate neurons)

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. The changes are validated through comprehensive unit tests.

### Test Results

All 12 new tests pass, plus all 372 existing tests continue to pass:

```
test test_tanh_saturation_detection_high_saturation ... ok
test test_tanh_saturation_detection_low_saturation ... ok
test test_relu_dead_neuron_detection_mostly_dead ... ok
test test_relu_active_neuron_detection ... ok
test test_leaky_relu_never_dead ... ok
test test_gradient_magnitude_calculation ... ok
test test_logistic_saturation_detection ... ok
test test_gradient_flow_integration_with_ranking ... ok
test test_dead_relu_deprioritised_in_ranking ... ok
test test_gradient_flow_stats_struct_fields ... ok
test test_identity_always_full_gradient ... ok
test test_aggregate_neurons_handled_gracefully ... ok
```

### Expected Impact

- **Better discovery targeting**: Neurons with high learning potential (good gradient flow) are prioritised
- **Avoid wasted analysis**: Saturated or dead neurons that cannot meaningfully change are de-prioritised
- **Complement existing ranking**: Gradient flow factor works alongside error × impact ranking

## Test Plan

1. **Unit tests added** (`tests/issue_206_gradient_flow_analysis.rs`):
   - `test_tanh_saturation_detection_high_saturation`: Verifies TANH neurons with |value| > 3 are detected as saturated
   - `test_tanh_saturation_detection_low_saturation`: Verifies TANH neurons in linear region have low saturation ratio
   - `test_relu_dead_neuron_detection_mostly_dead`: Verifies ReLU with negative inputs has high dead ratio
   - `test_relu_active_neuron_detection`: Verifies active ReLU has low dead ratio
   - `test_leaky_relu_never_dead`: Verifies LeakyReLU always has zero dead ratio (never truly dead)
   - `test_gradient_magnitude_calculation`: Verifies gradient magnitude is higher for non-saturated neurons
   - `test_logistic_saturation_detection`: Verifies LOGISTIC saturation detection at |value| > 5
   - `test_gradient_flow_integration_with_ranking`: Verifies saturated neurons rank lower than active ones
   - `test_dead_relu_deprioritised_in_ranking`: Verifies dead ReLU neurons rank lower than active ones
   - `test_gradient_flow_stats_struct_fields`: Verifies struct field access
   - `test_identity_always_full_gradient`: Verifies IDENTITY has gradient=1.0, never saturated/dead
   - `test_aggregate_neurons_handled_gracefully`: Verifies aggregate neurons (MINIMUM) get neutral stats

2. **Quality checks passed**:
   - All 384 tests pass (372 existing + 12 new)
   - Clippy lints pass with `-D warnings`
   - Release build succeeds
