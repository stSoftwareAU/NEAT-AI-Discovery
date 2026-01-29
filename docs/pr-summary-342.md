## Summary

Implements saturated neuron detection (Issue #342) to identify neurons that are permanently stuck at their activation ceiling or floor. Saturated neurons pass no gradient information and block learning. The detection scans all hidden neurons during `analyze_all` post-processing and recommends activation function changes (`ChangeSquash` to IDENTITY) or bias adjustments (`SetBias`) as coordinated structural candidates.

Key design decisions:
- **Reuses existing `coordinatedStructural` candidate type** — no changes needed in NEAT-AI
- **Runs as a post-processing step** in `analyze_all`, using the shared `RecordCache` to retrieve recorded activations for each hidden neuron
- **Supports bounded activations** (TANH, LOGISTIC, HARD_TANH, BIPOLAR_SIGMOID, SOFTSIGN, ISRU, ARCTAN, RELU6) and **RELU dead-zone** detection
- **Requires minimum 20 samples** to avoid false positives from insufficient data
- **Excludes unbounded activations** (IDENTITY, RELU for ceiling) since they cannot saturate

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. All behaviour is verified through unit tests.

## Test Plan

Added 13 tests in `tests/issue_342_saturated_neuron_detection.rs`:

1. `test_detects_tanh_saturated_at_positive_ceiling` — TANH with mean activation > 0.95
2. `test_detects_tanh_saturated_at_negative_floor` — TANH with mean activation < -0.95
3. `test_does_not_flag_non_saturated_tanh` — TANH in active region is not flagged
4. `test_detects_logistic_saturated_at_ceiling` — LOGISTIC output ≈ 1.0
5. `test_detects_logistic_saturated_at_floor` — LOGISTIC output ≈ 0.0
6. `test_detects_hard_tanh_clamped` — HARD_TANH clamped at +1.0
7. `test_unbounded_activations_not_flagged` — RELU and IDENTITY are not flagged
8. `test_insufficient_samples_not_flagged` — Too few samples (< 20) are ignored
9. `test_candidates_produce_coordinated_operations` — Correct JSON output with ChangeSquash/SetBias
10. `test_mixed_neurons_only_saturated_detected` — Only saturated neurons flagged in mixed set
11. `test_input_neurons_excluded` — IDENTITY-squash input neurons are not flagged
12. `test_detects_relu_dead_at_zero` — Dead RELU neuron (all outputs = 0) is detected
13. `test_bias_adjustment_direction` — Bias delta is negative for positive saturation

All existing tests continue to pass. `./quality.sh` passes cleanly.
