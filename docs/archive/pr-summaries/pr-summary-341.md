## Summary

Implements dead neuron detection (Issue #341) — a new discovery category that identifies
hidden neurons with near-zero activation across all samples and recommends their removal.

Dead neurons waste GPU computation during both training and inference without contributing
useful information to the network's output. This commonly occurs when weight changes push
a neuron's inputs to always land in the zero region of its activation function (e.g., RELU
neurons that never receive positive input).

### Detection Criteria

A neuron is considered "dead" if:
1. **Near-zero activation**: Mean absolute activation < 1e-6 across all samples
2. **Zero variance**: Activation standard deviation < 1e-6
3. **No meaningful activity**: Fewer than 1% of samples show activation above 0.01
4. **Minimum samples**: At least 20 samples required for reliable detection
5. **Hidden neurons only**: Output and input neurons are excluded

### Implementation

- New module `src/analysis/dead_neuron.rs` following the established pattern from
  saturated neuron detection (Issue #342) and bottleneck detection (Issue #343)
- Integrated as a post-processing pass in `analyze_all()` after bottleneck detection
- Emits `RemoveNeuron` operations via `CoordinatedStructuralCandidateJson`
- No new candidate types needed — reuses existing coordinated structural mechanism

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

Added 13 tests in `tests/issue_341_dead_neuron_detection.rs`:

1. `test_detects_all_zero_activation_neuron` — All-zero activation detected as dead
2. `test_detects_near_zero_activation_neuron` — Near-zero (1e-8) activation detected
3. `test_does_not_flag_active_neuron` — Active neuron (mean ~0.5) not flagged
4. `test_rarely_active_neuron_not_flagged` — Neuron active on 5% of samples not flagged (false positive prevention)
5. `test_constant_tiny_activation_is_dead` — Constant 1e-8 activation detected
6. `test_output_neurons_not_flagged` — Output neurons excluded from detection
7. `test_insufficient_samples_not_flagged` — Fewer than 20 samples skipped
8. `test_mixed_neurons_only_dead_detected` — Only dead neurons detected in mixed set
9. `test_candidates_produce_coordinated_removal_operations` — Correct `removeNeuron` operation emitted
10. `test_detects_negative_near_zero_activation` — Negative near-zero (LeakyRELU) detected
11. `test_zero_mean_high_variance_not_dead` — Bipolar activation (zero mean, high variance) not flagged
12. `test_sample_count_recorded` — Sample count correctly recorded in candidate
13. `test_connected_outputs_identified` — Downstream output neurons correctly found via BFS

All existing tests continue to pass. `./quality.sh` passes cleanly.
