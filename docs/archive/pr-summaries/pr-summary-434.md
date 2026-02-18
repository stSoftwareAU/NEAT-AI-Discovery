## Summary

Implements high noise-to-signal ratio detection for neurons and synapses as part of the "Brilliant but Brittle" initiative (Issue #432). This discovery module identifies network elements that amplify noise rather than contributing meaningful signal, helping reduce brittle predictions when bad or missing observations occur.

### Changes

**New Module**: `src/analysis/noise_signal.rs`
- Detects noisy neurons with high error variance relative to activation variance
- Detects noisy synapses that amplify noise from upstream neurons
- Proposes `removeNeuron`, `removeSynapse`, or `setWeight` candidates
- Configurable threshold via `NEAT_AI_DISCOVERY_NOISE_SIGNAL_THRESHOLD` environment variable (default: 2.0)

**Integration**:
- Added module to `src/analysis/mod.rs`
- Integrated with discovery dispatch system (separate detection for neurons and synapses)

**Documentation**:
- Updated `docs/DISCOVERY_TYPES.md` with new discovery type details

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. The implementation is validated through comprehensive unit tests.

## Test Plan

Created `tests/issue_434_noise_signal_ratio_detection.rs` with 13 test cases covering:

1. **Noisy Neuron Detection**:
   - `test_detects_high_noise_low_signal_neuron` - Detects neurons with high error variance but low activation variance
   - `test_does_not_flag_good_signal_neuron` - Neurons with good signal-to-noise ratio are not flagged
   - `test_input_output_neurons_not_flagged` - Only hidden neurons are considered
   - `test_minimum_samples_required` - Requires 20+ samples for detection

2. **Noisy Synapse Detection**:
   - `test_detects_noise_amplifying_synapse` - Detects synapses with large weights from noisy sources
   - `test_small_weight_synapse_not_flagged` - Small weight synapses don't amplify noise
   - `test_synapse_between_noisy_neurons` - Handles synapse chains

3. **Candidate Generation**:
   - `test_noisy_neuron_produces_remove_candidate` - Produces correct `removeNeuron` operations
   - `test_noisy_synapse_produces_remove_or_setweight_candidate` - Produces correct operations
   - `test_setweight_recommendation_for_noise_reduction` - Weight reduction for partially useful synapses
   - `test_multiple_noisy_neurons_sorted_by_improvement` - Sorted by estimated improvement

4. **Edge Cases**:
   - `test_threshold_respects_environment_variable` - Configurable threshold
   - `test_candidate_includes_diagnostic_comment` - Includes diagnostic information

All tests pass with `cargo test --test issue_434_noise_signal_ratio_detection`.
