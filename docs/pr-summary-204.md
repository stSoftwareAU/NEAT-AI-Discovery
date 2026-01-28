## Summary

This PR implements Issue #204: Include activation frequency in focus neuron ranking.

**What was changed:**
- Added `activation_frequency` field to `RankedNeuron` struct to track the proportion of samples where a neuron fires
- Implemented `activation_frequency_from_records()` function to calculate activation frequency as count_nonzero_activations / total_samples (where "fires" means |activation| > 1e-6)
- Implemented `compute_frequency_factor()` that returns 0.8 for extreme frequencies (< 10% or > 90%) and 1.0 for moderate frequencies
- Applied the frequency factor to the ranking score calculation in both `rank_focus_neurons` and `rank_focus_neurons_with_history`

**Why this matters:**
- Neurons that rarely fire (< 10% activation rate) have limited influence on most samples - they only contribute information on a small subset of training data
- Neurons that always fire (> 90% activation rate) behave like constants with no discriminative power - they output similar values regardless of input
- The "sweet spot" (10-90% activation rate) indicates neurons with good discriminative power that respond differently to different inputs

**The new ranking formula:**
```rust
score = total_error × (impact + epsilon)^gamma × gradient_factor × frequency_factor
```

Where `frequency_factor`:
- = 0.8 if activation_frequency < 0.1 (20% penalty for rarely-firing neurons)
- = 0.8 if activation_frequency > 0.9 (20% penalty for always-firing neurons)
- = 1.0 otherwise (no penalty for moderate-frequency neurons)

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

The feature is verified through comprehensive unit tests that confirm:
1. Rarely-firing neurons are correctly identified and rank lower
2. Always-firing neurons are correctly identified and rank lower
3. Moderate-frequency neurons receive no penalty
4. Edge cases (never fires, always fires) are handled gracefully

## Test Plan

Added tests in `tests/issue_204_activation_frequency_ranking.rs`:

- `test_rarely_firing_neuron_is_penalised` - Verifies neurons with < 10% activation rate are penalised and rank lower than moderate-frequency neurons
- `test_always_firing_neuron_is_penalised` - Verifies neurons with > 90% activation rate are penalised and rank lower than moderate-frequency neurons
- `test_moderate_frequency_neuron_not_penalised` - Verifies neurons with 10-90% activation rate (at boundaries and middle) receive no penalty
- `test_frequency_factor_integrates_with_error_impact_calculation` - Verifies the frequency factor correctly combines with existing error × impact ranking
- `test_never_firing_neuron_handled_gracefully` - Edge case: 0% activation rate
- `test_all_firing_neuron_handled_gracefully` - Edge case: 100% activation rate

All existing tests continue to pass, confirming backward compatibility.
