## Summary

Implements **Unbounded Capping Detection** (Issue #441), a new discovery type that identifies neurons with unbounded activation functions (RELU, IDENTITY, LEAKYRELU, etc.) producing high activations ("spiking") and recommends capping them with bounded versions like RELU6.

### The Problem

Unbounded activation functions like RELU can produce arbitrarily high values. When a neuron consistently outputs very high activations, it may introduce noise into the network. Capping these activations with RELU6 (which clips outputs to [0, 6]) can reduce this noise and improve the creature's score.

### Solution

The new discovery module:

1. **Detects spiking neurons**: Identifies hidden neurons using unbounded activations where:
   - Maximum activation exceeds the capping threshold (> 6.0 for RELU family)
   - At least 30% of samples exceed the threshold (consistent spiking, not occasional)

2. **Recommends bounded replacements**:
   - RELU → RELU6
   - LEAKYRELU → RELU6
   - IDENTITY → RELU6 (high positive) or HARD_TANH (mixed)
   - Other unbounded functions → appropriate bounded versions

3. **Emits coordinated structural candidates** with `changeSquash` operations

### Files Changed

| File | Description |
|------|-------------|
| `src/analysis/unbounded_capping.rs` | New detection module with `detect_unbounded_capping_candidates()` and `unbounded_capping_to_coordinated_candidates()` |
| `src/analysis/mod.rs` | Register new module and add dispatch in `analyze_all()` |
| `tests/issue_441_unbounded_capping.rs` | 8 comprehensive unit tests |
| `docs/DISCOVERY_TYPES.md` | Documentation for the new discovery type |

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface. The feature is validated through unit tests.

## Test Plan

Added 8 unit tests covering:

- `test_relu_spiking_detected` — RELU neurons with consistently high activations are detected
- `test_identity_spiking_detected` — IDENTITY neurons with high activations are detected
- `test_output_neurons_excluded` — Output neurons are not considered (hidden only)
- `test_bounded_activations_excluded` — Already-bounded activations (RELU6, TANH, LOGISTIC) are excluded
- `test_conversion_to_coordinated_candidates` — Detection results convert to proper coordinated candidates
- `test_insufficient_samples_skipped` — Neurons with < 20 samples are skipped
- `test_leakyrelu_spiking_detected` — LEAKYRELU spiking detection works
- `test_occasional_spikes_not_detected` — Occasional high activations (< 30% of samples) don't trigger detection

All tests pass:
```
running 8 tests
test test_insufficient_samples_skipped ... ok
test test_identity_spiking_detected ... ok
test test_leakyrelu_spiking_detected ... ok
test test_conversion_to_coordinated_candidates ... ok
test test_bounded_activations_excluded ... ok
test test_output_neurons_excluded ... ok
test test_occasional_spikes_not_detected ... ok
test test_relu_spiking_detected ... ok

test result: ok. 8 passed; 0 failed; 0 ignored
```
