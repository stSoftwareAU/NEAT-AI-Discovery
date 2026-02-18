# PR Summary: Input Sensitivity Analysis Module (#435)

## Summary

Part of the "Brilliant but Brittle" initiative (Issue #432). This module analyses how sensitive predictions are to small changes in input observations, identifying inputs with excessive leverage that contribute to brittle predictions.

The module detects two types of brittleness patterns:

1. **Dominant Inputs**: Input neurons with disproportionate leverage on predictions
2. **Threshold Effects**: Regions where small input changes cause large output changes

## Changes

### New Files

- `src/analysis/input_sensitivity.rs` - Core detection and candidate generation logic
- `tests/issue_435_input_sensitivity_analysis.rs` - 14 comprehensive test cases

### Modified Files

- `src/analysis/mod.rs` - Integration with discovery dispatch system

## Detection Criteria

### Dominant Inputs

An input neuron is flagged as dominant when:
- High sensitivity: Small input changes cause disproportionate output changes
- High leverage ratio: Input's variance contribution exceeds expected proportion
- Weight amplification: Large weights amplify input variance into output variance

### Threshold Effects

A threshold effect is detected when:
- Steep gradient: Large activation gradient in the operating region
- Threshold proximity: Operating point near steep activation regions
- Prediction flip potential: Small changes could flip predictions

## Candidate Proposals

| Candidate Type | Use Case |
|---------------|----------|
| `setWeight` | Reduce weight of excessive sensitivity connections |
| `addNeuron` | Add dampening/smoothing neuron to reduce sharp transitions |
| `setBias` | Shift operating point away from threshold regions |

## Configuration

| Environment Variable | Purpose | Default |
|---------------------|---------|---------|
| `NEAT_AI_DISCOVERY_DOMINANCE_THRESHOLD` | Leverage ratio threshold for dominant input detection | 2.0 |
| `NEAT_AI_DISCOVERY_GRADIENT_THRESHOLD` | Gradient magnitude threshold for threshold effect detection | 10.0 |

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. The functionality is verified through unit tests.

## Test Plan

14 test cases covering:

1. `test_detects_dominant_input_with_high_leverage` - Detects inputs with excessive leverage
2. `test_does_not_flag_balanced_inputs` - Balanced inputs are not flagged
3. `test_detects_threshold_effect` - Detects steep gradient threshold effects
4. `test_zero_variance_input_handled` - Constant inputs handled gracefully
5. `test_minimum_samples_required` - Minimum sample requirements enforced
6. `test_dominant_input_produces_setweight_candidate` - Correct candidate generation
7. `test_threshold_effect_produces_addneuron_candidate` - AddNeuron for dampening
8. `test_threshold_effect_produces_setbias_candidate` - SetBias for operating point shift
9. `test_sensitivity_scores_are_normalised` - Normalised metrics for comparison
10. `test_configurable_sensitivity_threshold` - Config thresholds work correctly
11. `test_candidate_includes_diagnostic_comment` - Diagnostic comments included
12. `test_multiple_dominant_inputs_sorted` - Candidates sorted by improvement
13. `test_hidden_neurons_excluded_from_dominant_detection` - Only input neurons flagged
14. `test_empty_records_handled` - Empty input handled without panic

All tests pass with `cargo test --test issue_435_input_sensitivity_analysis`.
