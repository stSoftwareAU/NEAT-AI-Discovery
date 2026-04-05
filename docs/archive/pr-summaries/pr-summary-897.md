## Summary

Replace linear gain estimation in the coordinated candidate pipeline with saturation-aware simulation that uses the target neuron's actual activation function. This addresses the root cause of the 1.9% success rate identified in #890: the current linear approximation `error - contribution_a - contribution_b` is mathematically incorrect for non-linear activation functions (TANH, LOGISTIC, ReLU, etc.). Also applies conservative weight scaling (0.2x) to reported gains to align estimation with evaluation variant weights. Closes #897.

## Changes

### Core estimation (`candidate_generation.rs`)
- `compute_combined_improvement_on_range()` now accepts an optional `target_activation_fn` parameter
- When the target neuron's activation function is available and samples have `target_value`/`target_activation` data, computes improvement in the activation domain by simulating both contributions through the actual activation function
- Falls back to linear approximation when target data is unavailable
- Applies `COORDINATED_ESTIMATION_WEIGHT_SCALE` (0.2x) discount to the reported `combined_improvement` value

### Epistatic detection (`candidate_generation.rs`)
- `detect_epistatic_pairs()` now accepts `target_squash: Option<&str>` parameter
- Resolves the activation function via `crate::activations::target_simulation_fn()`
- Passes the function through the call chain: `evaluate_pair_for_epistasis()` -> `compute_combined_improvement_from_samples()` -> `compute_combined_improvement_on_range()`, and `cross_validate_pair()`

### Synergistic detection (`pre_screening.rs`)
- `detect_synergistic_candidates()` now accepts `target_squash: Option<&str>` parameter
- `compute_residual_errors()` uses activation-aware simulation when available, computing residuals in the activation domain
- `evaluate_residual_reduction()` simulates complement contributions through the target activation function
- Applies `COORDINATED_ESTIMATION_WEIGHT_SCALE` (0.2x) discount to reported synergistic gain

### Caller updates (`candidate_selection.rs`)
- `detect_epistatic_and_synergistic()` retrieves target squash from `ctx.neuron_squash_map` and passes it to both detection functions

### Constants (`constants.rs`)
- Added `COORDINATED_ESTIMATION_WEIGHT_SCALE = 0.2` constant documenting alignment with `COORDINATED_CONSERVATIVE_WEIGHT_SCALE` in `variant_generation.rs`

## Evidence

The saturation-aware simulation correctly models non-linear activation effects:
- TANH saturation test: when both sources push target_value near +2 (deep in TANH saturation), the activation-aware estimate does not exceed the linear estimate, correctly reflecting diminishing returns
- LOGISTIC saturation test: deep saturation (target_value=5.0) produces finite improvement estimates
- IDENTITY test: activation-aware and linear paths produce equivalent results for the linear activation function
- Fallback tests: graceful degradation when target data or squash info is unavailable

## Test Plan

Added 6 new tests in `tests/synapse/issue_897_saturation_aware_coordinated_estimation.rs`:
- `tanh_saturation_reduces_estimated_gain_vs_linear` — verifies TANH saturation produces conservative estimates
- `logistic_saturation_caps_improvement` — verifies LOGISTIC deep saturation is handled
- `identity_activation_uses_conservative_weight_scale` — verifies IDENTITY produces equivalent results to linear
- `linear_fallback_works_without_squash_info` — verifies fallback when squash is None
- `falls_back_to_linear_when_samples_lack_target_data` — verifies fallback when samples lack target data
- `synergistic_detection_uses_saturation_aware_simulation` — verifies synergistic path uses activation simulation

All existing tests updated to pass `target_squash` parameter (None for backward compatibility). All 137 synapse tests + 42 neuron tests pass.
