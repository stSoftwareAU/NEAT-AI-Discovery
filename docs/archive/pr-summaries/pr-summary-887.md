## Summary

Add activation-function-aware boost/penalty multipliers for add-neuron candidate
scoring. Production discovery cache analysis shows dramatic differences in success
rates by activation function (e.g., GELU at 60% vs HARD_TANH at 7.2%), yet previously
all activation functions were treated equally during candidate scoring.

Per-activation boost constants are derived from Bayesian-smoothed success rates
(Beta posterior with K=20 prior centred on the 13.9% baseline) to handle small
sample sizes, then applied as multipliers to `expected_creature_score_gain` during
neuron candidate evaluation. Closes #887.

## Changes

- **`src/analysis/constants.rs`**: Added 14 per-activation boost constants with
  documented Bayesian smoothing methodology, a lookup function
  `activation_neuron_boost()`, and min/max range constants
- **`src/analysis/synapse/scoring.rs`**: Added `apply_activation_neuron_boost()`
  function that applies the boost as a direct multiplier on score gain
- **`src/analysis/synapse/mod.rs`**: Re-exported the new function
- **`src/analysis/neuron/evaluation.rs`**: Applied the activation boost during
  batched activation spec evaluation, after source variance discount and before
  cross-validation penalty

## Evidence

High-success activations (GELU 2.0x, ABSOLUTE 1.95x, Mish 1.78x) receive meaningful
boosts while low-success activations (HARD_TANH 0.80x, BIPOLAR 0.95x) receive
penalties. TANH (13.9% = baseline) receives neutral 1.0x. ReLU candidates are
unaffected (neutral 1.0x) as they are evaluated separately.

## Test Plan

- Added `tests/scoring/issue_887_activation_neuron_boost.rs` with 12 tests:
  - Compile-time constant range validation (all boosts in [0.5, 2.0])
  - High-success activations get boost > 1.0
  - Low-success activations get penalty < 1.0
  - TANH is baseline neutral (1.0)
  - Boost ordering reflects success rates
  - Lookup function returns correct values for all activations
  - Unknown activations get neutral boost (1.0)
  - ReLU gets neutral boost (1.0)
  - Boost correctly applied to expected score gain (increase/decrease)
  - Zero gain preserved, negative sign preserved
  - GELU ranked above HARD_TANH with equal raw gain
  - IDENTITY deprioritised relative to GELU
- `quality.sh` passes cleanly
