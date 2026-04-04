## Summary

Add compound bias+weight degradation detection to the coordinated structural
discovery pipeline. This enables the discovery engine to identify cases where
multiple parameters (neuron bias and synapse weight) are both degraded and
need simultaneous correction — neither fix alone fully restores performance.

The new detection module (`compound_degradation.rs`) analyses hidden neurons
for consistent error (suggesting bias drift) and synapses for
activation-error correlation (suggesting weight degradation), then combines
related corrections on the same forward path into atomic coordinated
candidates with `setBias` + `setWeight` operations.

Closes #929.

## Changes

- **New file: `src/analysis/detection/compound_degradation.rs`** — Detection
  module for compound bias+weight degradations. Uses error gradient analysis
  to compute optimal bias corrections and least-squares regression for
  weight corrections, combining related pairs into coordinated candidates.
- **`src/analysis/detection/mod.rs`** — Register the new module.
- **`src/analysis/module_dispatch_specs/structural_specs.rs`** — Wire the
  new detector into the parallel dispatch pipeline.
- **`src/analysis/module_dispatch_specs/mod.rs`** — Update expected module
  count in tests (46 → 47).
- **New file: `tests/synapse/issue_929_coordinated_structural_compound_degradation.rs`**
  — End-to-end scenario test verifying coordinated discovery for the
  compound degradation scenario from the issue.
- **`tests/synapse/main.rs`** — Register the new test module.

## Evidence

All 4 new tests pass on GPU:
- `issue_929_discovers_compound_bias_weight_degradation` — Finds compound
  candidate with 2 operations (setBias + setWeight), scoreGain=0.118
- `issue_929_set_bias_targets_correct_neuron` — setBias(hidden-A) bias=0.226
  (target ~0.3)
- `issue_929_set_weight_targets_correct_synapse` — setWeight(hidden-B ->
  hidden-C) weight=0.969 (target ~0.8)
- `issue_929_all_candidates_have_positive_improvement` — All candidates have
  positive expected score gain

Full quality gate passes: 158 tests, 0 failures, clippy clean, docs build.

## Test Plan

- Added `tests/synapse/issue_929_coordinated_structural_compound_degradation.rs`
  with 4 test cases:
  1. Discovery produces a coordinated-structural candidate with both setBias
     and setWeight operations
  2. The setBias operation targets the correct neuron with reasonable bias
  3. The setWeight operation targets the correct synapse with reasonable weight
  4. All returned candidates have positive expected improvement
- Existing 154 synapse tests continue to pass (no regressions)
