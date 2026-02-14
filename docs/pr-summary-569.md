## Summary

Add symmetry-breaking detection module that identifies pairs of hidden neurons whose
weight configurations, bias values, and activation functions have converged to
near-identical states. When detected, the module recommends perturbation of one
neuron (bias shift + incoming weight scaling) to break the symmetry and unlock
latent representational capacity. Closes #569.

## Changes

- **New file**: `src/analysis/detection/symmetry_breaking.rs` — detection module implementing:
  - `detect_symmetric_neurons()` — identifies symmetric pairs using cosine similarity of incoming weight vectors (threshold 0.95), bias tolerance (0.5), and activation function equality
  - `symmetric_neurons_to_coordinated_candidates()` — converts detections into `SetBias` + `SetWeight` coordinated candidates
- **Modified**: `src/analysis/detection/mod.rs` — added `symmetry_breaking` module declaration
- **Modified**: `src/analysis/mod.rs` — added re-export for backward compatibility
- **Modified**: `src/analysis/module_dispatch_specs.rs` — added dispatch spec for parallel execution

## Evidence

This is a backend detection module with no visual output. All changes verified through:
- 9 integration tests covering detection, edge cases, and candidate generation
- Full `quality.sh` pass (fmt, clippy, check, test, release build)

## Test Plan

- `tests/issue_569_symmetry_breaking.rs` — 9 tests:
  1. `test_detects_identical_symmetric_neurons` — verifies identical neurons are detected with cosine similarity > 0.95
  2. `test_ignores_dissimilar_neurons` — verifies neurons with different incoming weights are not flagged
  3. `test_different_squash_prevents_detection` — verifies different activation functions prevent detection
  4. `test_single_hidden_neuron_no_candidates` — edge case: single neuron cannot form a pair
  5. `test_no_hidden_neurons_no_candidates` — edge case: no hidden neurons
  6. `test_coordinated_candidates_include_perturbations` — verifies candidates include SetBias/SetWeight operations
  7. `test_insufficient_samples_returns_empty` — verifies minimum sample count is enforced
  8. `test_multiple_symmetric_pairs` — verifies multiple distinct pairs are each detected
  9. `test_large_bias_difference_prevents_detection` — verifies bias tolerance is enforced
