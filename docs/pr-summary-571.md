## Summary

Activation co-adaptation detection — identify redundant neuron pairs whose activations are highly correlated (or anti-correlated), wasting network capacity. Closes #571.

New detection module `src/analysis/detection/co_adaptation.rs` computes pairwise Pearson correlation between hidden neuron activations from recorded sample data. Pairs with |correlation| > 0.9 are flagged as co-adapted. For each pair, two candidate strategies are generated:
1. **RemoveNeuron** — remove the lower-impact neuron to eliminate redundancy
2. **SetWeight** — perturb one neuron's incoming weights to break the co-adaptation

Unlike the existing symmetry-breaking detection (Issue #569) which compares weight configurations, co-adaptation detection analyses the *recorded activations* — two neurons may have completely different weights and activation functions but still produce correlated outputs.

## Evidence

This is a backend detection module with no UI component. Evidence is provided by the integration test suite.

All 7 integration tests pass, covering:
- Correlated pair detection
- Anti-correlated pair detection
- Independent neuron exclusion
- Candidate generation (RemoveNeuron + SetWeight)
- Insufficient sample filtering
- Single hidden neuron edge case
- Input/output neuron exclusion

`quality.sh` passes cleanly (fmt, clippy, check, all tests, release build).

## Test Plan

- `tests/issue_571_co_adaptation_detection.rs` — 7 integration tests:
  - `test_correlated_pair_detected` — verifies highly correlated hidden neurons are flagged
  - `test_anti_correlated_pair_detected` — verifies anti-correlated neurons are also detected
  - `test_independent_neurons_ignored` — verifies uncorrelated neurons are not flagged
  - `test_candidates_generated_for_co_adapted_pair` — verifies coordinated candidates are produced with correct operations and comments
  - `test_insufficient_samples_skipped` — verifies minimum sample count is enforced
  - `test_single_hidden_neuron_no_candidates` — edge case: single neuron cannot form a pair
  - `test_only_hidden_neurons_paired` — verifies input/output neurons are excluded
