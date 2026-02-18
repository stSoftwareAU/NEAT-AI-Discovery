## Summary

Enhanced the error stagnation plateau detection module to generate coordinated `changeSquash` + `setBias` structural candidates for escaping local minimums. Previously the module only recommended a squash function change; now it also computes a bias adjustment based on the mean signed error, recentring the output after the activation function change. The `setBias` operation is only included when the adjustment exceeds a minimum threshold to avoid trivial changes. Closes #547.

## Changes

- **`src/analysis/detection/error_plateau.rs`**: Added `current_bias` and `recommended_bias` fields to `ErrorPlateauCandidate`. Added `compute_bias_adjustment()` to calculate bias correction from mean signed error. Updated `error_plateaus_to_coordinated_candidates()` to emit combined `changeSquash` + `setBias` operations when bias adjustment is meaningful.
- **`tests/issue_547_error_plateau_structural.rs`**: New integration test file with 6 tests covering coordinated squash+bias operations, bias adjustment computation, negligible bias suppression, multiple plateau neurons, healthy network rejection, and positive improvement estimates.
- **`tests/issue_545_error_plateau.rs`**: Updated issue reference in comment assertion from #545 to #547.

## Evidence

This is a backend detection module with no UI. All verification is via integration tests:

```
running 7 tests (issue_545_error_plateau) ... ok
running 6 tests (issue_547_error_plateau_structural) ... ok
quality.sh: All quality checks passed!
```

## Test Plan

- `test_coordinated_candidate_has_squash_and_bias_operations` — verifies both `changeSquash` and `setBias` appear in coordinated candidates
- `test_recommended_bias_adjusts_for_plateau` — verifies bias is computed from mean signed error
- `test_no_set_bias_when_adjustment_negligible` — verifies `setBias` is omitted when errors are symmetric
- `test_multiple_plateau_neurons_get_coordinated_pairs` — verifies each plateau neuron gets its own candidate
- `test_healthy_network_no_candidates` — verifies converged networks produce no false positives
- `test_estimated_improvement_positive` — verifies positive expected score gain
