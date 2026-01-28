## Summary

Implements redundant path pruning with renormalisation (Issue #164).

When two existing subnetworks (synapses) feeding the same output neuron compute effectively
the same thing (highly correlated activation patterns), one path can be pruned and the
survivor's weight scaled to compensate. This reduces network complexity without degrading
fitness.

### Detection signals
- **Highly correlated activations** – Pearson correlation ≥ 0.85
- **Anti-correlated error gradients** – Both paths push error in the same direction
- **Shared downstream synapses** – Both sources feed the same target

### Discovery type
`COORDINATED_PRUNE_AND_REWEIGHT` – emitted as a coordinated structural candidate with:
- `removeSynapse` – removes the weaker (redundant) path
- `setWeight` – renormalises the survivor's weight to `keep_weight + prune_weight`

No new operation types are needed – NEAT-AI handles these through the existing coordinated-structural path.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

The implementation uses existing `removeSynapse` and `setWeight` operations already supported by NEAT-AI's `ApplyCoordinatedStructuralCandidate.ts` (verified Issue #337).

## Test Plan

### Unit tests (11 tests in `src/analysis/redundant_path.rs`)
- `test_identical_activations_detected_as_redundant` – Verifies identical activation patterns are detected
- `test_uncorrelated_activations_not_detected` – Verifies independent patterns are NOT detected
- `test_anti_correlated_activations_detected` – Verifies anti-correlated patterns are detected (absolute correlation)
- `test_insufficient_samples_returns_empty` – Too few samples returns empty
- `test_single_path_returns_empty` – Single path returns empty
- `test_very_small_weights_skipped` – Negligible weights are skipped
- `test_redundant_paths_to_coordinated_candidates` – Candidate structure is correct (removeSynapse + setWeight)
- `test_compute_activation_correlation_identical` – Correlation of identical samples is ~1.0
- `test_compute_activation_correlation_anti_correlated` – Absolute correlation of anti-correlated samples is ~1.0
- `test_stronger_weight_is_kept` – The stronger (by absolute weight) synapse is kept
- `test_target_impact_affects_improvement` – Target impact discounting works correctly

### Integration tests (3 tests in `tests/issue_164_redundant_path_pruning.rs`)
- `redundant_identical_paths_detected` – End-to-end test with two identical input paths → detects and proposes pruning
- `independent_paths_not_detected_as_redundant` – End-to-end test with uncorrelated inputs → no false positives
- `redundant_path_candidate_has_expected_structure` – Validates candidate JSON structure (operations, score gain, comment)

### Files changed
- `src/analysis/redundant_path.rs` (new) – Detection module with unit tests
- `src/analysis/mod.rs` – Register new module
- `src/analysis/implementation.rs` – Integrate detection into analysis pipeline
- `tests/issue_164_redundant_path_pruning.rs` (new) – Integration tests
- `docs/DISCOVERY_TYPES.md` – Document new discovery type
- `README.md` – Document redundant path pruning feature
- `docs/pr-summary-164.md` (new) – This PR summary
