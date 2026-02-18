## Summary

Audit all discovery strategies to ensure only candidates with a positive predicted
score improvement are returned. Closes #557.

### Problem

Several discovery modules could produce candidates with zero or near-zero
`expected_creature_score_gain` at boundary conditions:

- **dormant_synapse**: `0.001 * (1.0 - ratio)` equals zero when `mean_abs_contribution`
  exactly equals `DORMANT_CONTRIBUTION_THRESHOLD`
- **error_plateau**: `mean_error * plateau_tightness * 0.3` equals zero when
  `plateau_tightness` is clamped to zero via `.max(0.0)`
- **correlated_error**: product of factors can be zero if `mean_correlation` or
  `mean_abs_error` is zero
- **activation_mismatch**: boundary conditions at detection thresholds
- **Impact discounting**: any candidate targeting a hidden neuron disconnected from
  outputs gets multiplied by 0.1, which can reduce already-small gains to effectively
  zero

Returning these zero-gain candidates wastes evaluation budget since NEAT-AI must
clone the creature, apply the mutation, and re-score against the full training set
for each candidate — all for zero predicted benefit.

### Solution

Added positive-gain filtering (`expected_creature_score_gain > 0.0`) at three
pipeline choke-points, providing defence-in-depth:

1. **`merge_coordinated_structural_replacements`** (mod.rs) — filters all coordinated
   structural candidates before merging into the synapse result. This catches every
   discovery module since they all merge through this single function.

2. **Synapse post-processing** (post_processing.rs) — filters helpful, harmful, and
   coordinated candidates after impact discounting, before sorting and truncation.

3. **Neuron pipeline** (neuron.rs) — filters neuron candidates after impact
   discounting, before sorting and truncation.

This is a central safety-net approach (DRY) rather than patching each of the 28+
individual discovery modules, ensuring that future modules also benefit automatically.

## Evidence

This is a backend/pipeline change with no visual output. Evidence is provided by
the test results from `quality.sh` passing cleanly, including all 502+ unit tests
and 97+ integration tests.

## Test Plan

- Added `tests/issue_557_audit_discovery_strategies.rs` — integration tests that
  run the full `analyze_all` pipeline and assert every returned candidate (synapse,
  neuron, coordinated) has strictly positive `expected_creature_score_gain`, plus
  metadata consistency checks
- Added unit tests in `src/analysis/discovery_dispatch_parallel_tests.rs`:
  - `parallel_dispatch_filters_zero_gain_candidates`
  - `parallel_dispatch_filters_negative_gain_candidates`
  - `parallel_dispatch_filters_all_non_positive_returns_empty`
- Added unit tests in `src/analysis/discovery_dispatch_tests.rs`:
  - `run_discovery_module_filters_zero_gain_candidates`
  - `run_discovery_module_filters_negative_gain_candidates`
- Added unit test in `src/analysis/mod_tests.rs`:
  - `merge_coordinated_structural_filters_zero_gain_candidates`
