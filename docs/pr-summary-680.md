## Summary

Expand property-based testing with proptest by adding 5 new test files covering
distinct areas of the codebase that were not exercised by the existing
`issue_573_proptest_mathematical_functions.rs`. Closes #680.

## Changes

### New Test Files

| # | File | Area | Test Count |
|---|------|------|------------|
| 1 | `issue_680_proptest_early_termination.rs` | SPRT early termination (`SequentialEvaluator`, `check_batch_early_termination`) | 12 |
| 2 | `issue_680_proptest_neuron_interning.rs` | UUID string interning (`NeuronIndex`) | 9 |
| 3 | `issue_680_proptest_activation_names.rs` | Activation name handling (`normalise_squash_name`, `is_known_squash_name`, `is_aggregate_squash`, `target_simulation_fn` consistency) | 11 |
| 4 | `issue_680_proptest_sample_statistics.rs` | Sample statistics (`HelpfulStats`, `HarmfulStats`, `NeuronStats`) | 18 |
| 5 | `issue_680_proptest_clustering_thresholds.rs` | Candidate clustering and variance thresholds (`cluster_candidates`, `compute_source_variance_discount`, `compute_dynamic_constant_source_threshold`) | 14 |

**Total: 64 new property-based tests across 5 files.**

### Properties Tested

- **Ratio bounds**: `improvement_ratio()`, `harmful_ratio()` always in [0, 1]
- **Mutual exclusion**: `is_strongly_beneficial()` and `is_strongly_harmful()` never both true
- **Minimum-sample guards**: heuristic decisions require >= 30 samples
- **Round-trip identity**: `NeuronIndex::get_uuid(intern(s)) == Some(s)`
- **Idempotency**: `normalise_squash_name` applied twice gives same result
- **Consistency**: `target_simulation_fn(name)(x) == apply_scalar_squash(name, x)` for all scalar squash names
- **Merge additivity**: `HelpfulStats::merge()` sums counts correctly
- **LLR monotonicity**: higher positive ratio yields higher log-likelihood ratio
- **Cluster structural guarantees**: min size >= 2, member_count consistency, sorted output
- **Variance discount bounds**: always in [0, 1], monotonically increasing with variance
- **Dynamic threshold monotonicity**: increases with source standard deviation
- **NaN/Inf safety**: NeuronStats filters non-finite values

## Evidence

This is a test-only change with no UI or performance impact. All 5 test files
compile and pass. The full quality gate (`./quality.sh`) passes cleanly.

## Test Plan

- All 64 new proptest cases pass with `--test-threads=2`
- Existing `issue_573_proptest_mathematical_functions.rs` tests are unmodified
- `./quality.sh` passes: fmt, clippy, check, test, doc, release build
