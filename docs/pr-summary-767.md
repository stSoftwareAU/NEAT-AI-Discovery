## Summary

Consolidate 6+ duplicated Pearson correlation implementations into the shared
`stats` module (`src/analysis/detection/stats.rs`). Closes #767.

Two new public variants were added to `stats.rs`:

- `pearson_correlation_samples(&[HelpfulSample], &[HelpfulSample], n_samples) -> f64` —
  correlates activation fields with f64 precision, used by `redundant_path`,
  `epistatic/pre_screening`, and `epistatic/scoring`.
- `pearson_correlation_hashmaps(&HashMap<u32, f32>, &HashMap<u32, f32>, min_samples) -> f32` —
  pairs values by shared keys (obs_index), used by `correlated_error`, `multi_hop`,
  and `weight_coherence`.

All private correlation functions now delegate to these shared implementations.
No behavioural changes — existing tests pass unchanged.

## Evidence

This is a backend refactoring with no UI changes. Evidence is the passing test
suite and quality gate:

- All 12 new integration tests pass (`tests/issue_767_stats_pearson_correlation.rs`)
- `quality.sh` passes cleanly (fmt, clippy, check, tests, doc, release build)
- No private `pearson_correlation` / `calculate_correlation` / `compute_*_correlation`
  implementations remain outside `stats.rs`

## Test Plan

- Added `tests/issue_767_stats_pearson_correlation.rs` with 12 tests covering:
  - `pearson_correlation` (existing): perfect positive, perfect negative, zero variance, too few samples
  - `pearson_correlation_samples`: perfect positive, perfect negative, too few samples, n_samples limit
  - `pearson_correlation_hashmaps`: perfect positive, shared keys only, below min_samples, no shared keys
