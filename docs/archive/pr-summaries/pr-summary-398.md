## Summary

Add observation effective range detection module (`src/analysis/observation_range.rs`) that analyses recorded samples per input observation to identify sentinel/null value clusters and compute the effective range where the observation actually influences the output.

This is a sub-issue of #395 (bounded range discovery). While `bounded_range.rs` detects boundary clustering and recommends gating neurons, this module focuses on **characterising** the effective range using error correlation analysis, providing metadata for downstream use:

- `effective_min`, `effective_max` — the useful range excluding sentinel clusters
- `sentinel_values` — detected sentinel/null values (e.g., -1.0, 0.0)
- `utilisation_ratio` — fraction of the full observed range that is effective

### Detection approach

1. For each input observation, collect activation values and corresponding errors
2. Check candidate sentinel values (-1, 0, +1) for density clusters (≥15% of samples)
3. Verify a gap exists between the sentinel cluster and the useful range
4. Compare error variance in sentinel cluster vs non-sentinel range to confirm sentinel has lower correlation
5. Compute effective range and utilisation ratio from non-sentinel values

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

Added 9 integration tests in `tests/issue_398_observation_range_detection.rs`:

1. `test_detects_effective_range_excluding_sentinel_at_minus_one` — sentinel at -1.0 detected, effective range excludes it
2. `test_detects_effective_range_with_sentinel_at_zero` — sentinel at 0.0 detected
3. `test_uniform_distribution_full_range_effective` — no sentinels in uniform data
4. `test_insufficient_samples_no_detection` — below minimum sample threshold
5. `test_utilisation_ratio_computed_correctly` — ratio reflects sentinel impact
6. `test_output_neurons_excluded` — only input neurons analysed
7. `test_error_correlation_distinguishes_sentinel` — error variance differentiates sentinel
8. `test_multiple_observations_independent_ranges` — per-neuron independent analysis
9. `test_all_identical_values_no_detection` — degenerate case handled

All tests pass via `./quality.sh`.
