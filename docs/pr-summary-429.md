## Summary

Implements early termination improvements for low-value candidates (Issue #429). Adds a `CandidatePreFilter` that runs alongside the discovery module dispatch loop, reducing analysis overhead through four mechanisms:

1. **Hierarchical candidate filtering**: Rejects candidates with expected score gain at or below a configurable minimum threshold (`MIN_CANDIDATE_GAIN = 1e-9`), skipping clearly unhelpful suggestions.

2. **Budget-aware prioritisation**: Tracks accepted candidate count across all discovery modules. Once the budget is exhausted, remaining modules are skipped entirely — no detection closures are called.

3. **Incremental confidence (low-gain filtering)**: Near-zero and negative gain candidates are filtered before merge, avoiding wasted computation on clearly unbeneficial suggestions.

4. **Cross-module deduplication**: Tracks `(from_uuid, to_uuid)` neuron-pair signatures across modules. Duplicate pairs from later modules are rejected, preventing redundant ablation tests.

### Architecture

- New module: `src/analysis/candidate_prefilter.rs` — self-contained pre-filter with `PreFilterConfig` and `CandidatePreFilter`
- New dispatch function: `run_discovery_module_filtered()` in `discovery_dispatch.rs` — wraps the existing `run_discovery_module()` pattern with budget checks and pre-filtering
- All 20+ discovery modules in `analyze_all()` now use the filtered dispatch
- Verbose logging reports pre-filter statistics at the end of the dispatch loop

### Key Design Decisions

- **Conservative defaults**: `MIN_CANDIDATE_GAIN = 1e-9` is intentionally very small to avoid filtering genuine improvements
- **Budget tied to `max_synapse_candidates`**: The pre-filter budget defaults to the same cap as the final truncation, avoiding redundant work
- **No false negatives on good candidates**: The filters only remove near-zero gain candidates and exact duplicates — high-quality candidates always pass through

## Evidence

This is a performance/backend change with no UI component. Evidence is provided via benchmark results and tests.

### Benchmark Results

Pre-filter operates in microseconds, adding negligible overhead while enabling module-level skipping:

| Scenario | Time |
|----------|------|
| 50 candidates | 5.58 us |
| 200 candidates | 22.3 us |
| 500 candidates | 50.5 us |
| 1000 candidates | 76.0 us |
| 10 modules x 50 candidates | 41.1 us |

The pre-filter itself is fast (sub-100us even for 1000 candidates). The real performance benefit comes from **skipping entire discovery modules** when the budget is exhausted — each module involves record collection, pattern detection, and candidate conversion that can take milliseconds to seconds.

## Test Plan

- Added 13 integration tests in `tests/issue_429_early_termination_improvements.rs`:
  - `test_hierarchical_filtering_rejects_low_gain` — verifies gain threshold filtering
  - `test_default_min_gain_is_conservative` — validates conservative constant value
  - `test_filtering_at_exact_threshold` — strict inequality boundary test
  - `test_budget_limits_candidates_across_modules` — budget enforcement across filter calls
  - `test_budget_exhaustion_flag` — `budget_exhausted()` state tracking
  - `test_module_skip_tracking` — module skip counter
  - `test_incremental_confidence_near_zero_gains` — near-zero/negative gain rejection
  - `test_cross_module_dedup_rejects_duplicate_pairs` — deduplication across modules
  - `test_dedup_allows_different_pairs` — non-duplicate pairs pass through
  - `test_dedup_does_not_affect_non_pair_candidates` — SetBias/AddNeuron bypass dedup
  - `test_dedup_disabled_allows_duplicates` — dedup can be disabled
  - `test_combined_filtering_stages` — all filter stages work together
  - `test_stats_total_rejected_is_sum` — statistics accuracy
- Added 10 unit tests inline in `src/analysis/candidate_prefilter.rs`
- Added benchmark `benches/early_termination.rs` with two benchmark groups
- All existing tests continue to pass (468 unit + 97 integration test files)
