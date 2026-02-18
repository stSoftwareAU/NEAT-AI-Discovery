## Summary

Study of the GRQ-sampler candidate cache and implementation of a candidate outcome cache to improve the discovery success rate (Issue #465).

### Analysis Findings

Studied the [GRQ-sampler](https://github.com/stSoftwareAU/GRQ-sampler) candidate cache and identified key patterns:
- **Input neurons as sources** have a 36.2% success rate vs 2.8–3.3% for hidden neurons
- **Existing hidden neurons as targets** have a 31.4% success rate vs 5.3–5.4% for others
- The GRQ-sampler tracks per-candidate outcomes to suppress repeated failures

### Changes Made

1. **New module: `src/analysis/candidate_cache.rs`** — Candidate outcome cache that:
   - Tracks per-candidate success/failure keyed by `(source_uuid, target_uuid, operation_type)`
   - Suppresses recently-failed candidates within a configurable staleness window (default: 100 epochs)
   - Re-enables candidates after the staleness window expires (creature may have changed)
   - Tracks per-source-type success rates using Bayesian scoring (same approach as `DiscoveryHistory`)
   - Computes source-type boost factors based on historical success rates
   - Supports JSON serialisation for persistence across runs
   - Supports pruning of stale entries

2. **New constants in `src/analysis/constants.rs`**:
   - `INPUT_SOURCE_BOOST` (1.5) — static boost for input-neuron source candidates
   - `MIN_BOOST_SAMPLES` (10) — minimum samples before applying source-type boost

3. **Sub-issues created** for follow-up work:
   - #466 — Candidate outcome cache integration into `analyze_all()`
   - #467 — Source-type prioritisation during candidate evaluation
   - #468 — Target-type prioritisation for existing hidden neurons
   - #469 — Per-module success rate tracking
   - #470 — Discovery history decay (recency weighting)
   - #471 — Confidence calibration against actual ablation outcomes

## Evidence

This is a backend/library change with no UI component. Evidence is provided by the test results:
- 14 tests in `issue_465_candidate_outcome_cache.rs` covering recording, suppression, staleness, serialisation, pruning
- 7 tests + compile-time assertions in `issue_465_source_type_scoring.rs` covering Bayesian scoring, boost application, constant validation
- All 457 unit tests + 97 integration test files pass
- `quality.sh` passes cleanly (fmt, clippy, check, tests, release build)

## Test Plan

- `tests/issue_465_candidate_outcome_cache.rs` — 14 tests:
  - `empty_cache_returns_no_outcome`
  - `record_success_and_retrieve`
  - `record_failure_and_retrieve`
  - `latest_outcome_overwrites_previous`
  - `recently_failed_candidate_is_suppressed`
  - `successful_candidate_is_not_suppressed`
  - `failed_candidate_becomes_eligible_after_staleness_window`
  - `unknown_candidate_is_not_suppressed`
  - `different_operations_tracked_independently`
  - `source_type_stats_track_success_rates`
  - `source_type_boost_factor_reflects_success_rate`
  - `serialisation_round_trip`
  - `prune_removes_stale_entries`
  - `custom_staleness_window`

- `tests/issue_465_source_type_scoring.rs` — 7 tests + compile-time assertions:
  - Compile-time: `INPUT_SOURCE_BOOST` range (1.0, 3.0], `MIN_BOOST_SAMPLES` range [5, 50], `DEFAULT_STALENESS_WINDOW` range [10, 1000]
  - `source_type_stats_default_is_empty`
  - `source_type_stats_bayesian_score_with_few_samples`
  - `source_type_stats_converges_with_many_samples`
  - `boost_not_applied_with_insufficient_samples`
  - `boost_applied_with_sufficient_samples`
  - `input_source_boost_applied_to_expected_score_gain`
