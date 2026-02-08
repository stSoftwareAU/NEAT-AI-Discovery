## Summary

Implements four early termination improvements for low-value candidates (Issue #429), extending the existing SPRT-based early termination to candidate generation:

1. **Hierarchical candidate pre-filtering** (`CandidatePreFilter`): Quick statistical checks (sample count, source variance, error correlation) ordered cheapest-to-most-expensive, allowing poor candidates to be skipped before expensive GPU analysis.

2. **Budget-aware prioritisation** (`BudgetTracker`): Tracks candidate generation budget and progressively skips low-priority candidates as the budget fills (>80% utilisation applies a rising gain threshold).

3. **Incremental confidence checking** (`IncrementalConfidenceChecker`): Monitors confidence of generated candidates and signals when enough high-confidence candidates exist to justify stopping generation early.

4. **Cross-module deduplication** (`CrossModuleDeduplicator`): Shared deduplicator across all 25 discovery modules prevents near-duplicate candidates (same comment key + similar expected gain) from consuming budget. Integrated into the discovery dispatch pattern via `run_discovery_module_dedup`.

All four components are added to `early_termination.rs`, keeping the module as the single source of truth for early stopping logic.

## Evidence

This is a backend/performance change with no UI. Benchmark results from `benches/early_termination.rs`:

| Benchmark | Samples | Time |
|-----------|---------|------|
| pre_filter/should_analyse | 100 | 177 ns |
| pre_filter/should_analyse | 1,000 | 2.6 µs |
| pre_filter/should_analyse | 10,000 | 27.8 µs |
| pre_filter/variance_check | 100 | 66 ns |
| pre_filter/variance_check | 1,000 | 1.0 µs |
| pre_filter/variance_check | 10,000 | 10.8 µs |
| budget_tracker/consume_and_check_1000 | — | 2.6 µs |
| incremental_confidence/100_candidates | — | 5.0 µs |
| cross_module_dedup/register_and_check | 100 | 15.8 µs |
| cross_module_dedup/register_and_check | 1,000 | 137 µs |
| cross_module_dedup/register_and_check | 10,000 | 1.1 ms |
| sprt_evaluator/decision_time | 10% positive | 29 ns |
| sprt_evaluator/decision_time | 50% positive | 29 ns |
| sprt_evaluator/decision_time | 90% positive | 29 ns |

Key observations:
- Pre-filter overhead is negligible (177ns for 100 samples) compared to GPU analysis time (milliseconds)
- Cross-module deduplication scales linearly and adds ~1ms for 10K candidates — well below the savings from avoiding redundant GPU evaluations
- SPRT decision is sub-30ns regardless of positive rate, confirming the statistical test is not a bottleneck
- Budget tracker and confidence checker are both microsecond-scale

The actual performance improvement depends on the creature size and cannot be measured in isolation from the full analysis pipeline, which requires GPU hardware. The pre-filter and deduplication are designed to reduce the number of candidates entering the expensive GPU evaluation path.

## Test Plan

- Added `tests/issue_429_early_termination_improvements.rs` — 23 new integration tests:
  - Pre-filter: variance check, sample count, error correlation, combined check, statistics tracking (7 tests)
  - Budget tracker: basic ops, overflow protection, zero budget, low-priority skip logic (4 tests)
  - Incremental confidence: basic, threshold met, minimum candidates, best confidence (4 tests)
  - Cross-module deduplication: unique candidates, exact duplicate, similar gain, different target, different gain, statistics, register_if_unique (8 tests)
- All 12 existing `tests/early_termination.rs` tests continue to pass unchanged
- All 11 existing unit tests in `src/analysis/early_termination.rs` continue to pass unchanged
- Created `benches/early_termination.rs` with Criterion benchmarks for all five components
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

Closes #429
