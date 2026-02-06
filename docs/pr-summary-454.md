## Summary

Audit all tests to ensure they are "what" tests (testing functionality/outcomes),
not "how" tests (testing implementation details) or benchmarks disguised as tests
(measuring performance with timing assertions).

### Audit Findings

Reviewed all ~100 integration test files and the unit test module. The vast
majority of tests are well-written "what" tests that verify correctness. Found
and fixed **5 tests** across 3 files that were benchmarks disguised as tests:

| File | Old Test | Problem | Fix |
|------|----------|---------|-----|
| `tests/observability.rs` | `phase_timer_overhead_disabled` | Loops 10K times, asserts on elapsed time | Replaced with functional test for create/drop correctness |
| `tests/observability.rs` | `gpu_metrics_overhead` | Loops 10K times, asserts on elapsed time | Replaced with functional test verifying accumulation correctness |
| `tests/observability.rs` | `profile_data_overhead` | Loops 10K times, asserts on elapsed time | Replaced with functional test verifying last recorded value |
| `tests/gpu_timing.rs` | `timing_collector_overhead_minimal` | Compares disabled vs enabled timing | Replaced with functional test: disabled returns None, enabled returns Some with valid fields |
| `tests/issue_186_rwlock_cache.rs` | `cache_uses_read_lock_for_existing_entries` | Asserts second duration < first duration | Replaced with functional test: loader invoked only once per UUID |

### Documentation Update

Updated **AGENTS.md** section 4 (Testing Philosophy) to clarify:
- Explicit definitions of "what" tests, "how" tests, and benchmarks disguised as tests
- Code examples showing all three categories (good, bad, bad)
- Clear rule: timing assertions (`Instant`/`elapsed`) belong in `benches/`, not `tests/`
- Explanation of why this matters (tests run in parallel, timing is unreliable)

### No "How" Tests Found

The audit found **zero** "how" tests in the codebase. All tests verify observable
outcomes (candidates produced, data structures returned, errors handled) rather
than implementation details.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- Converted 5 timing-based tests to functional tests (see table above)
- All existing tests continue to pass — `quality.sh` passes cleanly
- No tests were removed — all were converted to meaningful functional equivalents
