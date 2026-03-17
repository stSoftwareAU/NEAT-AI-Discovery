## Summary

Add optional lock contention tracing for diagnosing concurrent stalls in hot-path lock acquisitions. When `NEAT_AI_DISCOVERY_VERBOSE=1` is set, lock acquisitions exceeding a 100ms threshold emit a `tracing::warn` with the lock name and wait duration. In non-verbose mode, tracing is completely bypassed with zero overhead. Closes #837.

## Changes

- **New module `src/analysis/utils/lock_contention.rs`**: Provides `traced_lock` and `traced_lock_default` functions that wrap `parking_lot::Mutex::lock()` with optional timing when verbose mode is enabled
- **`src/analysis/shared.rs`**: Instrumented `shader_timings` Mutex in `TimingCollector::record_shader()` and `finalize()` with contention tracing
- **`src/analysis/utils/mod.rs`**: Updated `lock_or_bail` helper (used by neuron analysis `helpful_map` and other hot paths) to route through contention tracing

Note: Two of the three originally identified hot-path locks have already been eliminated:
- `error_values_for_distribution` was replaced with Rayon fold/reduce (#834)
- `shared_cache` in impact calculation was replaced with DashMap (#835)

The remaining `helpful_map` Mutex in neuron analysis and `shader_timings` in `TimingCollector` are now instrumented.

## Evidence

All 7 integration tests pass, plus all existing tests (124+ tests). `quality.sh` passes cleanly including clippy, fmt, doc build, and release build.

## Test Plan

- Added `tests/infrastructure/issue_837_lock_contention_tracing.rs` with 7 tests:
  - `traced_lock_acquires_mutex_and_returns_correct_value` — basic acquisition
  - `traced_lock_default_uses_100ms_threshold` — default threshold verification
  - `traced_lock_allows_mutation_through_guard` — mutation through guard
  - `traced_lock_with_custom_threshold` — custom threshold support
  - `traced_lock_concurrent_access_does_not_deadlock` — 4-thread concurrent stress test
  - `traced_lock_guard_drops_correctly` — guard release verification
  - `lock_or_bail_uses_contention_tracing` — integration with existing helper
- Unit tests in `src/analysis/utils/lock_contention.rs` (5 tests)
