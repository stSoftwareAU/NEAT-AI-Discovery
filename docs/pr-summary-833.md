## Summary

Migrate all `std::sync::Mutex` usage to `parking_lot::Mutex` so that the existing
`parking_lot::deadlock::check_deadlock()` background thread (in `debug.rs`) can
detect deadlocks on every mutex in the codebase. Closes #833.

Previously, four source files used `std::sync::Mutex`, making their locks invisible
to the deadlock detector. This change replaces those imports and removes the
poisoned-mutex recovery patterns (`match lock() { Ok(g) => g, Err(p) => p.into_inner() }`,
`.map_err(...)`, `PoisonError::into_inner`) that are unnecessary with `parking_lot`
(which does not poison on thread panic).

### Files changed

| File | Change |
|------|--------|
| `src/analysis/utils/mod.rs` | Switched `Mutex` import to `parking_lot`; simplified `lock_or_bail` and `into_inner_or_bail` (always succeed) |
| `src/analysis/neuron/mod.rs` | Switched `Mutex` import; removed `.map_err(...)` on lock call |
| `src/analysis/neuron/evaluation.rs` | Switched `Mutex` import |
| `src/analysis/neuron/post_processing.rs` | Switched `Mutex` import |
| `src/analysis/shared.rs` | Switched `Mutex` import; removed poison-recovery `else` branches in `TimingCollector` |
| `src/focus/impact.rs` | Switched `Mutex` import; removed all `match lock() { Ok/Err(poisoned) }` patterns and `PoisonError::into_inner` |
| `src/focus/ranking/record_providers.rs` | Switched `Mutex` import; simplified `len()` method |
| `tests/infrastructure/issue_525_graceful_mutex_handling.rs` | Updated tests to use `parking_lot::Mutex`; replaced poisoned-mutex tests with non-poisoning verification tests |

## Evidence

- `./quality.sh` passes (fmt, clippy, check, test, doc, release build)
- All 124 tests pass including the updated mutex handling tests
- Zero `use std::sync::Mutex` remaining in `src/`

## Test Plan

- Updated `test_healthy_mutex_returns_guard` — verifies `lock_or_bail` with `parking_lot::Mutex`
- Updated `test_healthy_mutex_into_inner_returns_value` — verifies `into_inner_or_bail`
- Added `test_mutex_usable_after_thread_panic` — verifies `parking_lot::Mutex` remains usable after a thread panics while holding the lock (key behavioural difference from `std::sync::Mutex`)
- Added `test_into_inner_after_thread_panic` — verifies `into_inner_or_bail` succeeds after thread panic

### Test modifications (documented per guidelines)

The four tests in `issue_525_graceful_mutex_handling.rs` were updated because the business
logic changed: `parking_lot::Mutex` does not poison, so the two "poisoned mutex returns error"
tests were replaced with "mutex usable after thread panic" tests that verify the new
non-poisoning behaviour.
