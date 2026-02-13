## Summary

Replace all production `panic!` paths from mutex `.expect("Mutex poisoned")` and `.unwrap()` calls with graceful error handling. Closes #525.

An FFI library must never panic — a panic unwinds across the FFI boundary and causes the calling process (NEAT-AI via Deno) to abort unexpectedly. This PR replaces **31 mutex `.expect()` calls** and **4 mutex `.unwrap()` calls** across 5 production source files with safe alternatives that return errors via the JSON `success: false` mechanism instead of crashing.

### Approach

1. **`lock_or_bail()`** — new helper in `analysis/utils/mod.rs` that converts `PoisonError` into `anyhow::Error`, propagating gracefully to the FFI boundary.

2. **`into_inner_or_bail()`** — companion helper for consuming a mutex.

3. **Poison-recovery pattern** — for code paths that cannot return `Result` (e.g. `par_iter().for_each()` closures, diagnostic `.len()` methods), uses `Err(poisoned) => poisoned.into_inner()` to recover the data from a poisoned mutex rather than panicking.

4. **`debug_assert!`** — invariants that "should never" fail are checked in debug builds but gracefully handled in release.

### Files changed

| File | Changes |
|------|---------|
| `src/analysis/utils/mod.rs` | Added `lock_or_bail()` and `into_inner_or_bail()` helpers |
| `src/analysis/neuron.rs` | Replaced 13 `.expect("Mutex poisoned")` calls with `lock_or_bail()` |
| `src/analysis/synapse/mod.rs` | Replaced 12 `.expect("Mutex poisoned")` calls with `lock_or_bail()` |
| `src/focus/ranking.rs` | Replaced 3 `.expect("lazy record cache poisoned")` calls; `len()` uses poison-recovery |
| `src/analysis/shared.rs` | Replaced 2 `.expect("Mutex poisoned")` in `TimingCollector` with `let Ok(...)` pattern |
| `src/focus/impact.rs` | Replaced 4 `.unwrap()` calls with poison-recovery pattern; replaced `.expect()` with `debug_assert!` + `unwrap_or_default()` |
| `tests/issue_525_graceful_mutex_handling.rs` | 4 new tests verifying graceful handling |

## Evidence

This is a backend/library change with no UI. Evidence is provided by the test suite:

- `test_poisoned_mutex_returns_error_instead_of_panic` — verifies `lock_or_bail()` returns `Err` (not panic) for a poisoned mutex
- `test_healthy_mutex_returns_guard` — verifies `lock_or_bail()` works normally
- `test_poisoned_mutex_into_inner_returns_error` — verifies `into_inner_or_bail()` returns `Err` for a poisoned mutex
- `test_healthy_mutex_into_inner_returns_value` — verifies `into_inner_or_bail()` works normally
- All 478 unit tests + 97 integration test files pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- Added `tests/issue_525_graceful_mutex_handling.rs` with 4 tests covering:
  - Poisoned mutex lock returns error instead of panicking
  - Healthy mutex lock returns guard normally
  - Poisoned mutex `into_inner` returns error instead of panicking
  - Healthy mutex `into_inner` returns value normally
- All existing tests continue to pass unchanged
