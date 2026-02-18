## Summary

Adaptive discovery module timeout — allocate time budget based on historical yield. Closes #603.

Adds an `allocate_time_budgets()` function to `module_weights.rs` that distributes a total
time budget across discovery modules proportionally to their historical acceptance rates
(Bayesian success rate from `ModuleOutcomeTracker`). Also adds an `apply_decay()` method
to `ModuleOutcomeTracker` so historical data can be gradually reduced, preventing permanent
bias from old outcomes.

### Key behaviours:
- **Adaptive allocation**: Modules with higher historical acceptance rates receive more time
- **Starvation prevention**: Minimum weight floor of 0.5 ensures no module is starved
- **Decay factor**: Blends between equal allocation (decay=0.0) and fully history-driven (decay=1.0)
- **Sparse history fallback**: Modules with fewer than `MIN_BOOST_SAMPLES` attempts receive neutral (equal) allocation
- **Budget conservation**: Allocated budgets sum to the total available time

## Evidence

This is a backend/library change with no visual output. Evidence is provided by the 14
integration tests that verify all allocation behaviours.

## Test Plan

- Added `tests/issue_603_adaptive_module_timeout.rs` with 14 tests:
  - `test_equal_history_gives_equal_budgets` — equal success rates → equal budgets
  - `test_high_yield_module_gets_more_time` — high-yield modules get more time
  - `test_zero_success_module_still_gets_minimum_budget` — no starvation
  - `test_sparse_history_falls_back_to_proportional` — insufficient history → equal
  - `test_unknown_modules_get_proportional_allocation` — unknown modules get neutral
  - `test_empty_tracker_gives_equal_budgets` — no history → equal
  - `test_decay_factor_reduces_historical_influence` — decay reduces bias
  - `test_partial_decay_is_intermediate` — partial decay blends correctly
  - `test_single_module_gets_full_budget` — single module gets all time
  - `test_empty_module_list_returns_empty_budgets` — edge case
  - `test_budgets_sum_to_total` — budget conservation
  - `test_apply_decay_reduces_counts` — decay halves counts
  - `test_apply_decay_zero_clears_history` — full reset
  - `test_apply_decay_one_preserves_history` — no-op decay
- All existing tests pass
- `cargo clippy` and `./quality.sh` pass cleanly
