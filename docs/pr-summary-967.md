## Summary

Implement adaptive candidate budget allocation by module success rate. Discovery modules now receive candidate generation budgets proportional to their Bayesian success rate from `ModuleOutcomeTracker`, with high-success modules getting up to 2x base allocation and low-success modules getting a minimum of 0.5x (never starving exploration). A global candidate cap prevents total candidate count explosion. Closes #967.

## Changes

1. **`CandidateBudgetConfig` struct** (`src/analysis/module_weights.rs`): Configurable budget allocation parameters — base candidates per module, global max cap, min/max allocation factors.
2. **`allocate_candidate_budgets` function** (`src/analysis/module_weights.rs`): Computes per-module candidate budgets using Bayesian success rates, with proportional scaling under a global cap and verbose logging for observability.
3. **`max_candidates` field on `DiscoveryModuleSpec`** (`src/analysis/discovery_dispatch.rs`): Each module spec now carries its allocated candidate budget.
4. **Per-module truncation** (`src/analysis/discovery_dispatch.rs`): During the sequential merge phase, each module's candidates are truncated to its allocated budget before merging.
5. **Budget allocation wiring** (`src/analysis/module_dispatch_specs/mod.rs`): `dispatch_and_merge_discovery_modules` now calls `allocate_candidate_budgets` and assigns budgets to each module spec before dispatch.
6. **Macro updates** (`src/analysis/module_dispatch_specs/macros.rs`): All four `discovery_spec!` macro variants initialise `max_candidates: 0` (overridden at dispatch time).

## Evidence

- All 12 new integration tests pass, covering: success-rate scaling, equal allocation, 0.5x–2.0x bounds, global cap, sparse/unknown module fallback, edge cases, and the `max_candidates` field on `DiscoveryModuleSpec`.
- `quality.sh` passes cleanly (fmt, clippy, tests, docs, release build).

## Test Plan

- Added `tests/analysis/issue_967_adaptive_candidate_budget.rs` with 12 tests:
  - `test_high_success_module_gets_more_candidates` — verifies proportional allocation
  - `test_equal_success_gives_equal_budgets` — verifies equal rates yield equal budgets
  - `test_max_budget_is_two_times_base` — verifies upper bound clamping
  - `test_min_budget_is_half_base` — verifies lower bound clamping
  - `test_global_cap_limits_total_candidates` — verifies global cap enforcement
  - `test_global_cap_preserves_proportions` — verifies proportions under cap
  - `test_sparse_history_gets_base_allocation` — verifies fallback for sparse data
  - `test_unknown_module_gets_base_allocation` — verifies fallback for unknown modules
  - `test_empty_module_list_returns_empty` — edge case
  - `test_every_module_gets_at_least_one_candidate` — minimum floor
  - `test_default_config_has_sensible_values` — config invariants
  - `test_module_spec_includes_max_candidates` — field accessibility
