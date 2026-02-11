## Summary

Split `src/focus.rs` (2,946 lines) into focused submodules under `src/focus/` to align with the project's ~1,500 line target per file (AGENTS.md §3).

**No changes to the public API** — all existing imports via `crate::focus::*` and `neat_ai_discovery::focus::*` continue to work unchanged.

| New Module | Responsibility | Lines |
|-----------|---------------|-------|
| `mod.rs` | Public API, re-exports, module declarations | 47 |
| `layers.rs` | Network layer computation via BFS | 167 |
| `allocation.rs` | Budget allocation strategies (Equal, Proportional, OutputFirst) | 215 |
| `gradient.rs` | Gradient flow analysis (saturation, dead neurons) | 582 |
| `impact.rs` | Impact calculation (squash-aware, selection stats) | 653 |
| `ranking.rs` | Neuron ranking, record providers, removal candidates | 1,248 |
| `tests.rs` | Unit tests for internal components | 71 |
| **Total** | | **2,983** |

Every submodule is now under the 1,500-line target. The largest file (`ranking.rs` at 1,248 lines) contains the core ranking functions that are tightly coupled and benefit from being in the same file.

Also updated `AGENTS.md` source layout to document the new module structure.

## Evidence

This is a pure refactoring change (code reorganisation) with no UI or performance changes. Evidence is provided by the test suite:

- All **463 unit tests** pass
- All **~97 integration test files** pass (including all focus-related tests)
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
- The new integration test `issue_491_split_focus_submodules.rs` verifies all public API items remain accessible

## Test Plan

- **New test**: `tests/issue_491_split_focus_submodules.rs` — 9 tests verifying that all public types, functions, and constants from the focus module remain accessible and produce correct results after the split
- **Existing tests preserved**: All existing tests pass without modification, including:
  - `tests/focus.rs` (17 tests)
  - `tests/issue_222_hierarchical_focus.rs` (20+ tests)
  - `tests/issue_204_activation_frequency_ranking.rs` (6 tests)
  - `tests/issue_206_gradient_flow_analysis.rs` (10+ tests)
  - `tests/issue_208_synapse_counts.rs` (9 tests)
  - `tests/issue_227_discovery_history.rs` (4 tests)
  - `tests/issue_306_constant_neuron_removal_with_bias_adjustment.rs` (4 tests)
  - `tests/issue_132_cost_of_growth.rs` (6 tests)
  - All impact calculation and regression tests
- **Inline unit tests preserved**: The 2 inline tests (lazy provider behaviour) were moved to `src/focus/tests.rs` and continue to pass
