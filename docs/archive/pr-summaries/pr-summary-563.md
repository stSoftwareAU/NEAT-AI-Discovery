## Summary

Split `src/analysis/recommendation/epistatic.rs` (~1,500 lines) into focused sub-modules
following the same decomposition pattern used for `synapse.rs` (Issue #482). Closes #563.

The monolithic file is replaced by an `epistatic/` directory with five sub-modules:

| Sub-module | Lines | Responsibility |
|-----------|-------|----------------|
| `mod.rs` | 155 | Public API, shared types, re-exports |
| `candidate_generation.rs` | 417 | Pair generation, complementarity analysis, conversion |
| `pre_screening.rs` | 413 | Residual analysis, synergistic detection (Issue #189) |
| `deduplication.rs` | 100 | Dominant-neuron deduplication (Issue #509) |
| `scoring.rs` | 534 | Interference detection, filtering (Issue #415) |

No public API changes — all existing imports via `analysis::epistatic::` continue to work
through re-exports in `epistatic/mod.rs`.

## Evidence

This is a pure refactoring with no behavioural changes. All existing tests pass unchanged,
and `quality.sh` passes cleanly (fmt, clippy, check, test, release build).

## Test Plan

- All existing unit tests preserved and moved to their respective sub-modules
- All 7 integration test files referencing epistatic remain unchanged and pass:
  - `tests/issue_202_epistatic_neuron_pairs.rs`
  - `tests/issue_189_synergistic_discovery.rs`
  - `tests/issue_415_combo_successful_interference.rs`
  - `tests/issue_508_prescreen_individual_operations.rs`
  - `tests/issue_509_deduplicate_dominant_neuron.rs`
  - `tests/issue_510_coordinated_structural_weight_variants.rs`
  - `tests/issue_340_discovery_categories.rs`
- `quality.sh` passes all checks (fmt, clippy, check, test, release build)
