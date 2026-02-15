## Summary

Split `src/focus/ranking.rs` (~1,250 lines) into a `ranking/` directory with four focused sub-modules, each well under 1,500 lines. Closes #564.

The decomposition follows natural responsibility boundaries:

| Sub-module | Lines | Responsibility |
|---|---|---|
| `ranking/mod.rs` | 519 | Public API, orchestration (`rank_focus_neurons`, `rank_focus_neurons_with_history`) |
| `ranking/record_providers.rs` | 161 | `RecordProvider` trait, eager and lazy implementations |
| `ranking/score_calculation.rs` | 200 | `RankedNeuron`, error/activation/frequency computation |
| `ranking/removal_candidates.rs` | 315 | `RemovalCandidate`, `SynapseCounts`, constant neuron removal |

No public API changes — all types and functions remain accessible via `neat_ai_discovery::focus::*`.

Also extracted a shared `identify_removal_candidates` helper to eliminate duplicated removal candidate logic between `rank_focus_neurons` and `rank_focus_neurons_with_history`.

## Evidence

This is a pure refactoring with no UI or performance changes. All existing tests pass without modification, confirmed by `quality.sh` passing cleanly (fmt, clippy, check, 600+ tests, release build).

## Test Plan

- Added `tests/issue_564_split_ranking_submodules.rs` with 7 integration tests verifying:
  - `SynapseCounts` accessible and correct after split
  - `calculate_removal_savings` accessible and correct after split
  - `rank_focus_neurons` accessible with correct `RankFocusStats` and `RankedNeuron` fields
  - `rank_focus_neurons_with_history` accessible after split
  - `RemovalCandidate` fields accessible after split
  - `SelectionStats` type alias accessible after split
  - `constant_neuron_removals` field accessible after split
- All 13 existing `focus_ranking` tests pass unchanged
- All 9 existing `issue_491_split_focus_submodules` tests pass unchanged
- Full test suite (600+ tests) passes with `--test-threads=1`
