## Summary

Split the monolithic `src/export.rs` (778 lines) into a focused `src/export/` module directory with four sub-modules, each with a single clear responsibility. Closes #980.

### Module breakdown

| Module | Lines | Responsibility |
|--------|-------|---------------|
| `types.rs` | 172 | All serialisable data structures (`SnapshotMeta`, `NeuronRecording`, `SynapseDerived`, `ReconstructionCheck`, etc.) |
| `stats.rs` | 185 | `compute_stats()`, `json_safe_f32()`, `apply_squash()` + unit tests |
| `snapshot.rs` | 377 | Main `export_visualisation_snapshot()` pipeline |
| `timestamp.rs` | 67 | `chrono_lite_now()`, `is_leap_year()` + unit test |
| `mod.rs` | 21 | Re-exports for backward compatibility |

All public API items are re-exported from `mod.rs` so existing code (`crate::export::ExportOptions`, `crate::export::export_visualisation_snapshot`, etc.) continues to work without changes.

No functional changes — pure structural refactor.

## Evidence
- All 158 tests pass (0 failed, 0 ignored)
- `cargo clippy` passes with zero warnings
- `cargo fmt` reports no formatting issues
- `./quality.sh` passes all checks including release build

## Test Plan
- All existing unit tests preserved and relocated to their respective modules:
  - `export::stats::tests` — 7 tests (squash functions, compute_stats variants)
  - `export::timestamp::tests` — 1 test (chrono_lite_now format)
- All integration tests in `tests/export/` pass unchanged
- No tests were added, removed, or modified — only relocated
