## Summary

Split `src/analysis/synapse/target_analysis.rs` (~1,087 lines) into focused sub-modules
under `src/analysis/synapse/target_analysis/`, improving maintainability and following the
Single Responsibility Principle. Closes #599.

### New sub-module structure

| File | Lines | Responsibility |
|------|-------|---------------|
| `mod.rs` | 360 | Public API, types (`TargetAnalysisContext`, `TargetAnalysisResults`, `InputMetadata`), main `analyse_single_target` orchestration |
| `evaluation.rs` | 409 | GPU work submission, helpful/harmful result collection and candidate processing |
| `candidate_selection.rs` | 143 | Epistatic pair detection, synergistic candidate detection, redundant path detection |
| `statistics.rs` | 326 | Source filtering, record loading, sample building orchestration, harmful sample preparation |

All files are well under the ~1,500-line target.

### What did NOT change

- Public API remains identical (`TargetAnalysisContext` and `analyse_single_target` accessed from `synapse/mod.rs`)
- No logic changes — pure structural refactoring
- All existing tests pass without modification

## Evidence

This is a pure code reorganisation with no UI or performance changes. Evidence is provided
by all existing tests passing unchanged:

- `cargo test --lib --tests --all-features -- --test-threads=1` — all tests pass
- `cargo clippy --all-targets --all-features -- -D warnings` — clean
- `./quality.sh` — all checks pass

## Test Plan

- No new tests required — this is a structural refactoring only
- All 97+ existing integration tests pass without modification
- All inline unit tests pass without modification
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
