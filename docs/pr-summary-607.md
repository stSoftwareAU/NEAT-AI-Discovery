## Summary

Split `src/analysis/activation.rs` (1,130 lines) into focused sub-modules under `src/analysis/activation/`. Closes #607.

The file was split into three sub-modules:
- **`activation/functions.rs`** — CPU activation function implementations (15 functions)
- **`activation/specs.rs`** — Activation candidate specifications, GPU ID mapping, bias range helpers
- **`activation/simulation.rs`** — Target simulation modes, activation predicates, output variance checking
- **`activation/mod.rs`** — Public API re-exports and all existing tests (unchanged)

All public API remains unchanged — existing imports like `use crate::analysis::activation::*` continue to work without modification.

## Evidence

This is a pure refactoring change with no UI or performance impact. All existing tests pass unmodified.

- `cargo build` — compiles cleanly
- `cargo clippy` — no warnings
- `cargo test` — all tests pass
- `./quality.sh` — passes all checks including release build

## Test Plan

- All existing activation module tests preserved in `activation/mod.rs` (31 tests)
- No test modifications required — all tests pass as-is via `use super::*` re-exports
- Full quality gate (`./quality.sh`) passes cleanly
