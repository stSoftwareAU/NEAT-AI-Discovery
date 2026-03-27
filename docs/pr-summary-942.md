## Summary

Split `src/record/mod.rs` into focused sub-modules by extracting the large unit test block (~730 lines) into a dedicated `tests.rs` file. Closes #942.

The `record/mod.rs` was 808 lines, with only 76 lines of orchestration code and 730 lines of tests. The business logic was already well-structured across `validation.rs` and `processing.rs` (from Issue #604). The refactoring extracts tests into `src/record/tests.rs` following the project convention (AGENTS.md: "If unit tests in `src/` grow large, extract them into a dedicated `tests.rs` module file").

After the split:
- `mod.rs` — 79 lines (orchestration + re-exports, well under the 200-line target)
- `validation.rs` — 84 lines (input validation and observation index resolution)
- `processing.rs` — 89 lines (record building from training data)
- `tests.rs` — 729 lines (all unit tests, unchanged)

Public API is unchanged. No modifications needed in other source files. All existing tests pass without modification. `./quality.sh` passes cleanly.

## Evidence

- `src/record/mod.rs` reduced from 808 lines to 79 lines (under 200-line target)
- Each sub-module has a single responsibility
- All 11 existing unit tests pass unmodified
- `./quality.sh` passes with no warnings

## Test Plan

- All existing tests extracted to `src/record/tests.rs` without modification
- No new tests required (this is a pure refactoring with no behaviour changes)
- Verified all tests pass via `./quality.sh`
