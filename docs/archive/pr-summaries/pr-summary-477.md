## Summary

Updated Rust edition from 2021 to 2024 in `Cargo.toml` and fixed all breaking changes introduced by the new edition across the entire codebase.

### Changes required by Rust 2024 edition

1. **`gen` is now a reserved keyword** — Replaced `rng.gen()` with `rng.r#gen()` in `src/analysis/utils/deadline.rs` (2 occurrences).

2. **`#[no_mangle]` requires `unsafe()` wrapping** — Changed all 13 `#[no_mangle]` attributes in `src/lib.rs` to `#[unsafe(no_mangle)]`.

3. **Pattern binding mode changes** — Fixed 4 closure patterns in `src/analysis/gpu/analyzer.rs`, `src/analysis/observation_range.rs`, and `src/analysis/sentinel_gating.rs` where explicit dereferences conflicted with implicit borrowing in edition 2024.

4. **`std::env::set_var` and `std::env::remove_var` are now unsafe** — Wrapped all 39+ occurrences across `src/`, `tests/`, and `benches/` in `unsafe {}` blocks with `// SAFETY:` comments explaining that tests run single-threaded (`--test-threads=1`).

5. **Collapsible `if` statements (clippy)** — Collapsed 22 nested `if let` + `if` patterns into single `if` expressions using Rust 2024 let-chains (`if let ... && ...`) across 14 source files.

## Evidence

This is a backend/build configuration change with no UI components. Verification is via the test suite:

- All 463 unit tests pass
- All integration tests pass
- Clippy passes with `-D warnings`
- Release build succeeds
- New edition-specific tests confirm Rust 2024 features are active

## Test Plan

- Added `tests/issue_477_rust_edition_2024.rs` with 3 tests:
  - `let_chains_available_in_edition_2024` — Verifies let-chain syntax (edition 2024 feature) compiles and works
  - `gen_is_reserved_keyword_in_edition_2024` — Confirms `gen` is reserved and `r#gen` raw identifier works
  - `unsafe_attribute_syntax_accepted` — Validates `#[unsafe(no_mangle)]` attribute syntax
- All existing tests continue to pass unchanged (no tests removed or modified in logic)
