## Summary

Add explicit `rustfmt.toml` and `clippy.toml` configuration files to pin formatting and lint rules, preventing churn when Rust toolchain defaults change between versions. Selectively enable useful pedantic and nursery clippy lints via `[lints.clippy]` in `Cargo.toml`, and fix all warnings introduced by the stricter configuration. Closes #675.

## Changes

### New files
- **`rustfmt.toml`** — Pins `max_width = 100`, `newline_style = "Unix"`, `edition = "2024"`, `use_field_init_shorthand`, and `use_try_shorthand`.
- **`clippy.toml`** — Pins threshold configuration: `too-many-lines-threshold`, `cognitive-complexity-threshold`, `enum-variant-size-threshold`, `too-large-for-stack`.

### Modified files
- **`Cargo.toml`** — Added `[lints.clippy]` section enabling selective pedantic/nursery lints:
  - `uninlined_format_args` (deny)
  - `cloned_instead_of_copied`, `explicit_iter_loop`, `implicit_clone`, `map_unwrap_or`, `redundant_closure_for_method_calls`, `semicolon_if_nothing_returned`, `unnested_or_patterns` (warn)
  - `redundant_clone` (nursery, warn)
- **93 source/test/bench files** — Auto-fixed warnings from the new lints (redundant closures, redundant clones, `map().unwrap_or()` patterns, semicolons, explicit iter loops, etc.)

### .gitignore
No update needed — `rustfmt.toml` and `clippy.toml` (without leading dots) are not affected by the `.*` ignore pattern.

## Evidence

This is a backend/configuration change with no visual output. Evidence:
- `./quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)
- All 191 test suites pass with zero failures

## Test Plan

- No new tests required — this change adds lint configuration and fixes existing warnings
- All existing tests continue to pass unchanged
- Quality gate (`./quality.sh`) passes with the new configuration
