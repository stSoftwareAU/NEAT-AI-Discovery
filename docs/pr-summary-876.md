## Summary

Consolidated clippy lint configuration to eliminate DRY violations across the project. Closes #876.

All clippy lint rules are now configured exclusively in `Cargo.toml` `[lints.clippy]`. The `quality.sh`
script and `AGENTS.md` documentation have been simplified to use `cargo clippy --all-targets --all-features -- -D warnings`
without additional `-D`/`-W` flags.

Changes:
- **Cargo.toml**: Added `filter_next = "deny"` and `collapsible_if = "deny"` (previously only enforced via CLI flags)
- **quality.sh**: Simplified clippy invocation to just `-D warnings` (lint rules come from Cargo.toml)
- **AGENTS.md**: Updated quality gate and quick reference sections to reflect simplified clippy command

**Note**: `.github/workflows/ci.yml` line 224 still has the old inline flags. This file cannot be pushed by
the automated worker (requires `workflow` OAuth scope). The ci.yml change should be applied manually or via
a workflow-scoped token.

## Evidence

- `cargo clippy --all-targets --all-features -- -D warnings` passes cleanly (verified via `./quality.sh`)
- All lint rules previously enforced via CLI flags are now in `Cargo.toml` — no regressions

## Test Plan

- Ran full `./quality.sh` suite which includes clippy, fmt, check, tests, doc build, and release build
- All checks pass with the consolidated configuration
- Verified `filter_next` and `collapsible_if` are enforced via Cargo.toml (any violations would cause clippy failure with `-D warnings`)
