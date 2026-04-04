## Summary

Split `src/config.rs` (587 lines) into a `src/config/` module directory with domain-specific submodules. Closes #981.

### Modules created

- **`config/helpers.rs`** — `parse_bool_env()`, `parse_optional_bool_env()` utilities
- **`config/user_facing.rs`** — User-controlled settings: verbose, GPU tuning, streaming, outlier analysis, watchdog, constant source thresholds, etc.
- **`config/observability.rs`** — `timing()`, `profile_mode()`, `ProfileMode`, `gpu_metrics()`
- **`config/detection.rs`** — Threshold tuning overrides: `noise_signal_threshold()`, `dominance_threshold()`, `gradient_threshold()`
- **`config/mod.rs`** — Module-level documentation, re-exports all public items, and unit tests

All public API paths (`crate::config::*`) remain unchanged via re-exports — no callers needed updating.

## Evidence

This is a pure refactoring with no functional changes. All existing tests pass unchanged.

## Test Plan

- All existing unit tests in `config::tests` preserved and passing
- All integration tests in `tests/infrastructure/issue_717_config_env_vars.rs` passing
- Full quality gate (`./quality.sh`) passes
