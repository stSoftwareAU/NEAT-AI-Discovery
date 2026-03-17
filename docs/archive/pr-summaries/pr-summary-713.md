## Summary

Deduplicated GPU environment variable setup (`suppress_mesa_warnings_if_requested` and
`ensure_xdg_runtime_dir`) that was identically defined in both `memory.rs` and `platform.rs`.
The canonical definitions now live solely in `platform.rs`, and the duplicates in `memory.rs`
have been removed. All existing callers already import via `analysis::utils` re-exports from
`platform.rs`, so no call sites needed updating. Closes #713.

## Evidence

This is a backend-only refactoring with no UI changes. The `quality.sh` gate passes cleanly,
confirming that all compilation, linting, and tests succeed after the removal.

## Test Plan

- Added `tests/issue_713_deduplicate_gpu_env_setup.rs` with three integration tests:
  - `platform_suppress_mesa_warnings_does_not_panic` — calls the canonical function directly
  - `platform_ensure_xdg_runtime_dir_does_not_panic` — calls the canonical function directly
  - `utils_reexports_resolve_to_platform` — calls the re-exported versions via `utils`
- Existing tests in `platform.rs` (`test_suppress_mesa_warnings_does_not_panic`,
  `test_ensure_xdg_runtime_dir_does_not_panic`) continue to pass
- Full `quality.sh` passes (fmt, clippy, check, test, doc, release build)
