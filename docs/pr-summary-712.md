## Summary

Add missing `// SAFETY:` comments to the two remaining undocumented `unsafe` blocks in
`src/analysis/utils/platform.rs` (lines 31 and 36). These blocks call `env::set_var()` for
`MESA_GLSL_CACHE_DISABLE` and `MESA_DEBUG` inside a `Once::call_once` guard, matching the
existing comment on the adjacent `EGL_LOG_LEVEL` block.

All other `unsafe` blocks in `src/` already had `// SAFETY:` comments from prior work
(#711/#724 for FFI blocks, and earlier commits for test env-var blocks). Closes #712.

## Evidence

No UI or performance changes. This is a documentation-only change (comments added to
existing code). Verified by running `quality.sh` which passes all checks including
`cargo clippy`, `cargo test`, and `cargo build --release`.

## Test Plan

- No new tests required — this change only adds comments to existing code
- All existing tests continue to pass (`quality.sh` green)
