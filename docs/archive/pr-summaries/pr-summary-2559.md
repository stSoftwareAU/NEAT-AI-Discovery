## Summary

The `coverage` job (`.github/workflows/cargo-quality.yml`) and the `msrv` job (`.github/workflows/msrv.yml`) now restore a Cargo dependency cache (`~/.cargo/registry`, `~/.cargo/git`) after Rust setup and before their first cargo command, so fresh runners stop re-downloading crates. Closes #2559.

Version bump: not applicable. Only workflow files and a test change, with no library code touched; CI's `version-increment` job owns the version.

## Spec

### Intent and Rationale

- Follows `ci.yml`'s existing "Cache dependencies" step (same `actions/cache` v5.0.5 SHA, re-verified with `gh api repos/actions/cache/commits/v5.0.5`).
- Each job gets its own key prefix, so caches are never shared across jobs.

### Essential Design Decisions

- `target/` is not cached, as the issue says: disk pressure (Issue #1489) and coverage's different instrumentation.
- Coverage key: `runner.os` + job name + `hashFiles('rust-toolchain.toml')` + `hashFiles('**/Cargo.lock')`. The MSRV key names the declared toolchain (`1.99`) literally; a test requires it to equal the `toolchain:` input, so an MSRV bump cannot reuse the old cache.
- The restore key is the key minus the lockfile hash, so a changed `Cargo.lock` still reuses the toolchain-matched cache and a miss falls back to a clean build.
- `actions/cache` isolates PR caches from each other by ref, so PR isolation needs no extra configuration.

### Undiscoverable Facts

- No cache-hit comparison was measured, so the saving is unmeasured, as the issue states. Cold-versus-warm timings come from the first CI runs.

## Evidence

Backend/CI only, no UI. `tests/issue_2559_cargo_cache_coverage_msrv.rs` runs plain-text checks against both real workflow files and against inline fixtures. `cargo test` over that file plus `issue_2395_msrv_declared`, `issue_2390_coverage_workflow_guards`, `issue_1287_workflow_timeouts` and `issue_1288_workflow_concurrency`: 67 passed.

## Acceptance Criteria

The issue has no acceptance-criteria heading and nothing it asks is left undone, so no review block is needed.

## Test Plan

- Added `tests/issue_2559_cargo_cache_coverage_msrv.rs`: 2 real-file tests (coverage, msrv), 1 positive fixture, 7 negative fixtures (cache step missing, cache after the cargo step, `target/` cached as an inline path or inside a `|` block, key lacking `Cargo.lock`, msrv key lacking the toolchain, coverage key lacking `rust-toolchain.toml`). Each negative test asserts the error text of the check it targets.
- Red check: removing the cache step from `msrv.yml` made `real_msrv_job_restores_cargo_cache_before_its_first_cargo_command` fail; the same for `cargo-quality.yml` and `real_coverage_job_restores_cargo_cache_before_its_first_cargo_command`. Both restored.
- Existing tests edited: none.
- `cargo fmt --all -- --check` and `cargo clippy --test issue_2559_cargo_cache_coverage_msrv -- -D warnings` pass.
- Twinned rules: `ci.yml`'s cache step is the model; it is left untouched (AGENTS.md forbids editing it without approval). `msrv_toolchain` in `tests/issue_2395_msrv_declared.rs` reads the first `toolchain:` line, and no new line starts with that word.

<!-- vibe-quality-gate-skipped reason="./quality.sh hit the 580 s tool cap during a cold build in a fresh worktree; clippy, fmt and the touched tests were run separately. CI and the worker run the full gate." -->
