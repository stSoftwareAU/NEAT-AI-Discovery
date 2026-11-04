## Summary

`test_ensure_xdg_runtime_dir_does_not_panic` and `test_suppress_mesa_warnings_does_not_panic` in `src/analysis/utils/platform.rs` are now `#[serial_test::serial]`, so they can no longer write `XDG_RUNTIME_DIR` (or the Mesa variables) while a serial test has it unset and asserts it stays unset. Each captures the variables it may touch and restores them on exit. Closes #2497.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- `#[serial]` only excludes other `#[serial]` tests, so the two unserialised smoke tests could fire their process-wide `Once` while `xdg_runtime_dir_rejects_preexisting_world_writable`, `xdg_runtime_dir_apply_rejects_symlink` or `test_setup_gpu_environment_skips_when_other_threads_live` had the variable unset.

### Essential Design Decisions

- A new test-only `EnvGuard::capture` records a variable's value without changing it; `Drop` restores or removes it. The guards are under `#[cfg(unix)]` (the `EnvGuard` type is unix-only), so the tests still compile elsewhere.
- The SAFETY comments now state the real precondition: serialised with every env-mutating test in the module (checked: every test that writes the environment on Linux is `#[serial]`).
- The separate integration-test binary `tests/gpu/issue_713_deduplicate_gpu_env_setup.rs` has the same smoke-test shape but shares no process with these tests, so it is untouched.

### Undiscoverable Facts

- None.

## Evidence

Backend/test-only change; no UI. Tests: `src/analysis/utils/platform.rs::tests::test_ensure_xdg_runtime_dir_does_not_panic` and `::test_suppress_mesa_warnings_does_not_panic`.

Docs: grepped `docs/` and `README.md` for both test names and found no mention; no doc change owed.

## Test Plan

- `cargo test --lib utils::platform`: 16 passed. `cargo fmt --check` and `cargo clippy --lib --tests -- -D warnings` clean.
- Red check: not performed. The race is a scheduling window and cannot be staged deterministically, so no test goes red on the base branch; the change is verified by the serial attribute and by reading that no other env-writing test in the module is non-serial.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh did not finish within the 600 s foreground cap (same as the previous run on this repo) and was stopped; fmt, clippy and the targeted tests were run; CI runs the full gate" -->

## Not checked

- Full `./quality.sh` did not complete locally.
- The race itself is not reproduced; the fix is by construction.
- No Spec/Standards reviewer sub-agents were dispatched: the issue has no acceptance-criteria section.
