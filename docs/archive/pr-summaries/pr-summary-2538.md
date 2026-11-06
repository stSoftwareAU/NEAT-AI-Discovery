## Summary

The `gpu` integration test binary ran the issue #713 smoke tests (which fire the process-wide `Once` that writes `XDG_RUNTIME_DIR`) in parallel with the issue #1873 `setup_gpu_environment_is_safe_and_consistent` test, whose two calls must return the same verdict and depend on that variable. The three #713 tests and the three #1873 tests that call `setup_gpu_environment` are now `#[serial_test::serial]`, and each saves and restores `XDG_RUNTIME_DIR` and the Mesa variables through a shared `EnvGuard`. Their SAFETY comments now state the real precondition. Test-only change. Closes #2538.

Version bump: `Cargo.toml` and the root `Cargo.lock` updated for this change, and `fuzz/Cargo.lock` restamped with `quality/fuzz_lockfile.sh sync`; the merged commit records the final version.

## Spec

### Intent and Rationale

- The lib-test copy of this race was fixed by #2497; the integration binary kept the original pattern. The fix mirrors it: `#[serial]` plus save/restore guards.
- `tests/gpu/issue_1873_gpu_env_setup_thread_guard.rs::setup_gpu_environment_never_writes_while_threads_live` and `..._skips_or_is_unnecessary_on_linux` are also serialised: they read the same variables and would otherwise observe the write.

### Essential Design Decisions

- Each serialised test begins with `assert!(serial_test::is_locked_serially(None), ...)`. A race cannot be reproduced deterministically, so this makes dropping the attribute turn the test red.
- The guard lives in `tests/gpu/env_guard.rs`, not `tests/common/mod.rs`, because that module is compiled into many other test binaries where an unused helper would warn.
- Assumed: tests elsewhere in the `gpu` binary do not touch these variables (grep of `tests/` for `XDG_RUNTIME_DIR`, `setup_gpu_environment` and `ensure_xdg_runtime_dir` finds only the two files edited, plus doc-parity and ledger tests in other binaries).

### Undiscoverable Facts

- `fuzz/Cargo.lock` was at 0.74.349 while the root was 0.74.350; `sync` corrected both.

## Evidence

Backend/test-only change, no UI. The six tests below run on the final tree: `cargo test --locked --test gpu env_setup` gave `6 passed`.

- `tests/gpu/issue_713_deduplicate_gpu_env_setup.rs`: `platform_suppress_mesa_warnings_does_not_panic`, `platform_ensure_xdg_runtime_dir_does_not_panic`, `utils_reexports_resolve_to_platform`
- `tests/gpu/issue_1873_gpu_env_setup_thread_guard.rs`: `setup_gpu_environment_is_safe_and_consistent`, `setup_gpu_environment_never_writes_while_threads_live`, `setup_gpu_environment_skips_or_is_unnecessary_on_linux`

## Acceptance Criteria

The issue states no acceptance-criteria list, so no reviewer sub-agents ran. Every item of the issue's "Suggested fix" is done: the three #713 smoke tests and the env-dependent #1873 tests are serial, the smoke tests save and restore `XDG_RUNTIME_DIR` and the Mesa variables, and the SAFETY comments state the real precondition.

## Test Plan

- Edited tests (no assertion removed from any of them; the existing calls and `assert_eq!`/`assert_ne!`/`assert!(matches!…)` are unchanged): the six listed above. Each gains `#[serial_test::serial]`, an `is_locked_serially` assertion and an `EnvGuard`.
- Red check: with every `#[serial_test::serial]` line removed from both files and the rest kept, `cargo test --locked --test gpu env_setup` gave `0 passed; 6 failed`, each panicking with "must be serialised with every env-mutating or env-dependent test in the binary". Attribute restored afterwards; the same command then passes (6 passed).
- Not reproduced: the original flaky failure itself (scheduler-dependent, one `Once` per run); the serial-lock assertion is the deterministic stand-in.
- `cargo fmt --all -- --check` clean; `cargo clippy --locked --tests -- -D warnings` clean (run by the executor before the version bump).
- `./quality/fuzz_lockfile.sh check`: OK.
- Docs: `CHANGELOG.md` entry added; `README.md` and `docs/` do not describe these test internals.
- Full gate: see the note below.
