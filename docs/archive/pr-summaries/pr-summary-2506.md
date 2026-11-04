## Summary

`fuzz/Cargo.lock` records the root crate as a path package with its version, and nothing restamped that line after a `Cargo.toml` bump, so `cargo fuzz run --locked` failed. The new `quality/fuzz_lockfile.sh` has two modes: `sync` rewrites only that one `version = "…"` line to the `Cargo.toml` version, and `check` runs `cargo metadata --locked` on `fuzz/Cargo.toml`. `./quality.sh` runs `sync` then `check`, and `./bump-deps.sh` runs `sync` (Phase 3c). `scripts/fuzz-ci.sh` and `docs/ci-fuzzing-workflow.yml` run `sync` then `check` before their `fuzz run --locked` steps, so the stale lock left by CI's automatic version bump is restamped at fuzz time. `fuzz/Cargo.lock` is restamped here (`0.74.322` → `0.74.342`). Closes #2506.

Version bump: the merge retains the PR's `0.74.349` in `Cargo.toml` and `Cargo.lock`, above Develop's `0.74.348`, and synchronises `fuzz/Cargo.lock` to match.

CI repair: document the sync and check commands as separate quality-gate steps, and recognise both ShellCheck diagnostics for the cleanup function invoked by the EXIT trap. The cleanup behaviour is unchanged.

Merge resolution: retain both the fuzz lockfile and Parquet decode preflight entries in the changelog.

## Spec

### Intent and Rationale

- The path-package line is the only part of `fuzz/Cargo.lock` that moves with the crate version, so a text restamp of that line fixes the failure without re-resolving anything. `cargo update -p neat_ai_discovery` (the issue's example) can reach the registry and rewrite more than the one line, which the `bump-deps.sh` / Issue #1865 rules guard against.
- `check` uses the issue's suggested `cargo metadata --locked --manifest-path fuzz/Cargo.toml`, so any other drift (a `fuzz/Cargo.toml` dependency edit with no lock update) also fails the gate locally.

### Essential Design Decisions

- `quality.sh` syncs before it checks, so a version bump never fails the gate; only genuine drift does. It already rewrites files (`cargo fmt --all`), and `sync` is a version stamp, not a dependency change, so Issue #1865 still holds (its test passes).
- `sync` refuses to guess: it needs exactly one `[[package]]` block named after the crate with no `source` line, and verifies exactly one line changed against a stable snapshot before atomically replacing the lock while preserving its mode.
- `.github/workflows/ci.yml` is deliberately unchanged (needs approval per `AGENTS.md`). Its `version-increment` job still bumps only the root `Cargo.lock`, so after CI's own bump `fuzz/Cargo.lock` is stale. The fuzz entry points cover that instead: `scripts/fuzz-ci.sh` runs `quality/fuzz_lockfile.sh sync` and `check` after installing cargo-fuzz and before the `cargo +nightly fuzz run --locked` loop, and the documented `docs/ci-fuzzing-workflow.yml` has a `Sync fuzz/Cargo.lock version` step ahead of its `--locked` runs. A `sync` step in `version-increment` would also work but needs approval.

### Undiscoverable Facts

- None.

## Evidence

Backend/scripting change; no UI. Commands run on the final tree:

- `./quality/fuzz_lockfile.sh sync` → `fuzz/Cargo.lock: neat_ai_discovery 0.74.322 -> 0.74.342`; a second run → `already records … nothing to do`.
- `./quality/fuzz_lockfile.sh check` → `OK — cargo metadata --locked accepts fuzz/Cargo.lock`. Before the restamp, `cargo metadata --locked --manifest-path fuzz/Cargo.toml` failed with `cannot update the lock file … because --locked was passed`.
- `bash tests/bump_deps_test.sh` → 107 passed, 0 failed. Its fixtures have no `quality/` root and run only `--dry-run`, so they never reach Phase 3c.
- A throwaway fixture run of `bump-deps.sh --no-network` (non-dry-run) restamped the lock; with `quality/fuzz_lockfile.sh` removed it exited 1 naming the missing script.
- `./quality/shellcheck.sh .` and `./quality/bash_syntax.sh .` passed (19 scripts).

## Test Plan

The existing `tests/issue_1992_ci_doc_accuracy.rs::ordered_gate_list_matches_quality_sh` checks the documented sequence, and `tests/issue_1898_shellcheck_gate.rs::gate_passes_over_the_repository_itself` checks the shell scripts.

New `tests/issue_2506_fuzz_lockfile_sync.rs` (fixtures built in temp directories; real `cargo`, offline):

- `sync_restamps_only_the_path_package_version`: registry packages sharing the stale version string stay untouched.
- `sync_is_a_no_op_when_already_current`.
- `sync_fails_loud_when_the_path_package_is_missing`: non-zero exit, stderr names the crate and `Cargo.lock`, lock unchanged.
- `check_rejects_a_stale_lock_then_accepts_it_after_sync`: the same fixture is rejected, then accepted after only the lock version changes.
- `check_accepts_a_current_lock`.
- `fuzz_ci_restamps_a_stale_lock_and_checks_it_before_fuzzing`: runs a copy of `scripts/fuzz-ci.sh` in a temp root whose lock is stale (stub `cargo`/`rustup`), asserts the lock is restamped and a `metadata --locked` call precedes the first `fuzz run --locked`.
- `documented_fuzz_workflow_syncs_the_lock_before_its_locked_runs`: extracts and executes both documented lockfile commands against a stale fixture with the real script (copied to its own repo-relative path, so a moved script turns the test red) and offline Cargo, and verifies the step precedes fuzzing.
- `quality_gate_syncs_then_checks_the_fuzz_lockfile`: runs the real `quality.sh` with a stubbed `cargo` and asserts a `metadata --locked … fuzz/Cargo.toml` call that comes before `deny`. It runs the gate and helpers in a private fixture with a stale lock.

Results on the final tree: `cargo test --test issue_2506_fuzz_lockfile_sync` → 8 passed, and with `--test issue_1912_fuzz_ci_pinned_install --test issue_1912_cargo_install_pinning_gate --test issue_1939_documented_commands` also green; `cargo fmt --all -- --check` clean; `cargo clippy --test issue_2506_fuzz_lockfile_sync -- -D warnings` clean.

Red checks (on a scratch copy of the tree; script behaviour broken one way at a time): sync rewriting every `version =` line → `sync_restamps_only_the_path_package_version` red; sync exiting 0 for a missing package → `sync_fails_loud_when_the_path_package_is_missing` red; `check` always exiting 0 → `check_rejects_a_stale_lock_then_accepts_it_after_sync` red; `quality.sh` check call replaced with `true` → `quality_gate_syncs_then_checks_the_fuzz_lockfile` red; check moved after `cargo deny` → same test red. The existing gate and fuzz CI tests were subsequently isolated for the review follow-up below.

Not covered by a test: the `bump-deps.sh` Phase 3c call site (exercised only by the throwaway fixture run above, since `tests/bump_deps_test.sh` cannot run a non-dry-run); `check_accepts_a_current_lock` was not red-checked. The new fuzz-ci test was red-checked by deleting the `sync`/`check` lines from `scripts/fuzz-ci.sh` (test failed), then restored.

Docs: `docs/ci-fuzzing-workflow.yml` (new sync step), `CONTRIBUTING.md` (quality.sh step list, renumbered), `README.md` (fuzz section), `AGENTS.md` (Version Bumps), `CHANGELOG.md`.

Full gate: `timeout 595 ./quality.sh` on the final tree passed `bash -n`, shellcheck, cargo-install pinning, the PR-summary layout check, the new `fuzz-lockfile … OK` step, `cargo deny check` (`advisories ok, bans ok, licenses ok, sources ok`), `cargo build`, `cargo fmt` and `cargo clippy --all-targets --all-features`. It then hit the 595 s cap (exit 124) while `cargo check --all-targets` was compiling, so `cargo test --lib --tests`, `cargo doc` and the release build did not run.

Not done / risk: the CI `version-increment` job is unchanged by design; the fuzz entry points restamp the lock themselves. The last gate stages (full `cargo test`, `cargo doc`, release build) were not run here.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 595 s foreground cap (exit 124) during cargo check; every stage before it passed; the new tests, fmt and clippy were run separately; CI runs the full gate" -->

Review follow-up: the old workflow test only matched a literal sync command and did not exercise the check. The remaining contract is that both documented commands successfully restamp and validate a stale lock before fuzzing, covered by `tests/issue_2506_fuzz_lockfile_sync.rs::documented_fuzz_workflow_syncs_the_lock_before_its_locked_runs`.

Review follow-up: concurrent syncs previously compared against a changing lock and rewrote it in place, and the gate and fuzz CI tests touched the checkout lock. Sync now snapshots the lock, accepts an already-current snapshot, and atomically replaces the file with its mode preserved. Gate and fuzz CI tests use private roots. The obsolete shared-checkout mutation and in-place write are replaced by isolated execution and atomic restamping, covered by `tests/issue_2506_fuzz_lockfile_sync.rs::concurrent_syncs_preserve_the_lock_and_its_mode`, `tests/issue_2506_fuzz_lockfile_sync.rs::quality_gate_syncs_then_checks_the_fuzz_lockfile`, `tests/issue_1865_quality_gate_no_dep_mutation.rs::quality_gate_never_mutates_the_dependency_graph`, and `tests/issue_1912_fuzz_ci_pinned_install.rs::both_fuzz_targets_still_run_with_the_requested_budget`.
