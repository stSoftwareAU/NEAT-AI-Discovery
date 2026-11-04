## Summary

`fuzz/Cargo.lock` records the root crate as a path package with its version, and nothing restamped that line after a `Cargo.toml` bump, so `cargo fuzz run --locked` failed. The new `quality/fuzz_lockfile.sh` has two modes: `sync` rewrites only that one `version = "…"` line to the `Cargo.toml` version, and `check` runs `cargo metadata --locked` on `fuzz/Cargo.toml`. `./quality.sh` runs `sync` then `check`, and `./bump-deps.sh` runs `sync` (Phase 3c). `fuzz/Cargo.lock` is restamped here (`0.74.322` → `0.74.342`). Closes #2506.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The path-package line is the only part of `fuzz/Cargo.lock` that moves with the crate version, so a text restamp of that line fixes the failure without re-resolving anything. `cargo update -p neat_ai_discovery` (the issue's example) can reach the registry and rewrite more than the one line, which the `bump-deps.sh` / Issue #1865 rules guard against.
- `check` uses the issue's suggested `cargo metadata --locked --manifest-path fuzz/Cargo.toml`, so any other drift (a `fuzz/Cargo.toml` dependency edit with no lock update) also fails the gate locally.

### Essential Design Decisions

- `quality.sh` syncs before it checks, so a version bump never fails the gate; only genuine drift does. It already rewrites files (`cargo fmt --all`), and `sync` is a version stamp, not a dependency change, so Issue #1865 still holds (its test passes).
- `sync` refuses to guess: it needs exactly one `[[package]]` block named after the crate with no `source` line, and verifies exactly one line changed before copying over the lock.
- `.github/workflows/ci.yml` is deliberately unchanged (needs approval per `AGENTS.md`). Its `version-increment` job still bumps only the root `Cargo.lock`, so after CI's own bump `fuzz/Cargo.lock` is stale until the next `./quality.sh` or `./bump-deps.sh` run. A `quality/fuzz_lockfile.sh sync` step in that job would close the window. The CHANGELOG states this.

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

New `tests/issue_2506_fuzz_lockfile_sync.rs` (fixtures built in temp directories; real `cargo`, offline):

- `sync_restamps_only_the_path_package_version`: registry packages sharing the stale version string stay untouched.
- `sync_is_a_no_op_when_already_current`.
- `sync_fails_loud_when_the_path_package_is_missing`: non-zero exit, stderr names the crate and `Cargo.lock`, lock unchanged.
- `check_rejects_a_stale_lock_then_accepts_it_after_sync`: the same fixture is rejected, then accepted after only the lock version changes.
- `check_accepts_a_current_lock`.
- `quality_gate_syncs_then_checks_the_fuzz_lockfile`: runs the real `quality.sh` with a stubbed `cargo` and asserts a `metadata --locked … fuzz/Cargo.toml` call that comes before `deny`. It snapshots `fuzz/Cargo.lock` and restores it on drop.

Results on the final tree: `cargo test --test issue_2506_fuzz_lockfile_sync --test issue_1865_quality_gate_no_dep_mutation --test issue_1427_fuzz_cargo_lock_committed` → 9 passed; `cargo fmt --all -- --check` clean; `cargo clippy --test issue_2506_fuzz_lockfile_sync -- -D warnings` clean.

Red checks (on a scratch copy of the tree; script behaviour broken one way at a time): sync rewriting every `version =` line → `sync_restamps_only_the_path_package_version` red; sync exiting 0 for a missing package → `sync_fails_loud_when_the_path_package_is_missing` red; `check` always exiting 0 → `check_rejects_a_stale_lock_then_accepts_it_after_sync` red; `quality.sh` check call replaced with `true` → `quality_gate_syncs_then_checks_the_fuzz_lockfile` red; check moved after `cargo deny` → same test red. No existing test was edited.

Not covered by a test: the `bump-deps.sh` Phase 3c call site (exercised only by the throwaway fixture run above, since `tests/bump_deps_test.sh` cannot run a non-dry-run); `check_accepts_a_current_lock` was not red-checked.

Docs: `CONTRIBUTING.md` (quality.sh step list, renumbered), `README.md` (fuzz section), `AGENTS.md` (Version Bumps), `CHANGELOG.md`.

Full gate: `timeout 595 ./quality.sh` on the final tree passed `bash -n`, shellcheck, cargo-install pinning, the PR-summary layout check, the new `fuzz-lockfile … OK` step, `cargo deny check` (`advisories ok, bans ok, licenses ok, sources ok`), `cargo build`, `cargo fmt` and `cargo clippy --all-targets --all-features`. It then hit the 595 s cap (exit 124) while `cargo check --all-targets` was compiling, so `cargo test --lib --tests`, `cargo doc` and the release build did not run.

Not done / risk: the CI `version-increment` job is unchanged (see above). The last gate stages (full `cargo test`, `cargo doc`, release build) were not run here.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 595 s foreground cap (exit 124) during cargo check; every stage before it passed; the new tests, fmt and clippy were run separately; CI runs the full gate" -->
