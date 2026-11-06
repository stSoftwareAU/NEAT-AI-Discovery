## Summary

Successful quality-gate stages are now quiet while failures keep their full diagnostics. `quality.sh` runs cargo build, clippy, check, test, doc and the release build with `-q`; the coverage job passes `--quiet` to the libtest harness; `cargo install` in `security.yml` and `npm install` in `markdown-lint.yml` and `renovate-validate.yml` are quiet with their pins, `--locked` and `--ignore-scripts` kept. The `.github/workflows/ci.yml` call sites are deliberately not changed (see below), so this is `Part of #2555`.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- Per-test success lines and compiler progress bury actionable failures; `-q` / `--quiet` / `--loglevel=error` drop only the success chatter, so each stage keeps its one-line `echo` header.
- A new structural test parses the non-comment `run:` lines (and the `quality.sh` commands) so the quiet flags cannot silently regress.

### Essential Design Decisions

- `.github/workflows/ci.yml` is left unchanged: `AGENTS.md` ("Do Not Modify CI Without Approval") forbids editing it without explicit approval, and the issue itself says it proposes no automatic change. `CONTRIBUTING.md` records that ci.yml still differs.
- `cargo fmt` and `cargo deny` in `quality.sh` are unchanged: neither is a call site the issue lists.
- Two existing tests that pinned the old command strings (`issue_1657`, `issue_2390`) were updated to the quiet commands, keeping their guard intent.

### Undiscoverable Facts

- None.

## Evidence

- Deliberate-failure check: a temporary failing test run under `cargo test -q` still printed the test name, the assertion message and the backtrace (with `RUST_BACKTRACE` set). The temporary file was deleted before the commit.
- Gate stages run on the head, one stage at a time because a single foreground call is capped at 10 minutes: `cargo fmt --check`, `shellcheck quality.sh` and `bash -n quality.sh`, `cargo clippy -q --all-targets --all-features -- -D warnings`, `RUSTDOCFLAGS="-D warnings" cargo doc -q --no-deps`, `cargo build -q --release --lib`, and `cargo test -q --tests --all-features -- --test-threads=2` (320 suites, 6779 passed, 0 failed, including the 1852 library tests).
- The remaining `quality.sh` stages (cargo deny, fuzz lockfile sync, PR summary location, cargo install pinning gate) were not re-run as a single `./quality.sh` invocation; the pinning gate is covered by `tests/issue_1912_cargo_install_pinning_gate.rs`, which passed.

## Acceptance Criteria

The issue has no acceptance-criteria heading. The call sites it lists that this PR leaves undone are recorded as `missing`:

- **missing** — `.github/workflows/ci.yml:327` "Remove `--verbose`; use `cargo test -q`" — reason: ci.yml may not be edited without explicit approval (`AGENTS.md`)
- **missing** — `.github/workflows/ci.yml:307`, `.github/workflows/ci.yml:310` "Lint and build progress — Cargo `-q`" — reason: same approval requirement
- **missing** — `.github/workflows/ci.yml:111` "`cargo install -q`, preserving existing pins and `--locked`" — reason: same approval requirement
- **missing** — `.github/workflows/ci.yml:545` "pip installation progress — `--quiet`, preserving `--require-hashes`" — reason: same approval requirement

## Test Plan

- Added `tests/issue_2555_quiet_quality_gate.rs` (10 tests: quality.sh commands quiet, `--test-threads=2` kept, coverage `--quiet` after the `--`, security.yml installs quiet/locked/pinned, npm installs quiet with `--ignore-scripts`, plus negative controls for each detector and a commented-out-command parser test). Result: 10 passed.
- Red checks (each break applied alone, test run, then restored): removing `-q` from the `quality.sh` test command → `quality_sh_cargo_build_clippy_check_test_and_doc_run_quiet` red; removing `--quiet` from the llvm-cov line → `coverage_llvm_cov_passes_quiet_after_the_separator` red; removing `-q` from `cargo install` in security.yml → `security_workflow_cargo_installs_are_quiet_locked_and_pinned` red; removing `--loglevel=error` from markdown-lint.yml → `npm_installs_are_quiet_and_skip_lifecycle_scripts` red.
- Edited `tests/issue_1657_workflow_dedup_audit.rs` and `tests/issue_2390_coverage_workflow_guards.rs`: the pinned command strings now include `-q` / `--quiet`. No assertion removed; both passed.
- Twinned copies: the quiet commands appear in `quality.sh`, the workflows, `CONTRIBUTING.md` and `docs/ci-doc-build-step.md`; all were updated and the doc citations (`quality.sh:75-76`) re-pointed.
- Not checked: the CI behaviour of the workflow edits (runs on the PR); `ci.yml` call sites (not changed).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
