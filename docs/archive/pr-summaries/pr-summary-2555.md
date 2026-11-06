## Summary

Quiet the remaining CI test, lint, build and tool-install commands, retaining failure diagnostics and existing install safeguards. The coverage workflow also sets `CARGO_TERM_QUIET=true` to suppress Cargo progress and quiet its test reporter. Addresses #2555.

The current instruction explicitly authorises the four requested command groups in the protected `ci.yml`. Its triggers, job structure and PAT handling are preserved. `CONTRIBUTING.md` now describes the resulting commands. The crate version is `0.74.364`, synchronised in both lockfiles without dependency changes.

## Spec

### Intent and Rationale

- Successful compiler, test and installation progress obscured failures. Quiet flags suppress that progress while preserving test names, assertion messages, backtraces and command exit status.
- The existing quiet settings in `quality.sh`, security installs and npm installs already satisfy their issue requirements on the base branch and remain covered by regression tests.

### Essential Design Decisions

- CI tests use `cargo test -q` without `--verbose`, retaining all target and feature selections and `--test-threads=2`.
- Clippy and the library build gain `-q`; lint denials and build scope remain intact.
- `cargo install -q` retains the cargo-outdated version pin and `--locked`. Pip gains `--quiet`, retaining `--user`, `--require-hashes` and the requirements file.
- Coverage uses `CARGO_TERM_QUIET=true`; a second libtest `--quiet` after `--` is omitted because the observed combination makes the harness reject the duplicated option.

### Undiscoverable Facts

- An isolated temporary Cargo crate ran the exact quiet CI test command with `RUST_BACKTRACE=1` and a deliberate assertion failure. It exited 101 and retained its test name, assertion message and stack trace, without Cargo compilation or execution progress. The temporary crate was removed.
- Earlier in this run, a deliberate coverage failure with `CARGO_TERM_QUIET=true` also retained its test name, assertion and backtrace. Its temporary probe was removed.

## Evidence

- The executor added four CI regression tests first: all four failed on the previous workflow, while the existing 11 tests passed. After the five command-line edits, all 15 focused tests passed.
- The Spec reviewer independently compiled and ran the focused suite with stdin from `/dev/null`: 15 passed. It reviewed the failure-probe output and agreed that all four previously unmet criteria are met.
- A temporary FFI probe loaded the built library, received `success: true` and version `0.74.364` from `get_library_version()`, and freed the returned pointer with `free_discovery_result()`.
- The Standards reviewer found the implementation compliant with `AGENTS.md` and `CONTRIBUTING.md`, including the authorised CI scope, unchanged credential handling, preserved pins and hash checks, and version consistency. This repository has no `CODING-STANDARDS.md`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `.github/workflows/ci.yml:327` "Tests with `--verbose` — Remove `--verbose`; use `cargo test -q`" — evidence: `tests/issue_2555_quiet_quality_gate.rs::ci_tests_are_quiet_and_keep_targets_features_and_thread_limit` — reviewer: met
- **met** — `.github/workflows/ci.yml:307`, `.github/workflows/ci.yml:310` "Lint and build progress — Cargo `-q`" — evidence: `tests/issue_2555_quiet_quality_gate.rs::ci_lint_and_build_are_quiet_and_keep_existing_checks` — reviewer: met
- **met** — `.github/workflows/ci.yml:111` "Tool installation progress — `cargo install -q`, preserving existing pins and `--locked`" — evidence: `tests/issue_2555_quiet_quality_gate.rs::ci_cargo_install_is_quiet_locked_and_keeps_the_existing_pin` — reviewer: met
- **met** — `.github/workflows/ci.yml:545` "pip installation progress — `--quiet`, preserving `--require-hashes`" — evidence: `tests/issue_2555_quiet_quality_gate.rs::ci_pip_install_is_quiet_and_keeps_hash_checked_requirements` — reviewer: met
- **met** — Coverage tests use a quiet reporter and suppress Cargo progress while retaining feature selection and the thread limit — evidence: `tests/issue_2555_quiet_quality_gate.rs::coverage_llvm_cov_quiets_cargo_and_keeps_test_threads_two` and the recorded coverage failure probe — reviewer: met
- **met** — Local quality-gate test, build, lint, check, documentation and release commands use Cargo `-q` (already present on base) — evidence: `tests/issue_2555_quiet_quality_gate.rs::quality_sh_cargo_build_clippy_check_test_and_doc_run_quiet` — reviewer: met
- **met** — Security tool installations are quiet, locked and pinned (already present on base) — evidence: `tests/issue_2555_quiet_quality_gate.rs::security_workflow_cargo_installs_are_quiet_locked_and_pinned` — reviewer: met
- **met** — npm installations use `--loglevel=error`, retaining pins and `--ignore-scripts` (already present on base) — evidence: `tests/issue_2555_quiet_quality_gate.rs::npm_installs_are_quiet_and_skip_lifecycle_scripts` and unchanged workflow pins — reviewer: met
- **met** — Retain stage summaries and failure test names, assertion messages and stack traces — evidence: existing quality-gate stage headers and the deliberate quiet CI test failure probe, expected exit 101 — reviewer: met
- **met** — Follow the `AGENTS.md` approval convention for `ci.yml` edits — evidence: the current user instruction explicitly authorises implementing all four CI criteria — reviewer: met

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Checked against `AGENTS.md` and `CONTRIBUTING.md`: authorised CI scope, preserved credentials and install safeguards, Australian English, regression coverage, contributor documentation, version bump and lockfile consistency.

## Test Plan

- `cargo test -q --all-features --test issue_2555_quiet_quality_gate --test issue_1342_workflow_dedup_cargo_check --test issue_1658_remove_redundant_cargo_check --test issue_1992_ci_doc_accuracy < /dev/null` — passed, 28 tests.
- Four new CI regression tests check quiet flags, required command presence, test selections and thread limit, lint denials, the existing cargo-outdated pin and lock flag, and the hash-checked pip requirements path. All four were observed failing before the workflow edits.
- Adapted existing assertions in the #1342, #1658 and #1992 suites to inspect required command arguments while accepting quiet options; their compile-scope and documentation checks remain in place.
- Earlier in this run: the 11-test focused coverage suite passed with `CARGO_TERM_QUIET=true`, retaining coverage report generation. Parser and coverage-environment mutation checks failed as expected before restoration.
- Temporary diagnostic probe: `RUST_BACKTRACE=1 cargo test -q --lib --tests --bins --all-features -- --test-threads=2 < /dev/null` — expected exit 101; failing name, assertion and backtrace visible, `Compiling` and `Running` progress absent.
- `cargo fmt --all < /dev/null`, `git diff --check`, and `actionlint .github/workflows/ci.yml .github/workflows/cargo-quality.yml < /dev/null` — passed.
- `./quality.sh < /dev/null` — passed on the final implementation, including the dependency audit, build, formatting, lint, type checks, 320 test suites (6,799 passed, 0 failed, 5 ignored), documentation and release library build.
- GitHub Actions execution remains for the worker's pull request.
