## Summary

Quiet the coverage workflow's Cargo progress with `CARGO_TERM_QUIET=true`, while retaining the test failure name, assertion and backtrace. The regression check now covers multiline workflow commands and rejects missing commands. The remaining `ci.yml` call sites are unchanged because no explicit approval to edit that protected file was found. Part of #2555.

Version bump: `Cargo.toml` updated for this change; the lockfiles record the crate version.
The Develop merge retains `0.74.363` in `fuzz/Cargo.lock` to match the merged `Cargo.toml`.

## Spec

### Intent and Rationale

- The coverage run still printed compiler progress after its test reporter was quieted. `CARGO_TERM_QUIET=true` quiets Cargo and the coverage test reporter together.

### Essential Design Decisions

- The explicit `--quiet` after `--` is removed because combining it with `CARGO_TERM_QUIET=true` makes the test harness reject the duplicated option.
- The coverage test thread limit and feature selection stay in place. `ci.yml` remains protected by `AGENTS.md` until explicit approval is recorded.

### Undiscoverable Facts

- A deliberately failing local coverage test with `CARGO_TERM_QUIET=true` exited 101 and printed its test name, assertion message and backtrace. The temporary probe file was removed.

## Evidence

- Before the change, `cargo llvm-cov --test issue_2555_quiet_quality_gate -- --test-threads=2 --quiet` printed `Compiling` progress. With `CARGO_TERM_QUIET=true`, the passing coverage run printed no compile progress and retained its coverage report.
- `CARGO_TERM_QUIET=true RUST_BACKTRACE=1 cargo llvm-cov --test issue_2555_failure_probe -- --test-threads=2` — expected failure, exit 101; the test name, assertion message and stack trace remained visible.
- A temporary FFI probe called `neat_ai_discovery::ffi::get_library_version()`, freed its result with `free_discovery_result()`, and received version `0.74.363`; the probe was removed.
- Coverage flow:

  ```mermaid
  flowchart LR
      A[Coverage workflow] --> B[Cargo quiet mode]
      B --> C[Quiet test reporter]
      C -->|pass| D[Coverage report]
      C -->|failure| E[Test name and diagnostics]
  ```

## Acceptance Criteria

The issue has no acceptance-criteria heading. These requested `ci.yml` call sites remain incomplete because explicit approval to edit the protected workflow was not present:

- **missing** — `.github/workflows/ci.yml:327` "Tests with `--verbose` — Remove `--verbose`; use `cargo test -q`" — reason: `AGENTS.md` requires explicit approval to modify `ci.yml`, and none was found in the issue or PR discussion.
- **missing** — `.github/workflows/ci.yml:307`, `.github/workflows/ci.yml:310` "Lint and build progress — Cargo `-q`" — reason: same approval requirement.
- **missing** — `.github/workflows/ci.yml:111` "Tool installation progress — `cargo install -q`, preserving existing pins and `--locked`" — reason: same approval requirement.
- **missing** — `.github/workflows/ci.yml:545` "pip installation progress — `--quiet`, preserving `--require-hashes`" — reason: same approval requirement.

## Test Plan

- `cargo test -q --test issue_2555_quiet_quality_gate` — passed, 11 tests.
- `env CARGO_TERM_QUIET=true cargo llvm-cov --all-features --lcov --output-path /dev/null --test issue_2555_quiet_quality_gate -- --test-threads=2` — passed, 11 tests; report generation completed.
- `workflow_parser_reads_multiline_run_blocks_and_rejects_a_missing_command` — passed; the multiline parser mutation and the missing-command diagnostic mutation each made this test fail.
- `coverage_llvm_cov_quiets_cargo_and_keeps_test_threads_two` — passed; removing `CARGO_TERM_QUIET=true` made it fail because the required coverage command was missing.
- Edited assertion in `tests/issue_2555_quiet_quality_gate.rs`: removed the requirement for libtest `--quiet` after `--`. Required by the observed duplicate-option failure when `CARGO_TERM_QUIET=true` and `--quiet` are both passed; the replacement checks the environment setting and rejects the duplicate option.
- `./quality/bash_syntax.sh .`, `./quality/shellcheck.sh .`, `cargo fmt --all --check`, and `./quality/fuzz_lockfile.sh check` — passed.
- `./quality.sh < /dev/null` — timed out at its 900-second command limit during the release-library stage (exit 124, no diagnostic). The test suites and documentation stage completed before it reached that stage; `timeout 1800 cargo build -q --release --lib < /dev/null` then passed.
- GitHub Actions checks for this PR are pending.

## Not closing #2555

This PR leaves #2555 open because the four requested `ci.yml` command groups remain unchanged under the repository's explicit approval requirement.
