## Summary

`scripts/install-rust-toolchain.sh` now validates `RUST_TOOLCHAIN_MAX_ATTEMPTS` (`^[1-9][0-9]*$`) and `RUST_TOOLCHAIN_RETRY_DELAY` (`^[0-9]+$`) with `[[ =~ ]]` straight after reading them. The check runs after `--help` and before `validate_name`, the `rustup` lookup and every `rustup` call. A bad value prints `::error::install-rust-toolchain.sh: invalid <VARIABLE> value '<v>'` to stderr and exits `2`. This closes the arithmetic-context injection recorded as finding #2126: `[[ -ge ]]` evaluates its operands arithmetically, so an array subscript such as `HOME[$(cmd)]` ran `cmd`. The usage text now states both constraints. In the chunk 16 ledger, the three #2126 entries are marked **fixed** and the section heading reads 140 lines. Closes #2277.

## Evidence

This is a CLI/build-script change with no UI, so there is no screenshot.

- `cargo test --test issue_1891_rust_toolchain_install`: 21 passed (15 existing plus 6 new).
- `quality/bash_syntax.sh` and `quality/shellcheck.sh`: OK (24 scripts).
- `./quality.sh < /dev/null`: see the Test Plan.

## Reproduction

- **Symptom:** on the unfixed script (`8f5f560`, bash 5.2.37), with the first install failing:
  - `RUST_TOOLCHAIN_MAX_ATTEMPTS='HOME[$(: > <sandbox>/pwned)]'` **created the sentinel and exited `0`**. The arithmetic threw a syntax error, the loop retried, and the second install succeeded.
  - `MAX_ATTEMPTS=0`, `MAX_ATTEMPTS=abc` and `RETRY_DELAY=abc` each exited `0` after four `rustup` calls.
- **Status:** verified.
- **Regression test:** `tests/issue_1891_rust_toolchain_install.rs::hostile_max_attempts_is_rejected_and_never_executed`. Against the unfixed script it fails: it gets exit `Some(0)` where it expects `2`, and the sentinel exists. With the fix it passes.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Guard after the reads, after `--help`, before `validate_name`, the `command -v rustup` check and every `rustup` call; allowlists `^[1-9][0-9]*$` and `^[0-9]+$`, tested only with `[[ =~ ]]` — evidence: `scripts/install-rust-toolchain.sh:57-67` — reviewer: met
- **met** — Each bad value exits `2`, prints a `::error::` line naming the variable, and makes no `rustup` call — evidence: `::assert_rejected_before_rustup`, used by all rejection tests — reviewer: met
- **met** — Usage text documents a positive integer (default 3) and a non-negative integer number of seconds (default 15) — evidence: `scripts/install-rust-toolchain.sh:38-41` — reviewer: met
- **met** — `run_with_env` added, with `run` delegating to it — evidence: `tests/issue_1891_rust_toolchain_install.rs` `Sandbox::run_with_env` — reviewer: met
- **met** — Hostile test, and the PR summary records that it fails on the unfixed script — evidence: `::hostile_max_attempts_is_rejected_and_never_executed`; see Reproduction — reviewer: partial — reason: the reviewer ran before this summary existed; the pre-fix result is recorded above, and the reviewer independently reproduced "sentinel created, exit 0"
- **met** — The hostile payload uses `HOME[...]` rather than the issue's literal `x[...]` — evidence: the test comment — reviewer: met — reason: on the unfixed script, `x[$(…)]` aborts under `set -u` (`x: unbound variable`, exit 1) **before** the substitution runs, so no sentinel is created and the test would not show the injection; `HOME` is always set, so the payload really executes. The fixed script rejects both forms with exit 2
- **met** — `RETRY_DELAY=abc`, `MAX_ATTEMPTS=0` and `MAX_ATTEMPTS=abc` tests — evidence: `::non_integer_retry_delay_is_rejected`, `::zero_max_attempts_is_rejected`, `::non_integer_max_attempts_is_rejected` — reviewer: met
- **met** — Existing tests still pass — evidence: 21 passed — reviewer: met
- **met** — `quality/bash_syntax.sh` and `quality/shellcheck.sh` clean — evidence: both OK — reviewer: met
- **met** — All #2126 ledger entries marked **fixed** with `file.rs::fn_name` citations; the heading reads 140 lines; "Re-confirmed, filed before this sweep" and `## Verify this record` are unchanged — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md` — reviewer: met
- **unrequested** — The `assert_rejected_before_rustup` helper — reviewer: unrequested — reason: it removes the four-fold duplication of the exit-2, stderr and empty-log assertions
- **unrequested** — Tests `::off_pattern_integers_are_rejected` (`03`, `-1`, ` 3`, `1.5`) and `::smallest_valid_max_attempts_is_accepted` (`1` gives exactly one attempt) — reviewer: unrequested — reason: added after the standards review flagged missing regex-edge and valid-boundary coverage
- **unrequested** — A note in the #2126 bullet that the other line numbers in that section are as at `4f269d6` — reviewer: unrequested — reason: both reviewers noted the 14-line shift makes the section's `:NN` citations stale; the note says so without editing non-#2126 rows

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

This repository has no `CODING-STANDARDS.md`, so the reviewer used `CONTRIBUTING.md` and `AGENTS.md`. This block records a fresh reviewer run over `git diff 8f5f560...HEAD`. It replaces an earlier run whose entries gave no evidence. That run's "no valid-boundary or regex-edge tests" violation was fixed by `::smallest_valid_max_attempts_is_accepted` and `::off_pattern_integers_are_rejected`, and the fresh run lists both under clean.

- **violation** — CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number": the new `(:57-67)` citation in the #2126 bullet uses HEAD line numbers rather than the pinned `4f269d6` ones, and no symbol sits next to it — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:371` — reason: stands. This retry may only amend the PR summary, so the audit record was not edited. The fix is to drop `(:57-67)` or name the guard by symbol.
- **note** — The same section uses one line number for two commits: `NAME_PATTERN (:57)` at baseline sits a few lines below the new `:57-67` at HEAD — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:378` — reason: not changed. The line-shift caveat at `:376-377` is the only disambiguation.
- **note** — The shift caveat is imprecise: it names no commit for `:55`, and it leaves out the +2 shift from the widened usage text — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:376-377` — reason: not changed, for the same reason as above.
- **note** — The heading says 140 lines while the Files swept row says 126; the `:385` precedent uses the form "151 lines (164 after #2209)" — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:366` — reason: not changed. The issue asks for the heading to read 140, and the Files swept table records counts at the baseline commit.
- **note** — The findings-table row lists four of the six guard tests, leaving out `::off_pattern_integers_are_rejected` and `::smallest_valid_max_attempts_is_accepted` — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:212` — reason: not changed. Both tests exist and pass, and the row cites the tests the issue asked for.
- **note** — The version is unchanged at `0.74.264` — evidence: `Cargo.toml:3` — reason: CI's `version-increment` job bumps it in the PR flow.
- **clean** — Script (`scripts/install-rust-toolchain.sh:57-67`):
  - The guard sits straight after the reads and before `validate_name`, the `rustup` lookup, the `[[ -ge ]]` test (`:118`) and `sleep` (`:124`).
  - It fails with `::error::` and exit 2, matching `validate_name`.
  - The usage text at `:38-41` is updated.
  - `bash -n` and `shellcheck` pass.
- **clean** — Tests (`tests/issue_1891_rust_toolchain_install.rs`):
  - They drive the real script and check exit code, stderr, the rustup log and the sentinel.
  - The hostile test has a positive precondition (first install fails).
  - The rejection tests use a zero-failure stub, so the empty-log assertion is not vacuous.
  - The boundary test proves the guard does not over-reject.
  - There are no timing assertions and no global state.
  - `run` delegates to `run_with_env`.
  - `rustfmt --check` passes, and all 21 tests pass.
- **clean** — Australian English; no Mermaid; `ci.yml` untouched; no dependency changes; no over-engineering.

## Test Plan

- [x] Red: 4 new tests failed on the unfixed script (the hostile test got exit 0).
- [x] Green: `cargo test --test issue_1891_rust_toolchain_install`, 21 passed.
- [x] `quality/bash_syntax.sh` and `quality/shellcheck.sh` OK.
- [x] Chunk 16 ledger #2126 entries updated.
- [ ] `./quality.sh < /dev/null` on the final tree.
