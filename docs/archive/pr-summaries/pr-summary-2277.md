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

This repository has no `CODING-STANDARDS.md`, so the reviewer used `CONTRIBUTING.md` and `AGENTS.md`.

- **violation** — Stale `:NN` citations in the section's "Clean" bullets (`docs/audits/security-sweep-chunk-16-build-scripts.md`) — reason: fixed with a line-shift note in the #2126 bullet. The bullets themselves are not #2126 entries, and the issue forbids changing other rows
- **violation** — The heading (140) and the Files swept row (126) disagree — reason: not changed. The issue asks for the heading to read 140 lines, and the Files swept table records line counts "as at the baseline commit"
- **violation** — No valid-boundary or regex-edge tests — reason: fixed with `::smallest_valid_max_attempts_is_accepted` and `::off_pattern_integers_are_rejected`
- **note** — The rejected raw value is echoed in the `::error::` line — reason: not changed. The issue specifies that format, the existing `validate_name` does the same, and the value comes from the workflow's own environment
- **clean** — Anchored `=~` allowlist before any arithmetic, `sleep` or `rustup` call; bash 3.2+ safe under `set -euo pipefail`; fails loud with exit 2; tests drive the real script through a stub (no source grepping, no wall-clock sleeps); Australian English; `ci.yml` untouched

## Test Plan

- [x] Red: 4 new tests failed on the unfixed script (the hostile test got exit 0).
- [x] Green: `cargo test --test issue_1891_rust_toolchain_install`, 21 passed.
- [x] `quality/bash_syntax.sh` and `quality/shellcheck.sh` OK.
- [x] Chunk 16 ledger #2126 entries updated.
- [ ] `./quality.sh < /dev/null` on the final tree.
