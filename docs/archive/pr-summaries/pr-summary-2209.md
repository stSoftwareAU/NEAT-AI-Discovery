# PR summary — Issue #2209

## Summary

Closes the #2127 finding for `scripts/install-rustup.sh` (part of #2127).
Closes #2209.

- **Function-name trap.** `install_rustup` used to arm
  `trap "rm -rf '$tmp_dir'" EXIT`. That expanded the `mktemp -d` path into a
  string, and bash re-parses it when the trap fires. Because `mktemp -d`
  honours a caller-controlled `TMPDIR`, a quote-bearing `TMPDIR` could run
  arbitrary commands on every exit path. The path now lives in the
  script-level `_INSTALL_TMP_DIR`, declared empty so `set -u` is satisfied
  after `install_rustup` returns. It is reaped by `_cleanup_tmp`, which runs
  `rm -rf --` only when the variable is non-empty and a directory, under
  `trap _cleanup_tmp EXIT`.
- **INT and TERM** are the fixed literals `trap 'exit 130' INT` and
  `trap 'exit 143' TERM`. `exit` fires the `EXIT` cleanup, so an interrupted
  run ends non-zero instead of resuming past a deleted directory. No trap
  argument expands anything, and the `# shellcheck disable=SC2064` line is
  gone.
- **Digest path unchanged.** `_pinned_digest`, `_sha256_of` and the mismatch
  refusal are untouched. `scripts/runlib.sh` is untouched.
- **Ledger.** `docs/audits/security-sweep-chunk-16-build-scripts.md` flips
  the #2127 Findings row to **fixed** and names the guarding tests. The
  file-coverage row and the `### scripts/install-rustup.sh` section are
  updated. They follow the ledger's baseline convention: the fix is cited by
  symbol, the coverage row keeps the `4f269d6` count (151), the section heading
  reads "151 lines (164 after #2209)", and a note records how far the fix
  shifted the baseline `#1911` line citations.

## Evidence

- **Fails against the unfixed script.** The hostile-`TMPDIR` test was run
  against the base script (`e670001:scripts/install-rustup.sh`, which still
  carries `trap "rm -rf '$tmp_dir'" EXIT`) in a scratch worktree.
  `a_quote_bearing_tmpdir_executes_nothing_and_is_reaped` **FAILED** at the
  sentinel assertion (`tests/issue_1911_rustup_digest_verification.rs:351`,
  "a quote-bearing TMPDIR must never execute a command"). The injected
  command ran. Against the fixed script it passes.
- `cargo test --test issue_1911_rustup_digest_verification` gives 12 passed:
  the 10 existing tests plus 2 new ones.
- `cargo test --test issue_2097_rustup_pin_parity` gives 5 passed.
- `grep -n SC2064 scripts/install-rustup.sh` returns nothing.
  `quality/shellcheck.sh` and `quality/bash_syntax.sh` both report
  "OK — 24 script(s) passed".
- `./quality.sh < /dev/null` exited 0 ("✅ All quality checks passed!") on
  this branch after the final code and ledger changes.

## Reproduction

- **symptom** — a `TMPDIR` containing `'; touch <path>; '` made `install-rustup.sh` run the embedded `touch` when its `EXIT` trap re-parsed the interpolated `mktemp -d` path
- **status** — `verified` — the regression test was observed failing against the unfixed code (the base script restored in place: 11 passed, 1 failed, panic at `tests/issue_1911_rustup_digest_verification.rs:351`, "a quote-bearing TMPDIR must never execute a command") and passing after the fix (12 passed)
- **regression test** — `tests/issue_1911_rustup_digest_verification.rs::a_quote_bearing_tmpdir_executes_nothing_and_is_reaped`

## Test Plan

`tests/issue_1911_rustup_digest_verification.rs`, extended with every
existing test kept:

- `Sandbox::run` takes an optional `TMPDIR` and pins `current_dir` to the
  sandbox root, so an injected relative command cannot write into the repo.
  Existing tests pass `None`.
- `a_quote_bearing_tmpdir_executes_nothing_and_is_reaped` (new) uses a
  `TMPDIR` containing `'; touch <sandbox>/pwned; '` and a failing stub `curl`.
  It asserts a non-zero exit, no sentinel, an empty `TMPDIR` and nothing
  executed.
- `executes_the_installer_when_the_digest_matches` (extended) asserts the
  plain `TMPDIR` is empty after a successful run.
- `a_term_during_the_download_exits_non_zero_and_reaps_the_temp_dir` (new)
  uses a stub `curl` that runs `kill -TERM "$PPID"` and then completes the
  download. It asserts a non-zero exit, the installer never executed and an
  empty `TMPDIR`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `scripts/install-rustup.sh` arms cleanup with `trap _cleanup_tmp EXIT` (a function name), and its INT/TERM traps are fixed literals — evidence: `scripts/install-rustup.sh::install_rustup` (`trap _cleanup_tmp EXIT`, `trap 'exit 130' INT`, `trap 'exit 143' TERM`), `::_cleanup_tmp` — reviewer: met
- **met** — `grep -n SC2064 scripts/install-rustup.sh` returns nothing, and `quality/shellcheck.sh` and `quality/bash_syntax.sh` pass — evidence: grep prints nothing; both gates report "OK — 24 script(s) passed" — reviewer: met
- **met** — The hostile-`TMPDIR` test asserts a non-zero exit, an absent sentinel and a reaped mktemp directory, and the PR summary states that it fails against the unfixed script — evidence: `tests/issue_1911_rustup_digest_verification.rs::a_quote_bearing_tmpdir_executes_nothing_and_is_reaped`; the unfixed-script failure is recorded under **Evidence** and **Reproduction** — reviewer: met
- **met** — The happy-path test asserts that the `TMPDIR` directory is empty after a successful run — evidence: `tests/issue_1911_rustup_digest_verification.rs::executes_the_installer_when_the_digest_matches` — reviewer: met
- **met** — The interrupt test asserts that a TERM during the download exits non-zero, never executes the installer, and reaps the temp directory — evidence: `tests/issue_1911_rustup_digest_verification.rs::a_term_during_the_download_exits_non_zero_and_reaps_the_temp_dir` — reviewer: met
- **met** — Every existing test in `tests/issue_1911_rustup_digest_verification.rs` and `tests/issue_2097_rustup_pin_parity.rs` still passes — evidence: 12/12 and 5/5 passed — reviewer: met
- **met** — The chunk-16 ledger records #2127 as **fixed** and names the guarding test(s) — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md` #2127 Findings row, coverage row and `### scripts/install-rustup.sh` section — reviewer: met
- **met** — `./quality.sh` passes — evidence: `./quality.sh < /dev/null` exited 0 ("✅ All quality checks passed!") on this branch after the ledger and test changes — reviewer: missing — reason: the reviewer said "unverified" because it did not run the full gate; it ran the shellcheck, bash-syntax and both test sub-gates, which passed, and the full gate was run here
- **unrequested** — a note in the ledger's `### scripts/install-rustup.sh` section recording how far the fix shifted the baseline `#1911` line citations — reviewer: unrequested — reason: the reviewer judged it not asked for but in line with the ledger's own baseline-citation rule; without it those citations read as current lines

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — no violations. The reviewer used `CONTRIBUTING.md` and `AGENTS.md`; the repository has no `CODING-STANDARDS.md`. It checked the function-name `EXIT` trap, the literal INT/TERM traps and the guarded `rm -rf --`, and confirmed shellcheck passes with the `SC2064` disable gone. It re-ran the hostile `TMPDIR` against the baseline script and saw the sentinel created. It confirmed `current_dir` is pinned to the sandbox. The ledger cites baseline lines alongside symbols and keeps the baseline 151-line count. Australian English is used throughout, and `ci.yml` and the dependencies are untouched. Optional notes: the ledger shift range should start at `:33`, not `:32` (fixed in this diff); the heading's "164 after #2209" is a HEAD-relative count, kept because the issue asked for the line count to be updated; the version bump is left to CI's `version-increment` job; the INT/TERM traps are global when the script is sourced, which is harmless because nothing else sources it.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
