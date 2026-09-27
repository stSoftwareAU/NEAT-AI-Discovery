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
  updated.

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
- `./quality.sh` passed on this branch in the worker's quality gate before the
  PR-summary step.

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
- **met** — `grep -n SC2064 scripts/install-rustup.sh` returns nothing, and `quality/shellcheck.sh` and `quality/bash_syntax.sh` pass — evidence: grep exits 1 with no output; both gates report "OK — 24 script(s) passed" — reviewer: met
- **met** — The hostile-`TMPDIR` test asserts a non-zero exit, an absent sentinel and a reaped mktemp directory, and the PR summary states that it fails against the unfixed script — evidence: `tests/issue_1911_rustup_digest_verification.rs::a_quote_bearing_tmpdir_executes_nothing_and_is_reaped`; the unfixed-script failure is recorded under **Evidence** above — reviewer: partial — reason: the reviewer found the assertions complete and reproduced the sentinel against the base script by hand; it marked partial only because this summary did not exist yet when it reviewed
- **met** — The happy-path test asserts that the `TMPDIR` directory is empty after a successful run — evidence: `tests/issue_1911_rustup_digest_verification.rs::executes_the_installer_when_the_digest_matches` — reviewer: met
- **met** — The interrupt test asserts that a TERM during the download exits non-zero, never executes the installer, and reaps the temp directory — evidence: `tests/issue_1911_rustup_digest_verification.rs::a_term_during_the_download_exits_non_zero_and_reaps_the_temp_dir` — reviewer: met
- **met** — Every existing test in `tests/issue_1911_rustup_digest_verification.rs` and `tests/issue_2097_rustup_pin_parity.rs` still passes — evidence: 12/12 and 5/5 passed — reviewer: met
- **met** — The chunk-16 ledger records #2127 as **fixed** and names the guarding test(s) — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md` #2127 Findings row, coverage row and `### scripts/install-rustup.sh` section — reviewer: met
- **met** — `./quality.sh` passes — evidence: the worker's quality gate passed on this branch before the PR-summary step — reviewer: partial — reason: the reviewer did not run the full `./quality.sh` itself (too long); it ran the shellcheck and bash-syntax sub-gates and both test targets, and all passed
- **unrequested** — WIP checkpoint `44a6570` removed the ledger caveat that the older #1911 line citations are at `4f269d6` and shift by 13 lines after `:31` — reviewer: unrequested — reason: an automated snapshot, not part of the issue; without the caveat, the #1911 citations at ledger `:103`, `:153`, `:163`, `:169` and `:175` read as current-file lines and are now stale. This retry is limited to the summary file, so it is left for a follow-up.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — the ledger's baseline line-count rule and single source of truth — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:56` and `:382` now say 164 lines, but the table header (`:49`) says counts are taken at the baseline commit (151 at `4f269d6`, inside the 2,106-line total) — reason: stands. The issue explicitly asked to update "its line count", and this retry is limited to the summary file. The #2139/#2140 fixes kept their baseline counts, so a follow-up should restore 151.
- **violation** — CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number" (Issue #1942) — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md:56`, `:213`, `:387-390` cite current-file lines (`:34`, `:38-42`, `:126`, `:127`, `:129-130`) outside the ledger's `4f269d6` baseline exception — reason: stands, softened because every citation also names its symbol (`_INSTALL_TMP_DIR`, `_cleanup_tmp`, `install_rustup`) and all of them match the current script. This retry is limited to the summary file, so dropping the line numbers is left for a follow-up.
- **clean** — Australian English in all added comments and prose. The script passes `shellcheck` and `bash -n`, the `rm -rf --` is quoted and guarded, and the comments are short and explain why. The tests check outcomes, live in `tests/` and set `TMPDIR` on the child `Command`, so no `#[serial]` is needed. They pass `rustfmt --check` and `cargo clippy -D warnings`. The change is scoped with no over-engineering, `.github/workflows/ci.yml` is untouched, and the version bump is left to CI's `version-increment` job.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
