# Security sweep chunk 16 — build/install shell scripts

## Summary

Read all 2,106 lines of the eleven chunk-16 scripts at `4f269d6` and recorded
the outcome in `docs/audits/security-sweep-chunk-16-build-scripts.md`:
per-file verdicts, the strict-mode/gate/download/checksum table, lint coverage,
the negative checks, the re-verified #1911/#1912/#1913 remediations, and the
findings table. Added `tests/issue_2097_rustup_pin_parity.rs`, the CI assertion
behind the #1911 re-verification, and fixed a null-record panic in the ledger
contract test that the new index entry exposes. `scripts/runlib.sh` is not
touched — it is byte-synced from NEAT-AI-core. Closes #2128.

Findings filed by this sweep: **#2139** and **#2140** (security, `severity:low`,
`confidence:high`, `lang:general`), plus **#2141** as a plain portability bug.
#2126 and #2127 were re-confirmed at this baseline and linked in the ledger.

## Evidence

### The pin-parity test fails when a digest in either pin file is altered

`scripts/rustup-init.sha256` and `scripts/runlib.sh` carry the same six rustup
digests and the same release; both files say in prose that they must move
together, and nothing asserted it. Each mutation below was applied to the
working tree, the test run, and the tree restored:

| mutation | result |
| --- | --- |
| one digest altered in `scripts/rustup-init.sha256` | `every_digest_matches_between_the_two_pin_sets` FAILED (4 passed, 1 failed) |
| one digest altered in `scripts/runlib.sh` | `every_digest_matches_between_the_two_pin_sets` FAILED (4 passed, 1 failed) |
| `RUSTUP_VERSION` bumped in `scripts/install-rustup.sh` only | `the_two_rustup_version_pins_are_equal` FAILED (4 passed, 1 failed) |
| unmutated tree | 5 passed |

The tests source each script and call its **own** pin reader
(`install-rustup.sh::_pinned_digest`,
`runlib.sh::_runlib_pinned_rustup_digest`) rather than parsing source text, so
an upstream rewrite of the `case` statement does not red-light a gate in a repo
that is forbidden to edit that file. No network; no digest is recomputed —
provenance for the pins is documentary (the upstream-published `.sha256`
values), so re-downloading to "confirm" one would verify the artefact against
itself.

### The #2139 fail-silent was reproduced, not inferred

`scripts/check-pr-summary-location.sh` copied into a tree with no `docs/`:

```text
--- no docs/ present:
✅ All pr-summary-*.md files live in docs/archive/pr-summaries/
exit=0
--- with 2>/dev/null removed:
find: ‘docs’: No such file or directory
✅ All pr-summary-*.md files live in docs/archive/pr-summaries/
exit=0
```

`set -euo pipefail` does not check a process substitution's status, so a `find`
that never ran is indistinguishable from one that found nothing — and this is a
gate, run from `quality.sh`.

### Screenshots

Not applicable: this change adds a documentation record and a `cargo test`
target. There is no web interface. What was tested instead is the test run and
the three mutations above, plus `./quality.sh`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Ledger at the exact path with eleven non-`pending` rows, both SHAs, sweep date, the complete script table with `file:line`, the lint-coverage section citing both gates' `find` predicates and both call sites, the three-row re-verified table, and the findings table listing #2126/#2127 plus those filed here — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md` — reviewer: met
- **met** — `tests/issue_2097_rustup_pin_parity.rs` exists and passes; the PR summary states it fails when a digest in either pin file is altered — evidence: `tests/issue_2097_rustup_pin_parity.rs::every_digest_matches_between_the_two_pin_sets`, and the mutation table above — reviewer: partial — reason: the reviewer saw a diff with no PR summary in it and marked the second half unwritten; this summary now carries the statement and the mutation evidence
- **met** — Each of the four candidates has an explicit written verdict — evidence: `check-pr-summary-location.sh` → #2139, `benchmark.sh:32` → #2140 (both with `security`, `lang:general`, `severity:low`, `confidence:high`); `fuzz-ci.sh:15` and the `runlib.sh` staging names `accepted` with reasons in the ledger's per-file detail — reviewer: met
- **met** — No edit to `scripts/runlib.sh` — evidence: the branch diff touches `docs/audits/*`, `docs/archive/pr-summaries/pr-summary-2128.md` and two files under `tests/` only — reviewer: met
- **met** — Comment on #2097 links the ledger and every finding — evidence: the chunk-outcome comment posted on #2097 in this run — reviewer: missing — reason: the reviewer read #2097 before the comment was posted; it was posted after the review returned, and #2097 was not closed by this run
- **met** — `./quality.sh` passes — evidence: full gate run after the final edit — reviewer: missing — reason: the reviewer found `clippy::collapsible_if` and a rustfmt diff in the first draft of the test; the test was then rewritten behaviourally, and `cargo clippy -- -D warnings` and the full gate now pass
- **unrequested** — `tests/issue_2088_sweep_ledger_contract.rs` record-lookup made null-safe — reviewer: unrequested — reason: `string_field` panics on an unswept entry's `record: null`, and chunk 16 is the first record whose index entry sits after null entries, so the pre-existing scan panicked as soon as the chunk-16 record was added; without it the ledger cannot land
- **unrequested** — `docs/audits/lib-sweep-coverage.json` chunk-16 entry filled in — reviewer: unrequested — reason: `docs/audits/README.md` requires the prose record and the index entry to move together, and `tests/issue_2088_sweep_ledger_contract.rs` fails without it
- **unrequested** — issue #2141 filed for the macOS `date +%s.%N` bug — reviewer: unrequested — reason: the sub-issue authorised filing it if no open issue existed; a `gh issue list` search found none

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — `CONTRIBUTING.md` "Cite Code by Symbol, Never by Line Number" (Issue #1942): the ledger cites `file:line` throughout — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md` (script table, re-verified table, per-file detail) — reason: **stands, deliberately**, and is now declared in a *Citation convention* section at the top of the record. The sub-issue makes `file:line` an acceptance criterion because a `set -euo pipefail` line and a bare `curl` have no enclosing symbol; and unlike a live doc, a sweep record is pinned to a baseline SHA and ships the `git diff` that falsifies it, so a citation here is a claim about one commit rather than about HEAD. Mitigated in this diff by naming the enclosing symbol alongside the line wherever one exists, with the `runlib.sh` citations — the ones that can move with no edit here — called out as the ones to re-derive from the symbol first.
- **violation** — markdownlint `MD018/no-missing-space-atx` — evidence: `docs/audits/security-sweep-chunk-16-build-scripts.md` "Verify this record" — reason: **fixed here**; the line began `#2126`, which markdownlint read as an ATX heading. Reworded to `Issues #2126, …`. `npx markdownlint-cli2` now reports no issue.
- **violation** — testing doctrine: the first draft of `tests/issue_2097_rustup_pin_parity.rs` hand-parsed the `case` arms of `_runlib_pinned_rustup_digest` and the `NAME="value"` spelling of the version pins — evidence: `tests/issue_2097_rustup_pin_parity.rs` — reason: **fixed here**; rewritten to source each script and call its own reader (`_pinned_digest`, `_runlib_pinned_rustup_digest`) and to read the version variables from the sourced shell, matching `tests/issue_1911_rustup_digest_verification.rs`. A `readonly` prefix, single quotes, or an associative array upstream now all still work.
- **violation** — DRY: the first draft re-implemented the `scripts/rustup-init.sha256` digest parser that `tests/issue_1911_rustup_digest_verification.rs::committed_pins` already has — evidence: `tests/issue_2097_rustup_pin_parity.rs` — reason: **fixed here**; the rewrite gets digests from `_pinned_digest` instead, so no second digest parser exists. Only the manifest's *target column* is read, by `manifest_targets`, and solely to keep `SUPPORTED_TARGETS` honest — a target added to the manifest without updating this test now fails `the_committed_manifest_declares_exactly_the_supported_targets`.
- **violation** — no PR summary file on the branch — evidence: the branch before this commit — reason: **fixed here**; this file.
- **clean** — Australian English throughout (artefact, sanitiser, canonicalised, behaviour); ledger placement and every field `docs/audits/README.md` requires; one file per chunk, no shared append; `lib-sweep-coverage.json` entry on one line with the mandated key order and no half-claim; the vacuity guard on both directional comparisons; no `Instant`/`elapsed` timing in tests; `.github/workflows/ci.yml` untouched; `scripts/runlib.sh` read but never written; no `cargo upgrade`/`cargo update` reintroduced; `Cargo.toml` version left to CI's `version-increment` job, as the three preceding sweep commits did.

## Security-Fix Evidence

- **Test file added in this branch:** `tests/issue_2097_rustup_pin_parity.rs`.
- **Regression test identifier:**
  `tests/issue_2097_rustup_pin_parity.rs::every_digest_matches_between_the_two_pin_sets`
  — declared in this branch's added lines.
- **Red/green linkage:** the flaw this closes is the *absence* of any check that
  the two rustup pin sets agree. Against the unfixed state — either pin file
  mutated, which is exactly what an undetected drift or a mis-copied digest
  looks like — the test fails (see the mutation table above); against the
  committed, consistent pins it passes. The companion
  `::the_two_rustup_version_pins_are_equal` covers the version half.
- **Original trigger closed, no trivial bypass:** the trigger is a digest or
  release moved on one side only. Both directions are now covered — the
  comparison runs over `SUPPORTED_TARGETS`, and
  `::the_committed_manifest_declares_exactly_the_supported_targets` fails if the
  manifest gains or loses a target without that list moving with it, so the
  comparison cannot be narrowed to pass vacuously. The digests are obtained by
  calling each script's own reader, so a bypass would have to make the script
  itself answer the wrong digest — which is the defect, not a way around the
  check. `::both_pin_sets_refuse_a_target_neither_supports` pins that a refusal
  stays a refusal, so an empty answer cannot be mistaken for agreement.

## Test Plan

- **Added** `tests/issue_2097_rustup_pin_parity.rs` — five tests:
  - `the_committed_manifest_declares_exactly_the_supported_targets`
  - `both_pin_sets_answer_for_every_supported_target`
  - `every_digest_matches_between_the_two_pin_sets`
  - `both_pin_sets_refuse_a_target_neither_supports`
  - `the_two_rustup_version_pins_are_equal`
- **Modified** `tests/issue_2088_sweep_ledger_contract.rs` — the record→entry
  lookup no longer panics on an unswept entry's `record: null`. No test was
  removed or weakened; the assertion it guards is unchanged.
- **Unchanged and still passing:** the #1911, #1912, #1913, #1918, #2072 and
  #2088 suites this sweep re-verified.
- `./quality.sh` run in full after the final edit.
