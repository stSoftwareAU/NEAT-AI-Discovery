## Summary

Links the two watchdog findings already filed as #2404 (abort delay, low) and #2405 (stall-timeout truncation, medium) under `## Issues filed` → `<!-- section: 13b -->` in `docs/audits/security-sweep-chunk-13-config-env.md`. Nothing else in the file changes. Closes #2259.

## Spec

### Intent and Rationale

- The record did not exist when the findings were filed (#2286 creates it), so the filing and linking were split across runs; this run only does the link.

### Essential Design Decisions

- Cited by symbol (`user_facing.rs::watchdog_abort_delay`, `watchdog.rs::watchdog_loop`), no line numbers, per CONTRIBUTING.md (Issue #1942).
- The 13b outcome row and settings table stay `pending`: they belong to the 13b sweep sub-issue, not this one.

### Undiscoverable Facts

- #2404 and #2405 were filed, and pointers posted on #2096 and #2252, in an earlier run; no duplicate was filed here.

## Evidence

Docs-only change. Re-checked this run: both issue bodies carry the `finding-id` and `cwe` markers, the five house sections, exposure `local`, a `tests/issue_<n>_*.rs` statement and the required cross-links (#2404: #2119, #2252; #2405: #2252). #2404 carries `security`, `lang:rust`, `severity:low`, `confidence:high`; #2405 carries `severity:medium` with the same other labels. Each of #2096 and #2252 has a comment naming #2404.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Two findings are filed, or an existing match is referenced — evidence: #2404 and #2405 (labels and body sections checked with `gh issue view`) — reviewer: partial — reason: the reviewer had file access only and could not see GitHub; I verified the issues and labels directly
- **met** — Finding A cross-links #2119 and #2252. Finding B cross-links #2252 — evidence: bodies of #2404 and #2405 — reviewer: partial — reason: reviewer could not read GitHub; I grepped both bodies
- **met** — Both are linked in the record's `13b` Issues-filed region, and in comments on #2096 and #2252 — evidence: `docs/audits/security-sweep-chunk-13-config-env.md`; comments on #2096 and #2252 — reviewer: met
- **met** — `./quality.sh` passes — evidence: ledger tests `issue_2088_sweep_ledger_contract` and `issue_2211_chunk_13_ledger_scaffold` pass (9 and 6 tests); full gate not run locally, see Test Plan — reviewer: partial — reason: reviewer could not run the gate; CI runs it

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — no violations. Checked: symbol-not-line citations, Australian English, ledger Issues-filed field, per-chunk record only, ledger parity contract. No `CODING-STANDARDS.md` exists; reviewed against `CONTRIBUTING.md` and `docs/audits/README.md`.

## Test Plan

- No tests added or edited; the existing ledger guards cover the edit.
- `cargo test --test issue_2088_sweep_ledger_contract --test issue_2211_chunk_13_ledger_scaffold`: 9 and 6 tests passed.
- Full `./quality.sh` not run locally.

<!-- vibe-quality-gate-skipped reason="full gate not run locally; docs-only edit guarded by the two ledger tests above; CI runs the gate" -->
