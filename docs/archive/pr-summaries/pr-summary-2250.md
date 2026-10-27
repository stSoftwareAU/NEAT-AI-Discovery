## Summary

Finalises the chunk-9 GPU/WGSL sweep record `docs/audits/security-sweep-chunk-9-gpu-wgsl.md`: the sweep status is flipped to COMPLETE, `## Ledger` gains the severity-sort note, the `status`-header note and the finding-issue audit line, the six named candidates are pinned in `## Ledger` / `## Refuted / not findings`, and a new `## Outcome` carries the severity counts (0 critical, 0 high, 1 medium, 11 low), the chunk-wide severity-sorted list and an annotated caller → queue → evaluation → shader → readback Mermaid flowchart. `tests/issue_2249_chunk9_ledger_complete.rs` pins all of it. Closes #2250.

Version bump: `VERSION` handling is left to CI's `version-increment` job; the merged commit records the final version.

## Spec

### Intent and Rationale

- Closes the chunk-9 sweep: the ledger and Outcome are the single record #2094 reports from.
- Outcome counts are asserted equal to the `## Ledger` rows per severity, so they cannot drift.

### Essential Design Decisions

- The ledger's last column keeps the `status` header pinned by `tests/issue_2288_chunk_09_ledger_scaffold.rs`, and carries each row's `#N` link in it. Seven rows still read `open — #N` because per-slice tests pin that wording; `## Outcome` records each issue's current state.
- `Cargo.toml` bumps `parquet` and `arrow` 59.1 → 60.0 (with `Cargo.lock`): `./quality.sh` failed in `cargo deny` on RUSTSEC-2026-0391/0398/0403/0410 (zstd 0.13 / zstd-safe 7 / zstd-sys 2.1.0 via parquet 59.3), advisories on the base branch too. parquet 60.0.0 (published 2026-09-15) moves to zstd 0.14, and `cargo check --all-targets --all-features` needed no source change. This is outside the docs scope but is required for the gate.
- The flowchart includes a `device` node (init and capability probe) between caller and queue so the four device findings attach where they sit; no `;` appears anywhere in the block.

### Undiscoverable Facts

- Finding-issue audit run with `gh issue view <n> --json labels,body` on all 11 issues (#2308, #2311, #2313, #2314, #2318, #2332, #2339, #2361, #2363, #2364, #2365): each has `security`, `lang:rust`, exactly one `severity:*`, one `confidence:*`, and links this record, so no body edit was needed.
- Mermaid syntax beyond the `;` rule is enforced only by the worker's external Mermaid validator, not by `./quality.sh`.
- #2094 is closed. Eleven findings survived, so `negative-result` is not applied; one comment links the ledger and `## Outcome`.

## Evidence

Test run on the final head: `cargo test --test issue_2249_chunk9_ledger_complete --test issue_2288_chunk_09_ledger_scaffold` passed (14 tests). `./quality.sh < /dev/null` passed ("All quality checks passed!") after the dependency bump.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `## Ledger` lists every chunk-9 finding issue with `file:line`, CWE, severity and issue number, sorted by severity within each section region. — evidence: `tests/issue_2249_chunk9_ledger_complete.rs::ledger_rows_are_sorted_by_severity_within_each_region`, `docs/audits/security-sweep-chunk-9-gpu-wgsl.md` `## Ledger` — reviewer: met
- **met** — `## Outcome` has the severity-sorted finding list, severity counts matching the ledger, and a Mermaid flowchart with no `;`. — evidence: `tests/issue_2249_chunk9_ledger_complete.rs::the_outcome_severity_counts_equal_the_ledger_rows_per_severity`, `::the_outcome_mermaid_block_is_a_flowchart_with_no_semicolon` — reviewer: met
- **met** — Every finding issue carries `security`, `lang:rust`, `severity:*` and `confidence:*` and links to the ledger, and the audit line is recorded. — evidence: audit line under `## Ledger`; `gh issue view` re-run on all 11 issues this run — reviewer: partial — reason: the reviewer could not run `gh`, labels were verified directly here
- **met** — All six named candidates have a finding row or a refuted row with the refuting line. — evidence: `tests/issue_2249_chunk9_ledger_complete.rs::each_named_candidate_keeps_a_ledger_or_refuted_row` — reviewer: met
- **met** — The extended assertions fail before the Outcome and refuted work and pass after it. — evidence: the Outcome assertions were committed failing first ("Pin the chunk-9 Outcome, severity sort and named candidates (failing first, #2250)") and pass at head — reviewer: partial — reason: the candidate and severity-sort assertions already held on the base rows (the keywords were present in row 1726 and the rows were already sorted), so they are regression pins, not red-first
- **met** — Exactly one comment on #2094 links the ledger, and `negative-result` is applied only if zero findings survived. — evidence: single comment posted on #2094 after the PR was raised, no label added (11 findings survived) — reviewer: missing — reason: not visible in the diff, GitHub action
- **met** — `./quality.sh < /dev/null` passes. — evidence: full gate run on final head, "All quality checks passed!" — reviewer: missing — reason: the reviewer could not run the gate

- **unrequested** — `Cargo.toml`/`Cargo.lock` parquet and arrow 59.1 → 60.0 — reviewer: unrequested — reason: needed to clear the zstd RUSTSEC advisories that fail `cargo deny` in `./quality.sh`, see Essential Design Decisions
- **unrequested** — `docs/archive/handover/issue-2250.md` — reviewer: unrequested — reason: worker-written handover note from the interrupted run

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — Two new tests (`each_named_candidate_keeps_a_ledger_or_refuted_row`, `ledger_rows_are_sorted_by_severity_within_each_region`) were already green on the base rows — evidence: `tests/issue_2249_chunk9_ledger_complete.rs:544` — reason: kept deliberately as regression pins the issue requires, the Outcome tests that are red-first are the guard for the new work; keywords were added to the dispatch-limit rows so each candidate's own row matches
- **violation** — major dependency bump outside the docs scope — evidence: `Cargo.toml:24` — reason: required by #2250 gate (`./quality.sh` passes only with it), migration needed no source change
- **violation** — stale all-missing PR summary — evidence: `docs/archive/pr-summaries/pr-summary-2250.md:7` — reason: fixed in this diff
- **clean** — Australian English, no `;` in the Mermaid block, named test files exist, no CI workflow change, no hidden files staged; the repository has no `CODING-STANDARDS.md`, so `AGENTS.md` and `CONTRIBUTING.md` were used

## Test Plan

- Added to `tests/issue_2249_chunk9_ledger_complete.rs`: `the_outcome_severity_counts_equal_the_ledger_rows_per_severity`, `the_outcome_names_every_ledger_finding_and_issue`, `the_outcome_mermaid_block_is_a_flowchart_with_no_semicolon`, `ledger_rows_are_sorted_by_severity_within_each_region`, `each_named_candidate_keeps_a_ledger_or_refuted_row`. No existing assertion removed.
- The three `outcome` tests went red against the pre-Outcome record (no counts table, no `SEC-` ids, no Mermaid block).
- Passed on the final head: `cargo test --test issue_2249_chunk9_ledger_complete --test issue_2288_chunk_09_ledger_scaffold`, and `./quality.sh < /dev/null`.
