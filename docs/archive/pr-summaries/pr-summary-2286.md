## Summary

Creates the chunk-13 (`src/config`) sweep record `docs/audits/security-sweep-chunk-13-config-env.md` with every region pre-marked for the later chunk-13 sub-issues, fills the chunk-13 line of `docs/audits/lib-sweep-coverage.json`, and adds `tests/issue_2211_chunk_13_ledger_scaffold.rs` to pin the scaffold. Closes #2286.

Version bump: not changed by hand; CI's `version-increment` job handles it (issue instruction, AGENTS.md).

## Spec

### Intent and Rationale

- The record, the index entry and the scaffold test must land together because `tests/issue_2088_sweep_ledger_contract.rs` checks index↔record parity both ways.
- `user_facing.rs` is split into two range rows (L1–664 → 13b, L665–end → 13c) so each owner owns one row.

### Essential Design Decisions

- The issue says `7694130` is the only commit since the baseline touching `src/config`. At HEAD three are (`7694130`, `8c5639e` #2402, `2660968` #2407), and `user_facing.rs` is 1445 lines (not 1379). The re-check note records the HEAD facts; this replaces that sentence of the issue.
- `last_swept` is filled while rows read `pending` (as the issue requires, following chunk 8b). To keep the index date from reading as a finished sweep, the record carries `### Sweep status — IN PROGRESS` and the test requires it while any row reads `pending` (same approach as `tests/issue_2103_chunk_08b_ledger_scaffold.rs`). The test is conditional, so it does not block #2273/#2125 later.
- The settings schema keeps the required `reader file:line` column; the record asks sub-issues to also give the `file.rs::symbol` (CONTRIBUTING.md "Cite Code by Symbol"). Line spans were dropped from the quoted #1869/#1904 titles.
- Per-section marker counts are asserted; no whole-record marker total, so later sub-issues are not constrained.

### Undiscoverable Facts

- `./quality.sh` runs the whole `cargo test --all-features` suite and did not finish inside the 10-minute foreground limit on this host.

## Evidence

Backend/docs only; no UI. Targeted runs on the final head:

- `cargo test --test issue_2088_sweep_ledger_contract` — 9 passed.
- `cargo test --test issue_2211_chunk_13_ledger_scaffold` — 6 passed.
- `rustfmt --edition 2024 --check` on the new test and `cargo clippy --test issue_2211_chunk_13_ledger_scaffold -- -D warnings` — clean.

Mutation checks (applied locally, scaffold test failed each time, then reverted):

| Mutation | Failing test |
| --- | --- |
| delete the `src/config/detection.rs` file row | `every_config_file_has_one_row_and_user_facing_is_split_across_two_regions` |
| move the `user_facing.rs` 13c row (732) under the 13b heading | `every_config_file_has_one_row_and_user_facing_is_split_across_two_regions` |
| delete the 13c marker in `## Settings table` | `each_section_marker_appears_exactly_once_in_its_section` |
| rename settings column `type` → `kind` in one header | `the_settings_table_has_one_ten_column_header_per_region` |
| delete the `### Sweep status — IN PROGRESS` heading | `an_unfinished_chunk_says_so_above_its_table` |

<!-- vibe-quality-gate-skipped reason="./quality.sh runs the full cargo test suite and exceeded the 10-minute foreground limit; targeted tests, fmt and clippy were run instead; CI runs the full gate" -->

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `docs/audits/security-sweep-chunk-13-config-env.md` exists at the top level of `docs/audits/` with the filled `## Record` including `**Swept by:**`, the re-check note citing `7694130`, the seven related remediations, the seven defect classes, `pending` rows for all six `src/config/*.rs` files with `user_facing.rs` as two range rows summing to 1396, four ten-column settings-table headers and every section marker — evidence: `tests/issue_2211_chunk_13_ledger_scaffold.rs::every_config_file_has_one_row_and_user_facing_is_split_across_two_regions` — reviewer: partial — reason: reviewer flagged only that the re-check note names three commits and 1445 lines at HEAD instead of the issue's one commit and 1379; HEAD verified with `wc -l` and `git log b85a551..HEAD -- src/config`
- **met** — the chunk-13 line of `lib-sweep-coverage.json` carries the ISO date, the full SHA and the record path in key order, and `cargo test --test issue_2088_sweep_ledger_contract` passes — evidence: `tests/issue_2088_sweep_ledger_contract.rs` 9 passed — reviewer: met
- **met** — `cargo test --test issue_2211_chunk_13_ledger_scaffold` passes — evidence: 6 passed — reviewer: met
- **met** — the four local mutations each fail the scaffold test and the PR body records all four — evidence: Evidence table above — reviewer: met
- **partial** — `./quality.sh` passes — reviewer: missing — reason: quality gate not yet run — remaining: merge-verification-only (gate exceeded the 10-minute foreground limit on this host; CI runs it on the PR)
- **unrequested** — `### Sweep status — IN PROGRESS` heading and its conditional test — reviewer: unrequested — reason: added to resolve the standards finding that a non-null `last_swept` over pending rows reads as a finished sweep (chunk 8b precedent)

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — index `last_swept` set while the record is a pending scaffold — evidence: `docs/audits/lib-sweep-coverage.json:12` — reason: fixed in this diff (in-progress status heading plus test; the date itself is required by the issue)
- **violation** — `reader file:line` column and quoted line spans conflict with cite-by-symbol — evidence: `docs/audits/security-sweep-chunk-13-config-env.md:80` — reason: fixed in this diff (column fixed by the issue, symbol requirement added to the record, line spans removed from quoted titles)
- **violation** — PR summary missing — evidence: `docs/archive/pr-summaries/pr-summary-2286.md` — reason: fixed in this diff (this file)
- **clean** — Australian English, no workflow files touched, no hidden files staged, tests call real code and pin positive preconditions, index entry stays one line in key order

## Test Plan

- Added `tests/issue_2211_chunk_13_ledger_scaffold.rs` (6 tests): `the_record_exists_and_pins_its_chunk_id_exposure_baseline_and_parent`, `every_config_file_has_one_row_and_user_facing_is_split_across_two_regions`, `each_section_marker_appears_exactly_once_in_its_section`, `the_settings_table_has_one_ten_column_header_per_region`, `the_index_names_the_chunk_13_record_and_its_baseline`, `an_unfinished_chunk_says_so_above_its_table`.
- Red checks: see the mutation table above; each test went red with only its guard broken.
- Removed assertions from existing tests: none (no existing test edited).
- Twinned rules: the record shape is also checked by `tests/issue_2088_sweep_ledger_contract.rs` (ledger-wide) — it passes unchanged.
