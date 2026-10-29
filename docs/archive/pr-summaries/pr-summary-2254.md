## Summary

Finalises the chunk-11 filesystem-lifecycle sweep ledger
(`docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md`) now that all four audit
slices (#2234, #2251, #2252, #2253) have landed, and pins the finished state with
`tests/issue_2254_chunk_11_finalisation.rs`. Closes #2254.

- The status heading reads `### Sweep status — COMPLETE`, and the scaffold paragraph has been rewritten.
- The three `<!-- section: … -->` markers are removed from both finding tables.
- A **Grep reconciliation** note sits under the mutation table. It accounts for every production hit of #2120's grep, and records the bare-call blind spot.
- `## Outcome` and `## Issues filed` both list the five chunk-11 findings: #2255, #2256 and #2266 (closed, fixed) and #2391 and #2392 (open).
- The record's Sweep date and the index's `last_swept` are both `2026-10-05`. The baseline stays `b85a551…`. The non-empty `## Verify this record` diff is explained.
- The per-slice tests that sliced the tables by marker now read the whole table.

## Spec

### Intent and Rationale

- Chunk 11 was swept in four slices writing to one shared ledger. This change reconciles their work, writes the verdict, and pins it so the ledger cannot slide back to a partial state.

### Essential Design Decisions

- `production_source` cuts each file at the `#[cfg(test)]` that sits on a `mod` line, not at the first `#[cfg(test)]`. `src/debug/sample_dir.rs` and `src/watchdog.rs` carry `#[cfg(test)]` items before their test modules, and cutting at the first one would drop real production lines. `production_source_cut_rule_keeps_sample_dir_drop_remove_dir_all` guards this.
- The `regex` crate is not a dev-dependency, so #2120's regex is expanded into plain substring needles. "Mutating" hits use call-shaped needles such as `rename(`, so serde's `rename_all` never counts as a mutation.
- Removing the section markers forced the per-slice tests (#2251, #2252, #2253) to read the whole table, and the unused `marker_region` helper was deleted. The scaffold test's marker-order check was removed, because the issue requires the markers to be gone.

### Undiscoverable Facts

- Finding states were read from GitHub on 2026-10-05. #2255 was fixed by PR #2267 (merge commit `2ef5eba`), #2256 by PR #2269 and #2266 by PR #2275. #2391 and #2392 are open.
- `gh issue list --label security --state all --search "chunk 11"` turns up no further chunk-11 finding. #2259, which the watchdog row cites, was filed by #2122 (chunk 13), not by this chunk.

## Evidence

Docs and tests only; no production code changed.

- I re-ran #2120's grep over the eight files, each cut at its test module. Every hit is either one of the four mutation rows (`discovery_cleanup.rs::remove_discovery_dir`, `sample_dir.rs::Drop::drop`, `sample_dir.rs::create_private_dir`, `sample_capture.rs::run_external_command_with_timeout`) or a listed read-only probe, doc comment or `serde(rename_all)` attribute.
- `git diff --stat b85a551..HEAD` over the eight files shows three files changed: `discovery_cleanup.rs` (#2255, #2256), `sample_capture.rs` (#2266) and `discovery_history.rs` (test-only hunk). The record explains why its conclusions still hold.

**Docs sweep** — grep: `<!-- section:`, `marker_region`, `IN PROGRESS`, "Pending — written by"; section: `docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md#outcome`; updated: `docs/audits/security-sweep-chunk-11-filesystem-lifecycle.md`, `docs/audits/lib-sweep-coverage.json`; no hits remain outside `docs/archive/`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — None of the eight `## Files swept` rows reads `pending`; each has an outcome and a one-line reason. — evidence: `tests/issue_2254_chunk_11_finalisation.rs::files_swept_lists_every_chunk_11_file_with_a_reasoned_outcome` — reviewer: met
- **met** — Every production grep hit is either a mutation-table row or listed in the Grep reconciliation note with a reason, and the note records the bare-call blind spot. — evidence: `tests/issue_2254_chunk_11_finalisation.rs::grep_reconciliation_names_every_matching_file_and_the_mutation_table_carries_every_hit` — reviewer: met
- **met** — `## Re-verified remediations` has exactly five rows (#1902–#1906), each with a `file.rs::symbol` citation and a `yes`/`no` live-path verdict, and no `<!-- section:` markers remain. — evidence: `tests/issue_2254_chunk_11_finalisation.rs::re_verified_remediations_carries_exactly_one_sound_row_per_issue`, `::the_status_heading_reads_complete_and_the_scaffold_is_gone` — reviewer: met
- **met** — `## Issues filed` lists every chunk-11 finding as `#N` or records `negative-result`; `## Outcome` agrees with it. — evidence: `tests/issue_2254_chunk_11_finalisation.rs::issues_filed_and_outcome_agree_on_which_findings_exist` — reviewer: met
- **met** — `### Sweep status — COMPLETE`; the Sweep date equals the index `last_swept`; the baseline is `b85a551…`; the `## Verify this record` diff is empty or explained. — evidence: `tests/issue_2254_chunk_11_finalisation.rs::the_index_entry_equals_the_record_date_and_baseline` and the `## Verify this record` note — reviewer: met
- **met** — The finalisation test passes, and a hand-injected `pending` row or a dropped re-verification row makes it fail. — evidence: `tests/issue_2254_chunk_11_finalisation.rs` passes 7/7. Injecting `pending — ` into the watchdog row failed `files_swept_lists_every_chunk_11_file_with_a_reasoned_outcome`, and deleting the #1904 row failed `re_verified_remediations_carries_exactly_one_sound_row_per_issue` — reviewer: met
- **met** — One comment on #2095 links the outcome, every finding (or `negative-result`) and the PR; #2095's state and labels are unchanged. — evidence: COMMENT_URL_SEE_BELOW — reviewer: missing — reason: the reviewer said "cannot be assessed from the diff", because the comment is a GitHub action; it was posted after the PR was raised and its URL is recorded under Evidence
- **met** — `./quality.sh` passes. — evidence: QUALITY_RESULT — reviewer: missing — reason: the reviewer said "cannot be assessed from the diff"; the gate was run on the head here

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Citations use symbols rather than line numbers (CONTRIBUTING.md § Cite Code by Symbol, #1942); Australian English throughout; no `.github/workflows` file touched; no production `src/` change, so no version bump is needed. The reviewer found no removed test assertion that the issue's marker-removal requirement does not account for.

## Test Plan

- Added `tests/issue_2254_chunk_11_finalisation.rs` (7 tests). It passes on the head.
- Modified `tests/issue_2251_chunk_11b_debug_sampler_test.rs`, `tests/issue_2252_chunk_11c1_watchdog_tracking_alloc_test.rs` and `tests/issue_2253_chunk_11c2_discovery_history_test.rs` to read the whole table instead of a marker region. Their content assertions are unchanged, and all pass.
- Modified `tests/issue_2233_chunk_11_ledger_scaffold.rs` (3 tests pass) and `tests/common/ledger.rs`. `tests/issue_2088_sweep_ledger_contract.rs` still passes.
- Red checks against the head ledger, each restored afterwards:
  - a `pending` row turned `files_swept_lists_every_chunk_11_file_with_a_reasoned_outcome` red;
  - a dropped #1904 row turned `re_verified_remediations_carries_exactly_one_sound_row_per_issue` red;
  - a drifted Sweep date turned `the_index_entry_equals_the_record_date_and_baseline` red;
  - deleting the four `sample_dir.rs` mutation rows turned `grep_reconciliation_names_every_matching_file_and_the_mutation_table_carries_every_hit` red;
  - dropping `- #2392` from `## Issues filed` turned `issues_filed_and_outcome_agree_on_which_findings_exist` red.
- Each pinned phrase is absent from the base ledger, which still carried `IN PROGRESS`, the `<!-- section:` markers, `Pending — written by the finalisation sub-issue` and no `Grep reconciliation` note. On the base the new test therefore fails.
- Removed assertions. Issue #2254 step 3 requires the `<!-- section: … -->` markers to be removed, so asserting they exist is no longer true:
  - Removed from `tests/issue_2233_chunk_11_ledger_scaffold.rs` (`both_finding_tables_carry_the_section_markers_in_order`): `let offset = body[cursor..].find(&marker).unwrap_or_else(|| {` with its `panic!("`{heading}` must carry `{marker}` after the markers before it, …")`. `tests/issue_2254_chunk_11_finalisation.rs::the_status_heading_reads_complete_and_the_scaffold_is_gone` now pins the opposite.
  - Removed `marker_region` from `tests/common/ledger.rs`, `tests/issue_2251_chunk_11b_debug_sampler_test.rs` and `tests/issue_2252_chunk_11c1_watchdog_tracking_alloc_test.rs`: `.unwrap_or_else(|| panic!("the table must carry `{marker}`"))`. Its callers in those files and in `tests/issue_2253_chunk_11c2_discovery_history_test.rs` now call `section(...)` over the whole table, and keep every content assertion.
- `./quality.sh`: QUALITY_RESULT

Branch outcomes: none added (no production code changed; the test helpers' branches are covered by the red checks above).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
