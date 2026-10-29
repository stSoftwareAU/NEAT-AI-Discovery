## Summary

Stages the chunk-8a (detection + neuron) sweep record skeleton under
`docs/audits/in-progress/` and pins its shape with a scaffold test.
Closes #2280.

- **Staged record:**
  `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`,
  copied from the sweep template. No top-level
  `docs/audits/security-sweep-chunk-08a-*` file is created, so
  `record_files()` in `tests/issue_2088_sweep_ledger_contract.rs` doesn't see it
  yet.
- **`## Record`:** baseline `b85a551`, audit HEAD `90e0c15`, exposure
  `internal`, and the citation convention "symbol names, not line numbers".
- **`## Files swept`:** 52 rows, all reading `pending`. Line counts come from
  `wc -l` at `90e0c15`. The rows are split under six `### ` headings, each fenced
  by a `<!-- section: … -->` marker. In order:

  | Section | Files | Owner |
  | --- | --- | --- |
  | shared | 5 | — |
  | graph | 8 | #2217 |
  | pairwise | 9 | #2150 |
  | per-neuron-a | 13 | #2218 |
  | per-neuron-b | 12 | #2152 |
  | neuron | 5 | #2153 |
- **`## Capacity and traversal table`:** added with its header row and no data
  rows yet.
- **`## Defect classes probed`:** the six #2092 classes, quoted word for word.
  #2092's own `topology_cache.rs:44` citation is kept inside the quote. A note
  after it gives the symbol, `CreatureTopologyCache::new`.
- **`## Related remediations (not sweep coverage)`:** the #2078/#1867 width caps,
  `MAX_CREATURE_INPUT_NEURONS` / `MAX_CREATURE_OUTPUT_NEURONS` = `1_000_000`,
  enforced by `validate_creature_input_bounds` in
  `src/ffi_types/creature_bounds.rs`.
- **`## Outcome`, `## Issues filed` and `## Verify this record`:** `pending`
  placeholders.
- **Index:** `docs/audits/lib-sweep-coverage.json` is untouched, and its `"8a"`
  entry stays all-`null`.

The change is docs and tests only, with no `src/` change. CI's
`version-increment` job handles the version bump.

## Evidence

This change touches only docs and tests, so there is no UI to capture.

- `cargo test --test issue_2280_chunk_08a_ledger_scaffold`: 6/6 pass.
- `cargo test --test issue_2088_sweep_ledger_contract`: 9/9 pass, including
  `prose_records_and_index_entries_match_both_ways` and
  `a_claimed_sweep_pins_a_baseline_commit_and_a_record`.
- The spec reviewer checked every row's line count against
  `git show 90e0c15:<path> | wc -l`, and all 52 match. The standards reviewer
  confirmed the counts are also identical at the baseline `b85a551`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` exists and `ls docs/audits/security-sweep-chunk-08a-* 2>/dev/null` prints nothing — evidence: `tests/issue_2280_chunk_08a_ledger_scaffold.rs::the_record_is_staged_under_in_progress`, `tests/issue_2280_chunk_08a_ledger_scaffold.rs::no_top_level_chunk_08a_record_exists` — reviewer: met
- **met** — `## Files swept` has exactly 52 file rows, each under its listed section, `grep -c '<!-- section:'` returns `6`, and every row reads `pending` — evidence: `tests/issue_2280_chunk_08a_ledger_scaffold.rs::files_swept_has_exactly_52_rows`, `tests/issue_2280_chunk_08a_ledger_scaffold.rs::files_swept_carries_exactly_six_section_markers_in_order`, `tests/issue_2280_chunk_08a_ledger_scaffold.rs::each_in_scope_file_has_exactly_one_row_under_its_owning_section`; the reviewer found all 52 rows `pending` with section counts 5/8/9/13/12/5 — reviewer: met
- **met** — `## Capacity and traversal table` heading and header row exist — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` — reviewer: met
- **met** — `## Defect classes probed` lists the six #2092 classes verbatim — evidence: the reviewer diffed the bullets against #2092's "Defect classes to probe" and found no difference — reviewer: met
- **met** — `## Related remediations (not sweep coverage)` quotes `1_000_000` — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, matching `src/ffi_types/creature_bounds.rs` — reviewer: met
- **met** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change — evidence: `git diff 90e0c15..HEAD -- docs/audits/lib-sweep-coverage.json` is empty, and `git diff --exit-code` against the PR base `origin/milestone/2083-security-scan-overflow-8-chunks-not-reached` exits 0; `tests/issue_2280_chunk_08a_ledger_scaffold.rs::the_chunk_8a_index_entry_is_still_all_null` — reviewer: met — reason: against `origin/Develop` the literal command does show the `"9"` and `"11"` entries changing, but those lines come from milestone commits a0aa61e and de7ef82, not this branch. The `"8a"` entry is still all-`null`.
- **partial** — `cargo test --test issue_2280_chunk_08a_ledger_scaffold`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: the two `cargo test` targets pass 6/6 and 9/9 — reviewer: partial — reason: the reviewer was read-only and did not run `./quality.sh`. The worker ran the gate on this branch before the PR-summary gate, and it passed. The worker re-runs it before raising the PR.
- **missing** — comments posted on #2217, #2150, #2218, #2152, #2153 and #2154 — reviewer: missing — reason: step 9 of the issue says to post these after the PR is raised, and no PR exists yet. The comments that already name the staged path on #2150, #2152, #2153 and #2154 are earlier "Plan published" or "Deferred" notes, not the required hand-offs.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Checked against `CONTRIBUTING.md`, `AGENTS.md`, `docs/audits/README.md` and the sweep template (the repo has no `CODING-STANDARDS.md`). No violations found:
  - **Australian English:** consistent throughout.
  - **Line-number citation:** the only one is `topology_cache.rs:44`, inside #2092's quote, which #2280 requires verbatim. It is tied to baseline `b85a551` and followed by the symbol `CreatureTopologyCache::new`, so the Issue #1942 rule is satisfied.
  - **Record shape:** every template section is present. The capacity-table header and the six-marker order match #2280.
  - **Line counts:** taken at audit HEAD, as #2280 asks, and identical at the baseline.
  - **Index:** the `"8a"` entry is still all-`null`.
  - **Mermaid:** the record has no Mermaid blocks.
  - **Markdown lint:** `markdownlint-cli2` reports 0 issues.
  - **Tests:** named after the issue and deterministic, since both directory listings are sorted. No test asserts `pending`. The helpers are reused from `tests/issue_2233_chunk_11_ledger_scaffold.rs`.
  - **Untouched areas:** CI, `Cargo.toml`, `Cargo.lock`, `deny.toml`, `fuzz/` and `src/` are all unchanged.

## Test Plan

- Added `tests/issue_2280_chunk_08a_ledger_scaffold.rs`, modelled on
  `tests/issue_2233_chunk_11_ledger_scaffold.rs`. It checks that:
  - the record is staged under `docs/audits/in-progress/`;
  - no top-level `security-sweep-chunk-08a-*` exists;
  - exactly six `<!-- section:` markers appear, in order;
  - there are exactly 52 rows;
  - every file in `src/analysis/detection/` and `src/analysis/neuron/` appears exactly once, under its owning section;
  - the `"8a"` index entry is still all-`null`.
- It deliberately doesn't assert `pending`, because the section issues flip
  those rows. Its module doc names the three assertions that finalisation (#2154)
  must delete or retarget when it `git mv`s the record to the top level.
- The existing `tests/issue_2088_sweep_ledger_contract.rs` passes unchanged.
