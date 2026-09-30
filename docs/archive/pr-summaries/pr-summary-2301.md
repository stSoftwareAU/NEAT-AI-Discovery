## Summary

Audits `src/analysis/neuron/evaluation.rs`, `post_processing.rs` and
`ranking_score.rs` against the six #2092 defect classes and completes the
`neuron` section of the staged chunk 8a ledger. No `neuron` row reads
`pending` any more, and the sweep contract test now covers all five neuron
files. This is a docs-and-tests change with no `src/` edit. Closes #2301.

- [x] Fill the `evaluation.rs`, `post_processing.rs` and `ranking_score.rs` rows
- [x] Add capacity rows for `apply_distinct_target_spread` (`spread`, `rest`), for 4 neuron capacity rows in total
- [x] Add traversal rows for the `apply_distinct_target_spread` partition loop and `sort_candidates_by_rank`
- [x] Record the `#2181` dedup and the no-new-finding result under `## Issues filed`
- [x] Widen `tests/issue_2300_chunk_08a_neuron_sweep.rs`
- [x] Post summary comments on #2153 and #2092

Verdicts:

| File | Outcome |
| --- | --- |
| `evaluation.rs` | clean — `passes_neuron_improved_ratio` divides only after the `total_count == 0` early return (NaN refuted). No capacity site. The per-source loop is deadline-checked. Every narrowing saturates |
| `post_processing.rs` | bounded — `spread` is clamped to ≤ 32 and `rest` is sized from the live `Vec`. Every `sort_by` and `expect("fits in u32")` is test-only. A NaN impact score is dropped by the `>= floor` gain floor. `RecordCache` is reached only through `compute_impact_scores_for_discounting`, with the per-creature key and lifetime recorded in the `neuron/mod.rs` row |
| `ranking_score.rs` | clean — `sort_candidates_by_rank` chains `total_cmp` on `candidate_rank_score`, then on `expected_creature_score_gain`, so it is total under NaN. Checked against #2181, which it does not duplicate |

**No findings were filed.** Every probe was either refuted or shown to be bounded.

## Evidence

This is a docs-and-tests change with no UI. Evidence:

- `cargo test --test issue_2300_chunk_08a_neuron_sweep`: 10 passed. The same
  run also covered `issue_2280_chunk_08a_ledger_scaffold`,
  `issue_2281_chunk_08a_shared_sweep`, `issue_2294_chunk_08a_pairwise_sweep`
  and `issue_2088_sweep_ledger_contract` (31 passed across 5 suites).
- I temporarily set the `evaluation.rs` row back to `pending` and confirmed
  that `each_swept_row_is_present_and_not_pending` and
  `no_row_in_the_neuron_region_reads_pending` both fail. I then restored the
  row.
- `docs/audits/lib-sweep-coverage.json` is not changed by this PR. It still
  differs from `origin/Develop`, but only through earlier milestone commits
  (the chunk 9 and chunk 11 entries). No top-level
  `docs/audits/security-sweep-chunk-08a-*` file exists.
- `./quality.sh < /dev/null` passed after the final commit.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — All five `neuron` rows carry a non-`pending` `<outcome> — <reason>` with a verdict for each of the six #2092 classes, and no row outside `neuron` changes — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, `tests/issue_2300_chunk_08a_neuron_sweep.rs::no_row_in_the_neuron_region_reads_pending` — reviewer: met
- **met** — The capacity table holds exactly 4 neuron capacity rows, plus traversal rows for `apply_distinct_target_spread` and the `ranking_score.rs` sort — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs::exactly_four_neuron_capacity_rows`, `::the_neuron_traversal_rows_are_recorded` — reviewer: met
- **met** — The `evaluation.rs` NaN and the `ranking_score.rs` comparator each carry an explicit verdict — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs::the_nan_and_comparator_verdicts_are_explicit` — reviewer: met
- **met** — Every surviving finding is filed in the #2078 shape, deduplicated against #2181 and linked from `## Issues filed` — evidence: no finding survived; the `#2181` dedup and the no-new-finding result are recorded under `## Issues filed`; `::every_issue_linked_from_a_swept_row_appears_under_issues_filed` — reviewer: met
- **met** — `lib-sweep-coverage.json` is unchanged and no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — evidence: `git diff` against the PR base (the milestone branch) shows no change to `lib-sweep-coverage.json`, and `ls docs/audits/security-sweep-chunk-08a-*` finds nothing — reviewer: partial — reason: the reviewer saw that the diff does not touch the file but did not run the command. Run here, `git diff --exit-code origin/Develop` does report a difference, but it comes only from earlier milestone commits (chunks 9 and 11), not from this PR
- **met** — `cargo test --test issue_2300_chunk_08a_neuron_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: the targeted runs above and the full gate after the final commit — reviewer: missing — reason: the reviewer saw only the diff and could not run commands. The commands were run here and passed

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — `CODING-STANDARDS.md` is absent, so the review used `CONTRIBUTING.md`. It checked that code is cited by symbol rather than by line number, that the text uses Australian English, and that the tests assert observable ledger content and pin positive preconditions (#1799). It also checked that there are no wall-clock assertions, no new `pub` surface, and no changes to CI, manifests or environment variables. It found no violations. (optional: the audit cells are dense)

## Test Plan

Modified `tests/issue_2300_chunk_08a_neuron_sweep.rs`:

- Widened `SWEPT` to all five neuron files.
- Added `no_row_in_the_neuron_region_reads_pending`.
- Added `exactly_four_neuron_capacity_rows`.
- Added `the_nan_and_comparator_verdicts_are_explicit`.
- Added `the_neuron_traversal_rows_are_recorded`.
- Added a second pinned precondition that the production source of `post_processing.rs` contains `Vec::with_capacity(min_distinct)`.
- Extended the zero-capacity negative-result check to `evaluation.rs` and `ranking_score.rs`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
