## Summary

Re-sweeps the #2181 fan-in rows of the chunk 8b ledger now that the #2303 / #2304 fixes (PR #2331, commit `87ebeba`) are merged into milestone/2083. Closes #2322.

- `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs`: the two #2181 "still broken" tests went red on the sync, as designed. They are now `a_finite_record_set_no_longer_drives_the_fan_in_correlation_to_nan` and `fan_in_candidates_no_longer_depend_on_the_order_the_caller_lists_neurons_in`, and they pin the fixed behaviour. The first test now pins `corr == 0.0`, as the #2304 amendment asked, and keeps the finitude precondition. The locally spelled "threshold filter" assertion has been dropped: the production filter lives in `rank_input_scores` and is pinned by `tests/issue_2181_fan_in_non_finite_correlation_ranking.rs`. The precondition message in `every_float_comparator_in_the_swept_files_has_a_table_row` now names the `total_cmp` sorts rather than the old `partial_cmp(…).unwrap_or(Equal)` ones.
- `docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md` (`recommendation core`): the `fan_in.rs` file row, the per-target float-comparison row, ranking-integrity row (b) and the `pearson_correlation` finitude note now read "since fixed". Each cites `rank_input_scores` and the tests that pin the fix by symbol. Those tests are in `tests/issue_2181_fan_in_non_finite_correlation_ranking.rs`, plus `tests/scoring/issue_767_stats_pearson_correlation.rs::pearson_correlation_f32_overflow_returns_zero`.

## Evidence

This is a test and docs change only, so there is no UI. With stdin redirected, `cargo test --test issue_2108_chunk_08b_recommendation_core_sweep --test issue_2103_chunk_08b_ledger_scaffold --test issue_2210_chunk_08b_finalisation --test issue_2181_fan_in_non_finite_correlation_ranking` passed 34 tests with 0 failures. The full `./quality.sh` also passed: "All quality checks passed!", exit 0, 236 test suites ok.

## Reproduction

- **symptom** — after #2303/#2304 synced into milestone/2083, the #2108 sweep still asserted the fixed fan-in NaN-ranking defect: the correlation was NaN and the two neuron orderings disagreed.
- **status** — `verified` — the rewritten tests were run against the pre-fix `src/analysis/detection/stats.rs` and `src/analysis/recommendation/fan_in.rs` (restored temporarily from `87ebeba^1`). Both failed: `left: NaN, right: 0.0`, and 25 vs 0 candidates. Both pass at head.
- **regression test** — `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::a_finite_record_set_no_longer_drives_the_fan_in_correlation_to_nan` and `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::fan_in_candidates_no_longer_depend_on_the_order_the_caller_lists_neurons_in`

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The rewritten #2108 sweep tests pass on milestone/2083 with the #2303 fix merged in. — evidence: `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs::a_finite_record_set_no_longer_drives_the_fan_in_correlation_to_nan` — reviewer: missing — reason: the reviewer said "cannot verify" because it could not run tests. The suite was run here and all 11 tests passed.
- **met** — The `FILED_FINDINGS` contract, the #2103 ledger-scaffold and #2210 finalisation tests still pass. — evidence: `tests/issue_2103_chunk_08b_ledger_scaffold.rs` (14 passed), `tests/issue_2210_chunk_08b_finalisation.rs` (4 passed) — reviewer: missing — reason: the reviewer said "cannot verify by execution; no structural risk found". Those tests were run here and passed.
- **met** — `./quality.sh` passes. — evidence: full gate run after the final edit, exit 0 — reviewer: missing — reason: the reviewer said "cannot verify" because it could not run the gate. The gate was run here and passed.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Checked against CONTRIBUTING.md and AGENTS.md (the repo has no CODING-STANDARDS.md). Code is cited by symbol, never by line number, and every newly cited test exists. The Issue #1799 non-vacuous precondition is kept. No `src/`, dependency or CI change. Australian English throughout.

## Test Plan

- Modified `tests/issue_2108_chunk_08b_recommendation_core_sweep.rs`: dropped the locally spelled threshold assertion and updated the precondition message.
- Ran the #2108, #2103, #2210 and #2181 test targets, then the full `./quality.sh`.
