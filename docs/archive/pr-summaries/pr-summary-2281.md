## Summary

Audits the five shared detection utilities (`src/analysis/detection/`
`mod.rs`, `helpers.rs`, `stats.rs`, `topology_cache.rs` and
`activation_properties.rs`) against the six #2092 defect classes. It fills the
`shared` section of the staged chunk-8a ledger, adds the 8 shared capacity rows
and files the one surviving finding. Closes #2281.

- **`shared` rows:** all five now read `<outcome> — <reason>` and cite code by
  symbol. `mod.rs`, `helpers.rs` and `activation_properties.rs` are `clean`,
  `topology_cache.rs` is `bounded` and `stats.rs` is `finding #2343`.
- **Length questions:** both are ruled out, with caller evidence cited by
  symbol.
  - `spearman_rank_correlation`: its only production caller,
    `monotonicity.rs::detect_non_monotonic_neurons`, truncates both series to
    the same length.
  - `pearson_correlation_samples`: `redundant_path.rs`, `pre_screening.rs` and
    `scoring.rs` each pass a `min` of the two sample lengths.
- **Capacity table:** 8 shared rows, 7 for `CreatureTopologyCache::new` and 1
  for `compute_ranks`.
  - The five `neuron_count` rows state there is no numeric cap.
  - The `output_uuids` / `input_uuids` rows cite the `1_000_000` cap from
    `validate_creature_input_bounds` at the FFI boundary. They also note that
    `new` is `pub`, so it is unbounded for a Rust caller that skips
    `validate_creature`.
- **`## Issues filed`:** links #2343 (`pearson_correlation_hashmaps` returns
  NaN on `f32` variance overflow), with labels `security`, `lang:rust`,
  `severity:low` and `confidence:high`.
- **Test:** adds `tests/issue_2281_chunk_08a_shared_sweep.rs`.
- **Untouched:** `docs/audits/lib-sweep-coverage.json` and all of `src/`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — All five `shared` rows carry a non-`pending` `<outcome> — <reason>`, and rows outside `shared` are unchanged — evidence: `tests/issue_2281_chunk_08a_shared_sweep.rs::every_shared_row_carries_a_non_pending_outcome_with_a_reason` passes, and the diff against the PR base has only three hunks (shared rows, capacity table, `## Issues filed`) — reviewer: met
- **met** — The capacity table has exactly 8 shared rows (7 `CreatureTopologyCache::new`, 1 `compute_ranks`), each naming its bound, and the five `neuron_count` rows state there is no numeric cap — evidence: `tests/issue_2281_chunk_08a_shared_sweep.rs::the_capacity_table_has_exactly_eight_shared_rows` and `tests/issue_2281_chunk_08a_shared_sweep.rs::every_shared_capacity_site_is_cited_by_symbol` pass; the reviewer matched the rows to the 7 `with_capacity` calls in `topology_cache.rs` — reviewer: met
- **met** — The `spearman_rank_correlation` and `pearson_correlation_samples` length questions each have a recorded verdict — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` `stats.rs` row, ruled out by `monotonicity.rs::detect_non_monotonic_neurons`, `redundant_path.rs::evaluate_pair_for_redundancy`, `pre_screening.rs::evaluate_residual_reduction` and `scoring.rs::compute_sample_correlation`; the reviewer confirmed each claim against `src/` — reviewer: met
- **met** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*` and `confidence:*`, is linked from `## Issues filed`, and duplicates no open issue — evidence: `gh issue view 2343` shows it open with `security`, `lang:rust`, `severity:low` and `confidence:high`. The reviewer searched all states and found no duplicate: #2304 (closed) fixed only `pearson_correlation`, #2322 only mentions `pearson_correlation_hashmaps` in passing, and #2078 and #2168 are closed — reviewer: met
- **met** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change, and no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — evidence: `git diff --exit-code` against the PR base `origin/milestone/2083-security-scan-overflow-8-chunks-not-reached` exits 0, and the record exists only under `docs/audits/in-progress/` — reviewer: met — reason: against `origin/Develop` the literal command exits 1 on 2 lines, but those lines come from milestone commits. No commit on this branch touches the file.
- **met** — `cargo test --test issue_2281_chunk_08a_shared_sweep` and `cargo test --test issue_2088_sweep_ledger_contract` pass — evidence: both suites now pass in full (4/4 and 9/9). The CI `Coverage` check failure on `every_issue_linked_from_a_shared_row_appears_under_issues_filed` (tracker `#2092`, cited from the `mod.rs` shared row, was missing from `## Issues filed`) is fixed by adding a bullet under `## Issues filed` noting `#2092` is the tracker issue, not a sweep-filed finding, following the existing "Prior remediations … not filed by this sweep" pattern — reviewer: met

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **resolved** — The new test failed on the prior tree — evidence: `tests/issue_2281_chunk_08a_shared_sweep.rs::every_issue_linked_from_a_shared_row_appears_under_issues_filed` panicked at `tests/issue_2281_chunk_08a_shared_sweep.rs:363` because the reference to tracker `#2092` in the `mod.rs` row was not under `## Issues filed` — reason: fixed by adding `#2092` under `## Issues filed`, explicitly marked as the tracker issue and not a sweep-filed finding, matching the record's existing "Prior remediations … not filed by this sweep" convention rather than weakening the test's own matching rule.
- **violation** — Prior remediations are listed under `## Issues filed` — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` "Prior remediations cited by `shared` rows" bullet (#2078, #2304), compared with `docs/audits/security-sweep-TEMPLATE.md` — reason: stands, because this run was limited to the summary. The template keeps that section for issues this sweep filed. #2304 belongs under `## Related remediations (not sweep coverage)`, which already lists #2078.
- **violation** — Dead code in the test — evidence: `tests/issue_2281_chunk_08a_shared_sweep.rs:341` computes `capacity_region`, and line 347 discards it with `let _ = capacity_region;` — reason: stands, because this run was limited to the summary. It is minor.
- **violation** — The test drifts from its sibling 08b section tests — evidence: `tests/issue_2281_chunk_08a_shared_sweep.rs` compared with `tests/issue_2104_chunk_08b_synapse_pipeline_sweep.rs` — reason: stands, because this run was limited to the summary. It is minor. The capacity scan doesn't count `.reserve(` and uses `find(']')` instead of `rfind`. There is no "cited symbols still exist" check. The reason check accepts any non-empty text where the siblings require more than 20 characters. The `topology_cache.rs` non-vacuous precondition is pooled across all five files instead of being pinned to that file as the issue asks.
- **violation** — Stale prose in the record — evidence: the `shared` heading in `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` still says "no dedicated sub-issue", and the `stats.rs` row keeps the audit-HEAD count of 224 lines while describing the current 230-line file — reason: stands, because this run was limited to the summary. It is minor.
- **clean** — The reviewer checked these areas and found no problems:
  - **Australian English:** consistent in new prose and comments.
  - **Citations:** all new rows cite by symbol, not line number.
  - **Mermaid:** none added.
  - **Facts in the rows:** confirmed against `src/`, including 7 `with_capacity` sites, 12 `as` casts in `stats.rs`, `total_cmp` sorts and the `1_000_000` caps.
  - **Test hygiene:** the test is named after the issue and deterministic. `rustfmt --check` and `cargo clippy --test issue_2281_chunk_08a_shared_sweep -- -D warnings` are clean.
  - **Other tests:** `tests/issue_2280_chunk_08a_ledger_scaffold.rs` still passes 6/6.
  - **Untouched areas:** `.github/workflows/`, `Cargo.toml`, `Cargo.lock`, `deny.toml`, `src/` and `docs/audits/lib-sweep-coverage.json` are unchanged.

## Test Plan

- `tests/issue_2281_chunk_08a_shared_sweep.rs` reads the
  `<!-- section: shared -->` marker region and checks four things:
  - no shared row reads `pending`;
  - every production `with_capacity` / `vec![…; n]` site in the five shared
    files is cited by symbol in the capacity table, with a non-vacuous
    precondition;
  - the table has exactly 8 shared rows;
  - every issue linked from a shared row appears under `## Issues filed`.
- Current result: all 4 tests pass, including the `#2092` tracker-linkage
  check fixed above.
- `tests/issue_2088_sweep_ledger_contract.rs` passes 9/9, and
  `tests/issue_2280_chunk_08a_ledger_scaffold.rs` passes 6/6.
