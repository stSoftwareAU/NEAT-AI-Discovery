## Summary

Audits `src/analysis/neuron/mod.rs` and `src/analysis/neuron/preparation.rs`
(1,681 lines) against the six #2092 defect classes. It fills their two rows in
the `neuron` section of the staged chunk-8a ledger, adds the neuron capacity and
traversal rows and files the one surviving finding. Closes #2300.

- **`neuron` rows:** `mod.rs` reads `finding #2352` and `preparation.rs` reads
  `bounded`. Both give a verdict for every #2092 class and cite code by symbol.
  `evaluation.rs`, `post_processing.rs` and `ranking_score.rs` stay `pending`.
- **Cache poisoning:** the `mod.rs` outcome records the following:
  - **Key:** the raw, un-namespaced neuron UUID, with every neuron call site
    listed.
  - **Lifetime:** one `RecordCache` per analysis and per creature. There is no
    `static`, `OnceLock` or `lazy_static` instance.
  - **Failed loads:** a failed load is memoised in the `OnceLock<Result<…>>` of
    `CachedNeuronRecords`.
  - **Synthetic UUIDs:** neither synthetic UUID reaches `RecordCache::get`. The
    collision verdict is cross-linked to #2295.
- **Capacity and traversal table:** adds two `preparation.rs` capacity rows. One
  records the unchecked `usize + usize` against the `1_000_000` input cap and
  the per-input `Arc<str>` loop. The table also gains one `try_fold` traversal
  row marked `yes (per focus target)`.
- **`## Issues filed`:** links #2352 (`security`, `lang:rust`, `severity:low`,
  `confidence:high`). In release, a failed `RecordCache::get(target_uuid)`
  drops the focus target silently, and `NeuronDiagnostics` reports it as
  `NoEligibleSources`.
- **Test:** adds `tests/issue_2300_chunk_08a_neuron_sweep.rs`.
- **Untouched:** `docs/audits/lib-sweep-coverage.json` and all of `src/`.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The `neuron/mod.rs` and `neuron/preparation.rs` rows carry a non-`pending` `<outcome> — <reason>` with a verdict for each of the six #2092 defect classes, and the other three neuron rows and every row outside `neuron` are unchanged — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs::each_swept_row_is_present_and_not_pending` passes; the diff touches only those two rows in the `neuron` section; the reviewer spot-checked the panic-site, division and guard claims against `src/` — reviewer: met
- **met** — The capacity table gains exactly 2 neuron capacity rows (both `preparation.rs`, including the unchecked `usize + usize` and its bounds) and 1 `try_fold` traversal row marked `yes (per focus target)` — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs::every_capacity_site_in_a_swept_file_is_cited_by_symbol` (pinned `HashMap::with_capacity(input.creature.input + input.creature.neurons.len())` precondition) and `::the_try_fold_traversal_row_is_cancellation_checked_per_focus_target` pass; the two rows match the only two regex hits in `preparation.rs` — reviewer: met
- **met** — The `neuron/mod.rs` outcome states the key derivation, the per-analysis/per-creature lifetime, the failed-load memoisation and the synthetic-UUID reach, and cross-links #2295 — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs::the_mod_rs_outcome_records_the_record_cache_verdict` passes; the reviewer confirmed each claim against `src/analysis/cache/mod.rs`, `src/analysis/orchestration.rs`, `hard_sample_cluster.rs` and `output_conflict.rs` — reviewer: met
- **met** — Every surviving finding is filed in the #2078 shape, linked from `## Issues filed`, and not a duplicate of an open issue — evidence: `gh issue view 2352` shows it open with labels `security`, `lang:rust`, `severity:low` and `confidence:high`, naming `tests/issue_2352_neuron_target_load_failure_surfaced.rs` as the failing-first test; `tests/issue_2300_chunk_08a_neuron_sweep.rs::every_issue_linked_from_a_swept_row_appears_under_issues_filed` passes; duplicate searches ("RecordCache", "NoEligibleSources", "target load failure neuron") found only unrelated issues (#1101 synapse diagnostics, #2181 closed fan-in comparator) — reviewer: met
- **met** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change, and no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — evidence: against the PR base `origin/milestone/2083-security-scan-overflow-8-chunks-not-reached` the command exits 0, neither branch commit (5685e12, a0696e1) touches the file, and `ls docs/audits/security-sweep-chunk-08a-*` finds nothing — reviewer: met — reason: against `origin/Develop` the literal command exits 1 on the chunk 9 and chunk 11 entries, which come from milestone commits a0aa61e and de7ef82, not this branch
- **met** — `cargo test --test issue_2300_chunk_08a_neuron_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: 6/6 and 9/9 pass; `./quality.sh` passed in the worker's quality gate for this run — reviewer: met — reason: the reviewer ran the two suites but not `./quality.sh`, so that pass comes from the worker gate

No unrequested change: the reviewer traced both files to the issue, including
the `#2295` bullet (step 5's cross-link) and
`production_source_skips_the_test_only_cfg_test_fn_in_post_processing` (step 8's
`#[cfg(test)]`-then-`mod tests` cut).

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — The quadratic-blowup verdict understates the per-target work — evidence: the `src/analysis/neuron/mod.rs` row and the `mod.rs::analyze_neurons_with_cache_and_gpu_queue` traversal row in `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` — reason: stands, because this run was limited to the summary. The `try_fold` closure calls `group_sources_by_locality` (the #2161 pairwise scan, capped at `MAX_SOURCES_FOR_LOCALITY_SCAN` = 1024 and deadline-checked per outer iteration). The real cost is O(targets × sources²), and the table has no `pairwise loop` row for it. The `bounded` outcome still holds.
- **violation** — The traversal row paraphrases the guard, and the test cannot catch a change to it — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs:411-467` — reason: stands (low). The source reads `analysis_timed_out.load(Ordering::Relaxed) || saturation_aborted.load(Ordering::Relaxed) || deadline_passed(&deadline)`, but the test only looks for `.try_fold(` and `deadline_passed(&deadline)`, so dropping either atomic would still pass.
- **violation** — The `RecordCache` verdict test only checks for keywords — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs:469-495` — reason: stands (low). The test would pass if the verdict were reversed, and nothing pins the "no static `RecordCache`" claim, although a grep shows it holds today.
- **violation** — Copies of the helpers are drifting between sibling tests — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs:87-174` compared with `tests/issue_2281_chunk_08a_shared_sweep.rs` — reason: stands (low). The new copy fixes `\|` cell splitting and the `#[cfg(test)]` cut, but the 2281 copy does not, which leaves three diverging copies.
- **violation** — A test pins a helper against a file outside `SWEPT` — evidence: `tests/issue_2300_chunk_08a_neuron_sweep.rs:176-211` pins `post_processing.rs` text — reason: stands (nit). Step 8 asks for it, but it will break on an unrelated `post_processing.rs` refactor.
- **violation** — Non-findings are listed under `## Issues filed` — evidence: the #2295 bullet in the ledger's `## Issues filed` — reason: stands (nit). The test contract requires every `#N` linked from a swept row to appear there, and the record already does the same for #2092, #2078 and #2304. The bullet is labelled "not a finding filed by this sweep".
- **violation** — The two sections cite capacity-table symbols differently — evidence: the neuron rows use `src/analysis/neuron/…::`, while the shared rows use basenames — reason: stands (nit). Step 2 of the issue requires the full path, but the ledger does not record why.
- **clean** — The reviewer checked these areas and found no problems:
  - **Australian English:** used in the new prose and comments.
  - **Citations:** all new rows cite by symbol, with no line numbers.
  - **Mermaid:** none added.
  - **Test hygiene:** the test is named after the issue and deterministic. `rustfmt --check` passes, and `cargo clippy --test issue_2300_chunk_08a_neuron_sweep -- -D warnings` is clean.
  - **Vacuous passes:** a pinned capacity site and exact row counts guard against them.
  - **Sibling tests:** `issue_2280`, `issue_2281` and `issue_2088` still pass.
  - **Facts in the rows:** confirmed against `mod.rs`, `preparation.rs`, `cache/mod.rs` and `orchestration.rs`.
  - **Untouched areas:** `.github/workflows/`, `Cargo.toml`, `Cargo.lock`, `src/` and `docs/audits/lib-sweep-coverage.json` are unchanged.

## Test Plan

- `tests/issue_2300_chunk_08a_neuron_sweep.rs` (6 tests) reads the
  `<!-- section: neuron -->` region and the capacity table, and checks five
  things:
  - the `SWEPT` rows are not `pending`;
  - every production capacity-regex hit is cited by symbol, with a pinned
    precondition;
  - the `try_fold` row reads `yes (per focus target)`;
  - the `mod.rs` outcome carries the `RecordCache` verdict;
  - every linked issue appears under `## Issues filed`.

  A sixth test proves the production-source cut skips the test-only
  `#[cfg(test)] fn` in `post_processing.rs`.
- `cargo test --test issue_2300_chunk_08a_neuron_sweep`: 6/6 pass.
  `cargo test --test issue_2088_sweep_ledger_contract`: 9/9 pass.
