## Summary

Audits the first seven `per-neuron-a` detectors in `src/analysis/detection/` (`activation_mismatch.rs`, `bias_perturbation.rs`, `bimodal_neuron.rs`, `bounded_range.rs`, `dormant_synapse.rs`, `error_dispersion.rs`, `error_plateau.rs`, 1,766 lines) against the six #2092 defect classes, in the staged ledger `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, and creates the per-neuron-a contract test. Closes #2284.

- **Rows:** all seven rows now carry a non-`pending` `<outcome> — <reason>`. `bias_perturbation.rs` and `error_plateau.rs` are `finding #2377`; the other five are `clean` for the six classes, some with sibling sites. The six follow-on rows stay `pending`.
- **`detect_error_plateaus` comparator:** ruled out and deduplicated against #2181. The sort uses `partial_cmp(..).unwrap_or(Ordering::Equal)`, but `assess_error_plateau` only admits a finite `estimated_improvement`, so no `NaN` can reach it. #2181 is closed and covered only `fan_in.rs`.
- **`compute_bimodality` index and slices:** ruled out. The `n < 2` and `n < 2 * min_cluster_size` guards keep the `sorted_gaps` median index in bounds. The `best_gap <= 0.0` return keeps both `values[..best_split_index]` and `values[best_split_index..]` in bounds and non-empty.
- **Shared utilities:** #2281's `stats.rs` verdict is cited for `compute_mean` / `compute_variance`, and its `helpers.rs` verdict for `sort_candidates_by_score_gain` and `build_record_map`. None of them is re-adjudicated.
- **Capacity and traversal table:** 9 capacity rows, one per production regex hit, each naming its bound. Hidden-neuron and synapse counts have no numeric cap; the `1_000_000` input/output caps do not bind the typed-neuron count. There are also 2 pairwise-loop rows for #2377.
- **Findings:** #2377 was filed (`security`, `lang:rust`, `severity:low`, `confidence:high`, with an N vs 4N ratio test). Sibling sites were added as comments on #2351, #2375 and #2359.
- **Contract test:** `tests/issue_2284_chunk_08a_per_neuron_a_sweep.rs`, modelled on the chunk 8a sibling tests.

Docs and tests only: no `src/` change, no `Cargo.toml` bump, no CI change, and `lib-sweep-coverage.json` is untouched.

## Evidence

This is an audit, so there is no UI.

- `cargo test --test issue_2284_chunk_08a_per_neuron_a_sweep`: 3 of 3 pass.
- `cargo test --test issue_2088_sweep_ledger_contract`: 9 of 9 pass.
- Sibling ledger tests still pass after the edit: `issue_2280` (5), `issue_2281` (4), `issue_2282` (6), `issue_2294` (9) and `issue_2300` (10).
- `cargo clippy --tests --test issue_2284_chunk_08a_per_neuron_a_sweep -- -D warnings` is clean.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The seven rows carry a non-`pending` `<outcome> — <reason>`; all other rows are unchanged — evidence: `tests/issue_2284_chunk_08a_per_neuron_a_sweep.rs::each_swept_row_is_present_and_not_pending`. The ledger diff has three hunks: the seven rows, the capacity table and `## Issues filed`. The six follow-on rows are still `pending` — reviewer: met
- **met** — The capacity table has one row per capacity-regex hit in the seven files, each naming its bound — evidence: `tests/issue_2284_chunk_08a_per_neuron_a_sweep.rs::every_capacity_site_in_a_swept_file_is_cited_by_symbol`, with the `Vec::with_capacity(neurons.len())` precondition pinned. The reviewer recounted 1+1+2+2+2+0+1 = 9 production hits — reviewer: met
- **met** — The `detect_error_plateaus` comparator and `compute_bimodality` index/slice sites each have a recorded verdict: filed/linked issue, or ruled out with evidence cited by symbol — evidence: the ledger's `error_plateau.rs` row (ruled out, deduplicated against #2181) and `bimodal_neuron.rs` row (ruled out). The reviewer re-derived both from source — reviewer: met
- **met** — Shared-utility behaviour is cited from #2281, not re-adjudicated — evidence: the ledger's `error_dispersion.rs` row (`stats.rs` verdict), `bounded_range.rs` row (`helpers.rs` `sort_candidates_by_score_gain`) and the `build_record_map` citations, plus the #2281 cross-reference under `## Issues filed` — reviewer: met
- **met** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*`, `confidence:*` and linked from `## Issues filed`; none duplicates an existing open issue — evidence: #2377 has all four labels and an N vs 4N ratio test, and is distinct from #2350, #2358 and #2369. The sibling comments on #2351, #2375 and #2359 were confirmed posted. `tests/issue_2284_chunk_08a_per_neuron_a_sweep.rs::every_issue_linked_from_a_swept_row_appears_under_issues_filed` — reviewer: met
- **met** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change; no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — evidence: against the PR base `origin/milestone/2083-security-scan-overflow-8-chunks-not-reached` the diff exits 0, and there is no top-level chunk-08a file — reviewer: met — reason: the literal `origin/Develop` check exits 1 because of chunk 9/11 entries inherited from the milestone base. This branch does not touch the file.
- **met** — `cargo test --test issue_2284_chunk_08a_per_neuron_a_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: 3/3 and 9/9 pass — reviewer: met — reason: the reviewer did not re-run `./quality.sh`. It relies on the worker's quality gate, which passed.
- **unrequested** — Two `pairwise loop` rows added to the capacity table for the `iter().find` scans in `bias_perturbation.rs` and `error_plateau.rs` — reviewer: unrequested — reason: these fall under step 3's O(n²) traversal audit and back finding #2377. The contract test counts only `capacity` rows, so they do not affect it.
- **unrequested** — A closing note in `## Issues filed` saying the remaining six rows stay `pending`, and a #2152 cross-reference — reviewer: unrequested — reason: informational only, inside the section the issue allows editing.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — The reviewer checked `CONTRIBUTING.md` and `AGENTS.md` and found:
  - The added prose uses Australian English.
  - Code is cited by symbol, never by line number (Issue #1942).
  - Every spot-checked cited symbol exists, including the dispatch record-loading claims and the sole `error_dispersion` caller.
  - The Lines column matches `wc -l` for all seven files.
  - The cited issue states and titles match the ledger.
  - There are no vacuous passes: the pinned precondition, the `is_capacity_site` self-test, non-empty assertions, and row count equal to site count (Issue #1799).
  - The test names describe what they assert.
  - The idiom matches `tests/issue_2282_chunk_08a_graph_sweep.rs`, and the test adds no over-engineering.
  - There is no CI or Mermaid change.
  - `cargo clippy --tests --test issue_2284_chunk_08a_per_neuron_a_sweep -- -D warnings` is clean.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
