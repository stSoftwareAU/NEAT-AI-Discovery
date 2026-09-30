## Summary

Completes #2299 (chunk 8a per-neuron-b-2). A prior commit on this branch
extended `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs` to cover all 12
`per-neuron-b` files but left the ledger itself unedited, so 3 of 4 tests in
that file failed CI (`Coverage`). This commit finishes the audit the test
demands.

`docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` now
carries, for `sentinel_gating.rs`, `squash_weight_rescale.rs`,
`topology_diversification.rs`, `unbounded_capping.rs`,
`weight_magnitude_reset.rs` and `weight_polarity_flip.rs`:

- A non-`pending` `<outcome> — <reason>` row auditing all six #2092 defect
  classes (allocation, recursion, quadratic/exponential blowup, panic sites,
  integer overflow, cache poisoning) plus NaN/comparator and division safety.
- 9 new `## Capacity and traversal table` capacity rows (2/1/1/2/1/2 per
  file, matching every `production_capacity_sites` hit in source) and 2 new
  traversal rows for `topology_diversification.rs::has_unhealthy_intermediates`
  and `::dfs_max_hidden_depth`.
- A new `## Issues filed` entry for `#2383` (already filed on GitHub:
  `dfs_max_hidden_depth` enumerates every simple path recursively —
  exponential time and a stack-overflow DoS on a deep hidden chain), plus
  sibling-site comments on the existing #2350, #2375, #2377, #2351 and #2359
  findings for shapes repeated in these six files, and a dedup note against
  #2181 for two non-total `partial_cmp(..).unwrap_or(Ordering::Equal)`
  comparators that never actually see a `NaN`.

Docs only: no `src/` change, no `Cargo.toml` bump, no CI change, and
`lib-sweep-coverage.json` is untouched — consistent with the prior commit's
scope.

## Evidence

This is an audit, so there is no UI.

- `cargo test --test issue_2298_chunk_08a_per_neuron_b_sweep`: **4 passed, 0 failed** (was 1 passed, 3 failed before this commit).
- `cargo test --test issue_2088_sweep_ledger_contract`: 9 of 9 pass.
- `cargo test --test issue_2280_chunk_08a_ledger_scaffold --test issue_2281_chunk_08a_shared_sweep --test issue_2282_chunk_08a_graph_sweep --test issue_2284_chunk_08a_per_neuron_a_sweep --test issue_2294_chunk_08a_pairwise_sweep --test issue_2300_chunk_08a_neuron_sweep`: 37 of 37 pass (no regression in the other chunk-08a ledger sections).
- `cargo fmt --check`: clean (no `.rs` file changed by this commit).

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The six rows carry a non-`pending` `<outcome> — <reason>`, and all 12 `per-neuron-b` rows are non-`pending` — evidence: `each_swept_row_is_present_and_not_pending` passes; no other row in the section was touched.
- **met** — The capacity table gains exactly 9 capacity rows plus the two `topology_diversification.rs` traversal rows, and each names its bound — evidence: `every_capacity_site_in_a_swept_file_is_cited_by_symbol` passes (17 total capacity rows for the section), and both traversal rows carry non-empty Bound / Cancellation-checked? cells.
- **met** — `dfs_max_hidden_depth` has a recorded recursion-depth verdict — evidence: the `topology_diversification.rs` row and its traversal row cite `#2383` (already filed on GitHub: `severity:medium`, `confidence:high`) by name, with the exponential-time and stack-overflow bound recorded.
- **met** — The `weight_polarity_flip.rs` NaN/weight-overflow gain and the two non-total `unwrap_or(Ordering::Equal)` comparators each have a recorded verdict, deduplicated against #2181 and #2298's findings — evidence: `weight_polarity_flip.rs`'s row records a `#2351` sibling site (a finite weight overflowing `weight_change` into a `+∞` gain), and `squash_weight_rescale.rs`/`weight_magnitude_reset.rs`'s comparators are refuted against #2181 with the specific reason neither ever observes a `NaN`.
- **met** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*` and `confidence:*`, and linked from `## Issues filed` — evidence: `#2383` is added to `## Issues filed` with all four labels; `every_issue_linked_from_a_swept_row_appears_under_issues_filed` passes non-vacuously (the six new rows reference #2042, #2181, #2298, #2350, #2351, #2359, #2370, #2375, #2377, #2383, all present under `## Issues filed`).
- **unchanged, not in scope for this fix** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change, and no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — this remains true; finalisation (#2154) is a separate, later step.
- **met** — `cargo test --test issue_2298_chunk_08a_per_neuron_b_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and the other chunk-08a ledger tests pass — evidence: see Evidence above (4/4, 9/9 and 37/37).

No change in this diff is unrequested: every added sentence documents either a
required `SWEPT`/`TRAVERSAL` row or an existing-issue cross-reference the test
demands.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **fixed** — Ledger consistency and completeness — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` (the `per-neuron-b` section and `## Capacity and traversal table`) now matches `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::SWEPT` / `TRAVERSAL` — reason: this commit wrote the six missing rows, 9 capacity rows and 2 traversal rows; all 4 sweep tests pass.
- **unchanged, pre-existing (not in this commit's scope)** — Accurate comments — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::TRAVERSAL` doc comment — reason: the comment says `has_unhealthy_intermediates` "walks the graph via `dfs_max_hidden_depth`", but it runs its own independent iterative walk. This commit did not touch the test file; fixing the test's own doc comment is separate work.
- **unchanged, pre-existing (not in this commit's scope)** — Accurate test naming — evidence: `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` only checks the Bound / `Cancellation-checked?` cells are non-empty, not that they read `no`. Not touched by this commit.
- **unchanged, pre-existing (not in this commit's scope)** — DRY / #1799 precondition placement in `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row`. Not touched by this commit.
- **unchanged, pre-existing (not in this commit's scope)** — Accurate comments (minor) in `each_swept_row_is_present_and_not_pending`. Not touched by this commit.
- **unchanged, pre-existing (not in this commit's scope)** — Accurate comments (stale) in `production_source`'s doc comment. Not touched by this commit.
- **clean** — This commit:
  - Uses Australian English throughout the new ledger prose.
  - Cites code by symbol, with no line numbers (Issue #1942).
  - Touches no `src/`, `Cargo.toml`, or CI file.
  - Reuses existing issues (#2350, #2351, #2359, #2375, #2377) via sibling-site comments rather than filing duplicates, and cites the already-filed #2383 rather than re-filing it.
  - `cargo fmt --check` passes (no `.rs` file changed).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
