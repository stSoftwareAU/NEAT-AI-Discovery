## Summary

Partial work on #2299 (chunk 8a per-neuron-b-2). **This branch does not complete the issue and must not close it.**

The only change since #2298 (`d8b7b84`) is the extension of `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs` to cover all 12 `per-neuron-b` files:

- `SWEPT` now lists all 12 files.
- The test asserts 12 non-`pending` rows and 17 capacity-regex hits.
- A new `TRAVERSAL` list and the test `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` require one `traversal` row each for `topology_diversification.rs::has_unhealthy_intermediates` and `topology_diversification.rs::dfs_max_hidden_depth`.
- That test also pins the `fn dfs_max_hidden_depth` precondition.

The ledger itself, `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, has **not** been edited:

- The six rows for `sentinel_gating.rs`, `squash_weight_rescale.rs`, `topology_diversification.rs`, `unbounded_capping.rs`, `weight_magnitude_reset.rs` and `weight_polarity_flip.rs` still read `pending`.
- There are no new capacity or traversal rows.
- There are no new `## Issues filed` entries.

As a result, the extended sweep test currently fails. Finding #2383 was filed on GitHub for `dfs_max_hidden_depth`, but neither the ledger nor a `tests/issue_2383_*.rs` reproduction on this branch records it.

Docs and tests only: no `src/` change, no `Cargo.toml` bump, no CI change, and `lib-sweep-coverage.json` is untouched.

## Evidence

This is an audit, so there is no UI.

- `cargo test --test issue_2298_chunk_08a_per_neuron_b_sweep`: **1 passed, 3 failed**.
  - `each_swept_row_is_present_and_not_pending` fails: "sentinel_gating.rs … must not read `pending`".
  - `every_capacity_site_in_a_swept_file_is_cited_by_symbol` fails: `detect_sentinel_gating_candidates` / `candidates` is not cited.
  - `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` fails: "got 0" for `has_unhealthy_intermediates`.
  - `every_issue_linked_from_a_swept_row_appears_under_issues_filed` passes, but only vacuously.
- `cargo test --test issue_2088_sweep_ledger_contract`: 9 of 9 pass.
- `cargo fmt --check` and `cargo clippy --tests --test issue_2298_chunk_08a_per_neuron_b_sweep -- -D warnings` are clean.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **missing** — The six rows carry a non-`pending` `<outcome> — <reason>`, and all 12 `per-neuron-b` rows are non-`pending`. No other row changes — reviewer: missing — reason: `git diff d8b7b84..HEAD -- docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` is empty, and all six rows still read `pending`.
- **missing** — The capacity table gains exactly 9 capacity rows plus the two `topology_diversification.rs` traversal rows, and each names its bound — reviewer: missing — reason: the 9 regex hits exist in source (2/1/1/2/1/2), but the table has no rows for any of the six files and no traversal rows.
- **missing** — `dfs_max_hidden_depth` has a recorded recursion-depth verdict — reviewer: missing — reason: #2383 was filed on GitHub (`severity:medium`, `confidence:high`), but no ledger verdict records it and no stack-overflow reproduction binary `tests/issue_2383_*.rs` is on the branch.
- **missing** — The `weight_polarity_flip.rs` NaN-weight division and the two `unwrap_or(Ordering::Equal)` comparators each have a recorded verdict, deduplicated against #2181 and #2298's findings — reviewer: missing — reason: no verdict is recorded anywhere, and #2381 is never cross-referenced.
- **missing** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*` and `confidence:*`, and linked from `## Issues filed` — reviewer: missing — reason: #2383 carries the four labels but is not linked from `## Issues filed` and has no failing-before-fix test.
- **partial** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change, and no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — evidence: against the PR base `origin/milestone/2083-security-scan-overflow-8-chunks-not-reached` the diff exits 0, and `ls docs/audits | grep chunk-08a` finds nothing — reviewer: partial — reason: the literal command against `origin/Develop` exits 1. The difference comes from the milestone base, not from this branch.
- **missing** — `cargo test --test issue_2298_chunk_08a_per_neuron_b_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: `issue_2088_sweep_ledger_contract` passes 9/9 — reviewer: missing — reason: the per-neuron-b sweep test fails 3 of 4, so `./quality.sh` cannot pass. The reviewer did not run `./quality.sh`.

No change in the diff is unrequested. The test extension implements item 7 of the issue's "What Needs to Be Done".

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — Ledger consistency and completeness — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` (the `per-neuron-b` section and `## Capacity and traversal table`) against `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::SWEPT` / `TRAVERSAL` — reason: stands. The test demands ledger content that was never written, so 3 of 4 sweep tests fail, and `./quality.sh` and CI would fail too.
- **violation** — Accurate comments — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::TRAVERSAL` doc comment — reason: stands. The comment says `has_unhealthy_intermediates` "walks the graph via `dfs_max_hidden_depth`". In fact it runs its own iterative `queue` / `visited` walk. `dfs_max_hidden_depth` is called only from `max_hidden_depth_to_output`, so these are two independent walks.
- **violation** — Accurate test naming — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` — reason: stands. The test only checks that the Bound and `Cancellation-checked?` cells are non-empty. A correct ledger would record `no` for both walks, so the name claims more than the test asserts.
- **violation** — DRY / #1799 precondition placement — evidence: the `fn dfs_max_hidden_depth` precondition in `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` — reason: stands. It duplicates the `TRAVERSAL` loop's own `fn {name}` check, and its "otherwise this test checks nothing" message is inaccurate. The issue wants this pin to keep the 17-hit capacity check from passing vacuously, which means it belongs in `every_capacity_site_in_a_swept_file_is_cited_by_symbol`.
- **violation** — Accurate comments (minor) — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::each_swept_row_is_present_and_not_pending` — reason: stands. The "whole per-neuron-b section must now be non-pending" comment claims more than the test checks. It never asserts `rows.len() == 12`, so a 13th `pending` row would pass.
- **violation** — Accurate comments (stale, outside the diff) — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::production_source` doc comment — reason: stands. It says some `SWEPT` files carry a `#[cfg(test)]` module, but none of the 12 does.
- **clean** — The reviewer checked `CONTRIBUTING.md` and `AGENTS.md` (there is no `CODING-STANDARDS.md`) and found:
  - Australian English is used in all added lines.
  - Code is cited by symbol, with no line numbers (Issue #1942).
  - There is no CI, `src/` or `Cargo.toml` change.
  - `SWEPT` lists all 12 files, and the 17-site count is correct (9 new same-line `let` bindings).
  - The updated #2298/#2299 assertion messages are accurate.
  - `cargo fmt --check` and clippy with `-D warnings` pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
