# PR Summary — Issue #2191

## Summary

Closes #2191

`deduplicate_by_dominant_neuron` and `deduplicate_synergistic_by_dominant_neuron`
grouped candidates in a `HashMap` (fresh random seed per instance) and then
stable-sorted on `combined_improvement` alone. Tied candidates therefore kept
hash-seed order, both in the final output and in which members a group of more
than three kept after `truncate(3)`, so discovery runs were not reproducible.

Both deduplicators now sort with a total comparator: `combined_improvement`
descending via `total_cmp`, then the candidate UUIDs
(`source_a`, `source_b`, `target` for epistatic; `primary`, `complement`,
`target` for synergistic). One shared comparator per type is used for both the
per-group sort and the final sort.

- [x] Total comparators `cmp_epistatic` / `cmp_synergistic` in
      `src/analysis/recommendation/epistatic/deduplication.rs`
- [x] Applied to the per-group sort (before `truncate(3)`) and the final sort
- [x] Regression test `tests/issue_2191_dominant_neuron_dedup_determinism.rs`
- [x] `./quality.sh` gate

## Evidence

Backend-only change; the evidence is the regression suite.

```mermaid
flowchart LR
    A[candidates] --> B[HashMap group by dominant neuron]
    B --> C[per-group sort: improvement desc, then UUIDs]
    C --> D[truncate to 3]
    D --> E[final sort: improvement desc, then UUIDs]
    E --> F[identical output every run]
```

- Before the fix, all five tests in
  `tests/issue_2191_dominant_neuron_dedup_determinism.rs` failed; the
  determinism tests failed with `run 1 ordered tied candidates differently from
  run 0`, and the big tied group kept `p-e, p-b, p-d` (input order) instead of
  `p-a, p-b, p-c`.
- After the fix all five pass, and the existing Issue #509 suite
  (`tests/recommendation/issue_509_deduplicate_dominant_neuron.rs`) still
  passes (`cargo test --test recommendation`: 249 passed).

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- Epistatic comparator is total (improvement desc, then
  `source_a_uuid`, `source_b_uuid`, `target_uuid`) — reviewer: met —
  evidence: `cmp_epistatic` in `deduplication.rs`.
- Synergistic comparator is total (improvement desc, then
  `primary_source_uuid`, `complement_source_uuid`, `target_uuid`) — reviewer:
  met — evidence: `cmp_synergistic` in `deduplication.rs`.
- Applied to the per-group sort before `truncate(3)` and to the final sort, in
  both functions — reviewer: met — evidence: the four `sort_by(cmp_*)` call
  sites; `*_is_independent_of_input_order` tests pin which tied members survive
  truncation.
- Test runs both deduplicators many times over tied candidates, including a
  group larger than 3, and asserts an identical sequence every run — reviewer:
  met — evidence: `epistatic_dedup_orders_tied_candidates_identically_on_every_run`
  and `synergistic_dedup_orders_tied_candidates_identically_on_every_run`
  (64 runs, nine groups, one of five), pinned to the expected UUID order.
  Note: the fixture uses a single `target_uuid`, so the final `target_uuid`
  tie-break is not independently exercised.
- Flip `tests/issue_2109_chunk_08b_batch_successful_epistatic_sweep.rs::dominant_neuron_dedup_still_orders_tied_candidates_non_deterministically`
  — reviewer: missing — reason: that file does not exist in this tree, so there
  is no test to flip; nothing was deleted. The new Issue #2191 test covers the
  behaviour it would have asserted.
- Sweep ledger
  (`docs/audits/security-sweep-chunk-08b-synapse-scoring-recommendation.md`,
  "recommendation batch_successful + epistatic") — reviewer: missing — reason:
  deliberate departure; every file in that section is still "pending" a
  whole-file sweep, and marking the row here would claim a sweep that has not
  been done.
- Input-order independence tests and `higher_combined_improvement_still_ranks_first`
  — reviewer: unrequested — reason: small guards that the tie-break does not
  override the primary improvement ordering and that truncation keeps the same
  members regardless of input order.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

The repo has no `CODING-STANDARDS.md`; the review used `CONTRIBUTING.md` and
`AGENTS.md`.

Violations: none in the changed lines. The one DRY nit raised (the singleton
list built twice in the test) was fixed by a shared `singleton_ids()` helper.

Clean areas:

- Australian English (codespell clean).
- Testing doctrine — real public API calls, observable output, no source
  grepping, non-vacuous assertions, no sleeps or timing.
- Scope/KISS/DRY — one comparator per type, reused at both sort sites.
- No `Cargo.toml` version bump (CI does it); CI workflow and dependencies
  untouched.

## Test Plan

- [x] `cargo test --test issue_2191_dominant_neuron_dedup_determinism` — red
      before the fix, green after
- [x] `cargo test --test recommendation` — 249 passed
- [x] `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
- [x] `./quality.sh < /dev/null`
