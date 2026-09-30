## Summary

Partial sweep of the four traversal-heavy chunk 8a graph detectors (`topology.rs`, `skip_connection.rs`, `dead_neuron.rs`, `compound_degradation.rs`) into the staged ledger `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, plus the new graph-sweep contract test. Refs #2282. **This change is incomplete and does not close the issue.** The contract test it adds fails 3 of its 4 tests against the ledger on this branch (see the Acceptance Criteria below).

- [x] **Rows:** the four `graph` rows no longer read `pending`. Each reads `finding #NNNN — …` and walks the #2092 defect classes. The other four graph rows, every other section and `lib-sweep-coverage.json` are untouched.
- [ ] **Capacity and traversal table:** no rows were added for these four files. The 7 capacity-regex hits (topology 1, skip_connection 2, dead_neuron 4, compound_degradation 0), the 4 traversal rows and the ≥4 pairwise-loop rows are all still missing.
- [x] **Findings filed:** #2367 (`compound_degradation.rs`), #2368 (`skip_connection.rs`), #2369 (`topology.rs`) and #2370 (`dead_neuron.rs`) are open. Each has the labels `security`, `lang:rust`, `severity:*` and `confidence:high`, and each specifies an N vs 4N ratio test. The NaN/`+∞` gain sites were added to #2351 as a comment.
- [ ] **`## Issues filed`:** none of #2367–#2370 is listed there yet.
- [ ] **Row text vs filed issues:** three of the four rows describe a different defect from the issue they link. The topology and compound_degradation rows describe the NaN/`+∞` sites (which went to #2351), not the uncancellable scans #2369 and #2367 are titled for. The dead_neuron row records "quadratic — none" against #2370's O(D·(V+E)) finding. The skip_connection row is missing its **recursion** verdict.
- [x] **Contract test:** `tests/issue_2282_chunk_08a_graph_sweep.rs` has the `SWEPT` and `TRAVERSAL` consts and assertions (a)–(d), modelled on the chunk 8a sibling tests.

Docs and tests only: no `Cargo.toml` bump and no CI change.

## Evidence

This is an audit, so there is no UI. `cargo test --test issue_2282_chunk_08a_graph_sweep`: 1 passed, 3 failed.

- `each_swept_row_is_present_and_not_pending` — passes
- `every_capacity_site_in_a_swept_file_is_cited_by_symbol` — fails (`detect_topology_issues` `candidates` is not cited)
- `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` — fails (0 rows for `topology.rs::compute_shortest_paths_to_output`)
- `every_issue_linked_from_a_swept_row_appears_under_issues_filed` — fails (#2367 is not under `## Issues filed`)

`cargo test --test issue_2088_sweep_ledger_contract`: 9 passed.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **partial** — The four rows read `<outcome> — <reason>`, not `pending`; the other four graph rows are unchanged; the diff touches only the graph section, the capacity table and `## Issues filed` of the staged ledger, plus the new test file, never `lib-sweep-coverage.json` — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` (graph section), `tests/issue_2282_chunk_08a_graph_sweep.rs::each_swept_row_is_present_and_not_pending` — reviewer: partial — reason: the form and scope are right, but three of the four rows point their outcome at an issue that records a different defect (#2369, #2367, #2370), so the recorded outcome is not accurate.
- **missing** — Four traversal rows, at least four pairwise-loop rows, and exactly one capacity row per regex hit in these four files (7 at `f9cc777`), each with filled `Bound` and `Cancellation-checked?` cells — reviewer: missing — reason: no graph-file rows were added to `## Capacity and traversal table`; `every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` and `every_capacity_site_in_a_swept_file_is_cited_by_symbol` fail.
- **missing** — `compute_depths_from_inputs` reads `unbounded (DAG-dependent)` and cites #1184 and the `validate_creature` path by symbol; `reachable_outputs` records `max_depth = 3` and the missing visited set — reviewer: missing — reason: the ledger contains neither `DAG-dependent` nor `#1184` and gives no `validate_creature` call path. The `reachable_outputs` text sits in a file row, not a traversal row, and calls it "bounded regardless of graph size", which contradicts #2367.
- **partial** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*`, `confidence:*` (complexity findings specify a ratio test) and linked from `## Issues filed`, or a negative result is recorded — evidence: #2367, #2368, #2369, #2370 (labelled, N vs 4N ratio tests specified) and the #2351 comment — reviewer: partial — reason: none of them is listed under `## Issues filed`; `every_issue_linked_from_a_swept_row_appears_under_issues_filed` fails on #2367.
- **missing** — `cargo test --test issue_2282_chunk_08a_graph_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: `tests/issue_2088_sweep_ledger_contract.rs` (9 of 9 pass) — reviewer: missing — reason: the graph-sweep test passes only 1 of 4, so `./quality.sh` cannot pass either.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — Test weaker than its model (`issue_2105`, `issue_2281`): the Outcome cell is not required to carry a `— <reason>` — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::each_swept_row_is_present_and_not_pending` — reason: stands. A bare `clean` would pass; not fixed here because this retry is limited to the summary.
- **violation** — No six-defect-class check (sibling `issue_2294::each_swept_row_states_all_six_defect_classes_probed`), and a real gap slips through it — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` `skip_connection.rs` row, which has no **recursion** verdict — reason: stands, not fixed here.
- **violation** — Test named for more than it asserts: it checks only that the `Bound` / `Cancellation-checked?` cells are non-empty — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` — reason: stands, not fixed here.
- **violation** — The detector precondition is pinned on raw source text, not on the detector's output (Issue #1799; model `issue_2210` pins `set.contains(known)`) — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::every_capacity_site_in_a_swept_file_is_cited_by_symbol` — reason: stands, not fixed here.
- **violation** — Doc comment inaccurate: `is_capacity_site` claims to be equivalent to the regex but also skips `//` lines — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::is_capacity_site` — reason: minor, stands.
- **violation** — Over-engineering (CONTRIBUTING "Avoid Over-engineering"): `production_source` handles `#[cfg(test)] mod` forms none of the four files has, and `split_row_cells` handles `\|` escapes absent from the rows it reads — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::production_source`, `::split_row_cells` — reason: minor, stands.
- **clean** — Australian spelling in added prose and comments; code cited by symbol, never by line number (Issue #1942), with every cited symbol present in its file; no `Cargo.toml` or `.github/workflows/ci.yml` change; `cargo clippy --tests --test issue_2282_chunk_08a_graph_sweep -- -D warnings` shows no warnings; helper idiom matches the chunk 8a sibling tests.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
