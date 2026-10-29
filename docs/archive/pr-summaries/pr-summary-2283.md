## Summary

Completes the chunk 8a `graph` section of the staged ledger `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md` by auditing the last four graph detectors (`redundant_path.rs`, `bottleneck.rs`, `low_impact_neuron.rs`, `cross_detection_synthesis.rs`), and extends the graph-sweep contract test to all 8 files. Closes #2283.

- **Rows:** all 8 `graph` rows now carry a non-`pending` outcome with a reason. The rows are `finding #2374` (`redundant_path.rs`), `finding #2375` (`bottleneck.rs`) and `clean` for `low_impact_neuron.rs` and `cross_detection_synthesis.rs`. The four rows #2282 filled were rewritten so they describe the defects #2367–#2370 were actually filed for.
- **`redundant_path` verdicts:**
  - **P²:** the sole caller is `candidate_selection.rs::detect_redundant_path_candidates`. P is capped only when `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET` is set (via `apply_source_budget`), so P² is filed as #2374.
  - **Casts:** a recount finds 12 production lines holding 13 casts: 8 lossless widenings, `n_samples as f64`, and 4 `f64`→`f32` narrowings.
  - **Indexing:** `compute_error_gradient_product` and `estimate_pruning_improvement` cannot index out of range, because both loop to `min(len_a, len_b)`.
  - **`pearson_correlation_samples`:** cited to #2281, not re-adjudicated.
- **`bottleneck.rs`:** the `candidates.len() * 2` multiply cannot overflow `usize`, because there is at most one candidate per hidden neuron.
- **Capacity and traversal table:** 12 capacity rows (one per regex hit across the 8 files), 5 traversal rows (adding `detect_redundant_paths`), and the pairwise-loop rows for `redundant_path.rs` and `bottleneck.rs`.
- **Findings filed:** #2374 and #2375. Both carry `security`, `lang:rust`, `severity:*` and `confidence:high`, and #2374 specifies a ratio test.
- **Sibling sites:** added as comments on #2350, #2351 and #2359.
- **Cancellation:** none of the 8 files calls `deadline_passed` or `is_cancelled`.
- **Contract test:** `tests/issue_2282_chunk_08a_graph_sweep.rs` extends `SWEPT` to 8 files and `TRAVERSAL` with `detect_redundant_paths`, and adds `graph_section_holds_exactly_the_eight_swept_rows`.

Docs and tests only: no `Cargo.toml` bump, no CI change, and `lib-sweep-coverage.json` is untouched.

## Evidence

This is an audit, so there is no UI.

- `cargo test --test issue_2282_chunk_08a_graph_sweep`: 5 of 5 pass.
- `cargo test --test issue_2088_sweep_ledger_contract`: 9 of 9 pass.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — All 8 graph rows carry a non-`pending` outcome with a one-line reason, and the diff changes no ledger row outside the graph section, the capacity table and `## Issues filed`, and not `lib-sweep-coverage.json` — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::graph_section_holds_exactly_the_eight_swept_rows`, `::each_swept_row_is_present_and_not_pending`; the diff has 3 hunks, one per allowed section — reviewer: met
- **met** — The capacity table holds exactly one row per regex hit across all 8 files (12) and at least 5 traversal rows, all with filled `Bound` and `Cancellation-checked?` cells — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::every_capacity_site_in_a_swept_file_is_cited_by_symbol`, `::every_traversal_symbol_has_a_bounded_and_cancellation_checked_row`; reviewer recount 1+2+4+0+1+2+2+0 = 12 — reviewer: met
- **met** — The `redundant_path` P² verdict, the recounted cast verdict and the indexing verdict are recorded, and #2281 is cited for `pearson_correlation_samples` — evidence: ledger `redundant_path.rs` row, #2374; cast recount checked against source (12 lines, 13 casts) — reviewer: met
- **met** — The `candidates.len() * 2` overflow verdict is recorded — evidence: ledger `bottleneck.rs` row and the `bottleneck.rs::bottleneck_neurons_to_coordinated_candidates` (`results`) capacity row — reviewer: met
- **met** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*`, `confidence:*` (complexity findings specify a ratio test) and linked from `## Issues filed` — evidence: #2374 (ratio test `pair_scan_grows_linearly_with_fan_in`), #2375, sibling comments on #2350, #2351 and #2359, `::every_issue_linked_from_a_swept_row_appears_under_issues_filed` — reviewer: met
- **partial** — `cargo test --test issue_2282_chunk_08a_graph_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass, and the #2092 comment is posted — evidence: 5/5 and 9/9 pass — reviewer: partial — reason: the #2092 graph-section comment is not posted yet, because the issue says to post it after the PR is raised.
- **unrequested** — The four rows #2282 filled (topology, skip_connection, dead_neuron, compound_degradation) and their #2367–#2370 entries under `## Issues filed` were rewritten — reviewer: unrequested — reason: the old text described those findings differently from how they were filed (for example, #2370 as `documentation` and #2367 as `severity:low`). The rewrite stays inside the edit boundary and matches the ledger to the live issues.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — The test is named for more than it asserts: it checks `found.len()` on a `BTreeSet`, so a duplicated row collapses and still passes, and it never checks `rows.len()` — evidence: `tests/issue_2282_chunk_08a_graph_sweep.rs::graph_section_holds_exactly_the_eight_swept_rows` — reason: stands, not fixed here because this retry is limited to the summary. `::each_swept_row_is_present_and_not_pending` does catch a duplicate.
- **violation** — Redundant assertions (CONTRIBUTING "Avoid Over-engineering"): the set-equality check already implies the length check, and the pending loop repeats the first test — evidence: `::graph_section_holds_exactly_the_eight_swept_rows` — reason: minor, stands.
- **violation** — No six-defect-class check (sibling `issue_2294::each_swept_row_states_all_six_defect_classes_probed`), so the `cross_detection_synthesis.rs` row lacks a `**quadratic**` label and the `skip_connection.rs` row still lacks a `**recursion**` label — evidence: ledger rows for `cross_detection_synthesis.rs` and `skip_connection.rs` — reason: stands, not fixed here.
- **violation** — The Outcome cell is still not required to carry `— <reason>` (sibling idiom `split_once(" — ")`); all 8 rows comply today — evidence: `::each_swept_row_is_present_and_not_pending` — reason: stands, not fixed here.
- **violation** — The new `bottleneck.rs` precondition is pinned on raw source text with two independent `contains` calls, not on detector output (Issue #1799) — evidence: `::every_capacity_site_in_a_swept_file_is_cited_by_symbol` — reason: stands, not fixed here.
- **violation** — The test is named for more than it asserts: it only checks that the `Bound` and `Cancellation-checked?` cells are non-empty, and it now covers a fifth symbol — evidence: `::every_traversal_symbol_has_a_bounded_and_cancellation_checked_row` — reason: stands, not fixed here.
- **violation** — The one i<j loop in `redundant_path.rs::detect_redundant_paths` is recorded twice, as both a `traversal` row and a `pairwise loop` row, which the `TRAVERSAL` doc comment's "or pair scan" wording allows — evidence: `TRAVERSAL` doc comment and the ledger's two `detect_redundant_paths` rows — reason: minor, stands.
- **violation** — Ledger attribution: the `topology.rs` `total_err` and `compound_degradation.rs` `baseline_error_sq` #2351 sibling sites are credited to #2283, but #2282 recorded them first — evidence: the #2351 entry under `## Issues filed` — reason: minor, stands.
- **clean** — Australian spelling in the added prose and comments. Code is cited by symbol, never by line number (Issue #1942), and about 30 spot-checked symbols all exist. The `redundant_path.rs` cast audit is exact, the Lines column matches `wc -l`, and the issue labels match the ledger. No `Cargo.toml` or `.github/workflows/ci.yml` change. `cargo clippy --tests --test issue_2282_chunk_08a_graph_sweep -- -D warnings` is clean, and the helper idiom matches the chunk 8a sibling tests. The earlier over-engineering charge against `production_source` / `split_row_cells` is resolved, since the new files exercise both.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
