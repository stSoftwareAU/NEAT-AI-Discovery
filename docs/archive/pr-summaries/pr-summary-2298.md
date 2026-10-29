## Summary

Audits the first six `per-neuron-b` detectors in `src/analysis/detection/` (`operating_point.rs`, `oscillating_neuron.rs`, `output_range_compression.rs`, `output_squash_mismatch.rs`, `restricted_range.rs`, `saturation.rs`, 2,132 lines) against the six #2092 defect classes, in the staged ledger `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, and creates the per-neuron-b sweep test. Closes #2298.

- **Rows:** the six rows now carry a non-`pending` `<outcome> — <reason>`. `output_squash_mismatch.rs` is `finding #2381`; the other five are `clean` for the six classes, four of them with sibling sites. The remaining six rows stay `pending` for #2299.
- **Capacity and traversal table:** 8 new capacity rows (operating_point 1, oscillating_neuron 2, output_range_compression 1, output_squash_mismatch 1, restricted_range 1, saturation 2), each naming its bound.
- **Findings:** #2381 was filed (`security`, `lang:rust`, `severity:low`, `confidence:high`): a `NaN` `mean_error` fails the skip gate open and reaches the non-total `partial_cmp(..).unwrap_or(Ordering::Equal)` sort. It is deduplicated against #2181, which is closed and whose fix (#2303) covered only `fan_in.rs`. Sibling sites were added as comments on #2350, #2351, #2375 and #2377.
- **Contract test:** `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs` reads only the `per-neuron-b` marker region and has a six-file `SWEPT` const. It asserts no `pending` rows, that all 8 capacity sites are cited by symbol (with the pinned `saturation.rs` `Vec::with_capacity(candidates.len() * 2)` precondition), and that every issue linked from a swept row appears under `## Issues filed`.

Docs and tests only: no `src/` change, no `Cargo.toml` bump, no CI change, and `lib-sweep-coverage.json` is untouched.

## Evidence

This is an audit, so there is no UI.

- `cargo test --test issue_2298_chunk_08a_per_neuron_b_sweep`: 3 of 3 pass.
- `cargo test --test issue_2088_sweep_ledger_contract`: 9 of 9 pass.
- `cargo fmt --check` and `cargo clippy --tests --test issue_2298_chunk_08a_per_neuron_b_sweep -- -D warnings` are clean.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The six rows carry a non-`pending` `<outcome> — <reason>`; no other row in any section changes apart from the capacity-table and `## Issues filed` additions — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::each_swept_row_is_present_and_not_pending`. `git diff 1f3ba81 HEAD -- docs/` removes exactly the six `pending` lines and adds nothing outside those rows, the capacity table and `## Issues filed`. The six #2299 rows still read `pending`, and no added line cites code by line number — reviewer: met
- **met** — The capacity table gains exactly 8 rows, one per capacity-regex hit in the six files, and each names its bound — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::every_capacity_site_in_a_swept_file_is_cited_by_symbol`. The reviewer's grep found exactly 8 hits (1, 2, 1, 1, 1, 2), matched one-to-one by function, each with a named bound and `Cancellation-checked?` = `no` — reviewer: met
- **met** — Every `as` cast on a creature-derived count in the six files is listed with its bound and a narrowing verdict — evidence: the ledger rows record 0, 4, 0, 15, 0 and 1 casts, all `usize as f32` and none narrowing. The reviewer counted per occurrence (`grep -c` shows 13 for `output_squash_mismatch.rs` because two lines hold two casts each), and the ten-cast breakdown matches the source site by site — reviewer: met
- **met** — The `oscillating_neuron.rs` subtraction, the `saturation.rs` `* 2` and the `output_squash_mismatch.rs` comparator each have a recorded verdict — evidence: the subtraction is ruled out by the `records.len() < MIN_SAMPLES_FOR_OSCILLATION` `continue` (`MIN_DISCOVERY_SAMPLE_COUNT` = 20). The `* 2` is ruled out by the `isize::MAX / size_of::<SaturatedNeuronCandidate>()` slice bound. The comparator is filed as #2381 after deduplication against #2181 — reviewer: met
- **met** — Shared-utility behaviour is cited from #2281, not re-adjudicated — evidence: the `oscillating_neuron.rs`, `restricted_range.rs` and `saturation.rs` rows each carry "cited from #2281, not re-adjudicated", and `## Issues filed` has a #2281 citation bullet — reviewer: met
- **partial** — Every surviving finding is filed with `security`, `lang:rust`, `severity:*` and `confidence:*`, linked from `## Issues filed`, and duplicates no open issue — evidence: #2381 is open with all four labels and is linked under `## Issues filed`. A search for `output_squash_mismatch` across all issues finds no duplicate. The sibling comments on #2350, #2351, #2375 and #2377 were confirmed posted. `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::every_issue_linked_from_a_swept_row_appears_under_issues_filed` — reviewer: partial — reason: step 6 asks for a `tests/issue_2381_*.rs` that fails before the fix, and none is on the branch. The issue's own Failure Detection section defers that test to the finding's eventual fix.
- **met** — `git diff --exit-code origin/Develop -- docs/audits/lib-sweep-coverage.json` shows no change, and no top-level `docs/audits/security-sweep-chunk-08a-*` file exists — evidence: against the PR base (`1f3ba81`, `origin/milestone/2083-security-scan-overflow-8-chunks-not-reached`) the diff exits 0, and `ls docs/audits/ | grep security-sweep-chunk-08a` finds nothing — reviewer: met — reason: against the literal `origin/Develop` the file differs, but that difference already exists between Develop and the milestone base. This branch does not touch the file.
- **partial** — `cargo test --test issue_2298_chunk_08a_per_neuron_b_sweep`, `cargo test --test issue_2088_sweep_ledger_contract` and `./quality.sh < /dev/null` pass — evidence: 3/3 and 9/9 pass — reviewer: partial — reason: the reviewer did not re-run `./quality.sh`, so that part rests on the worker's quality gate, which passed.

No change in the diff is unrequested: it touches only the staged ledger and the new sweep test.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — Ledger consistency — evidence: `docs/audits/in-progress/security-sweep-chunk-08a-detection-neuron.md`, the `operating_point.rs`, `output_range_compression.rs`, `restricted_range.rs` and `output_squash_mismatch.rs` rows — reason: stands. Four uncancellable walks appear only in row prose, with no `pairwise loop` row, although the capacity table's preamble promises every pairwise loop in scope and the per-neuron-a section gave rows to the same shapes. The walks are the three `*_to_coordinated_candidates` `creature.synapses` filters (#2350 siblings) and the per-output `neuron_records.iter().find` (#2377 sibling). The rows follow the issue's "gains exactly 8 rows" literally and say so.
- **violation** — Accurate documentation — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs::production_source` doc comment — reason: stands. It says some `SWEPT` files carry a `#[cfg(test)]` module, but none of the six does. The comment was carried over from the per-neuron-a sibling. The code is harmless.
- **violation** — DRY / KISS (low severity) — evidence: `tests/issue_2298_chunk_08a_per_neuron_b_sweep.rs` helpers `read`, `section`, `table_rows`, `production_capacity_sites`, `collect_issue_refs` and others — reason: stands. About 230 lines are copied from `tests/issue_2284_chunk_08a_per_neuron_a_sweep.rs` and `tests/issue_2282_chunk_08a_graph_sweep.rs` rather than shared through `tests/common/`. The issue asked for a sibling-modelled test, and the sweep tests all follow this pattern.
- **violation** — Factual wording (minor) — evidence: the `output_squash_mismatch.rs` and `oscillating_neuron.rs` ledger rows — reason: stands, with no verdict affected:
  - The `output_squash_mismatch.rs` row's "finite errors whose `f32` sum overflows give a `+∞` gain" cites a reproduction (`[inf]`) that uses a non-finite error.
  - The same row leaves out that Strategy 4 also fails closed.
  - The `oscillating_neuron.rs` row lists "`records.len()` twice" where the second cast is `(records.len() - 1) as f32`. The count of four is correct.
- **clean** — The reviewer checked `CONTRIBUTING.md` and `AGENTS.md` (there is no `CODING-STANDARDS.md`) and found:
  - Every cited symbol exists.
  - The panic-site, cast-count, division-guard, `records.len() - 1 ≥ 19` and `* 2` claims are correct, as are the overflow thresholds and the `NaN` behaviour claims.
  - Every comparator claim is correct.
  - The Lines column matches `wc -l` at the audit HEAD.
  - #2381's labels match, and every linked issue exists.
  - Australian English is used throughout, and code is cited by symbol with no line numbers (Issue #1942).
  - There are no Mermaid blocks and no `.github`, `src/` or `Cargo.toml` change.
  - The pinned #1799 preconditions are present, and test names describe their assertions.
  - `cargo fmt --check` and `cargo clippy --tests --test issue_2298_chunk_08a_per_neuron_b_sweep -- -D warnings` exit 0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
