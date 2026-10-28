## Summary

Fills the `<!-- section: 13a -->` regions of the chunk-13 ledger (`docs/audits/security-sweep-chunk-13-config-env.md`) for `helpers.rs`, `detection.rs`, `drought_mitigation.rs`, `observability.rs` and `mod.rs` under `src/config/`, and extends `tests/issue_2211_chunk_13_ledger_scaffold.rs` so a `pending` row or an empty validation cell in 13a fails CI. Closes #2273.

Version bump: not changed here (docs and test only); CI's `version-increment` job owns it.

## Spec

### Intent and Rationale

- The 13a region was the last chunk-13 region still reading `pending`. Filling it, and gating it, stops it reverting silently.
- Accepted-range and on-invalid cells cite the tests pinned by #2272 (`tests/issue_2272_config_small_modules.rs`). Where no test pins a value, the cell says "from the code".

### Essential Design Decisions

- Code is cited by symbol, not line number (the record's rule), so `drought_mitigation.rs:92-106` is cited as `log_effective_drought_mitigation_config`.
- The `13a-findings` entry (class findings) stays `pending` on purpose: it belongs to #2276. The status text says so, and the new tests read only the exact `13a` marker.
- Probe (a) links the existing #2438 (filed by #2272); no new issue. Probe (b) is a negative result. #2213 / #2276 are linked on the `TIMING`, `GPU_METRICS` and `STRICT_CANDIDATE_RECONCILIATION` rows, not refiled.
- Beyond the issue: one comment on #2125 carrying the 13a doc drift (as 13b to 13d did), and the `## Probe dispositions (13a)` section.

### Undiscoverable Facts

- `docs/CONFIGURATION.md`, `mod.rs` and several doc comments say `TIMING`, `GPU_METRICS` and `PROFILE` write to stderr. The code emits `tracing::debug!` / `info!`, which the default `RUST_LOG=warn` hides.
- The "parent's list" of `mod.rs` doc-table drift (the 15 variables in #2121) was re-fetched and re-checked: all 15 are absent from the `mod.rs` tables, no other live variable is, and only `MODULE_TIERING_HIDDEN_THRESHOLD` is read by a 13a file. The other 14 have settings rows in 13b, 13c and 13d.

## Evidence

Backend/docs only. Comments posted: [#2096](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2096#issuecomment-6472996686), [#2125](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2125#issuecomment-6472994227).

## Test Plan

- Added in `tests/issue_2211_chunk_13_ledger_scaffold.rs`: `region_13a_files_swept_rows_are_decided`, `region_13a_settings_rows_cover_the_twelve_variables_with_filled_cells`, `region_13a_issues_filed_and_doc_drift_are_recorded_without_pending`, `region_13a_probe_dispositions_record_probe_a_and_probe_b`, `pending_detection_matches_standalone_word_only`, `region_text_stops_at_the_next_marker_bold_heading_or_heading`. The top doc comment was reworded (it said regions start empty).
- Red checks: `helpers.rs` row reverted to `pending` → `region_13a_files_swept_rows_are_decided` red. `TIMING` row "on invalid input" cell blanked → `region_13a_settings_rows_cover_the_twelve_variables_with_filled_cells` red. Both restored.
- Existing marker test `each_section_marker_appears_exactly_once_in_its_section` stayed green once the status paragraph no longer contains the literal `## Issues filed` (the helper matches the first occurrence of a heading string).
- Run: `cargo test --test issue_2211_chunk_13_ledger_scaffold --test issue_2124_chunk_13d_ledger --test issue_2088_sweep_ledger_contract` → 12, 10 and 9 passed. `./quality.sh` → passed.
- Not run: nothing skipped.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — All five 13a Files swept rows show an outcome and a reason. — evidence: `tests/issue 2211 chunk 13 ledger scaffold.rs::region 13a files swept rows are decided; docs/audits/security-sweep-chunk-13-config-env.md small modules rows (helpers, detection, drought mitigation, observability, mod.rs), each swept (Issue #2273) — <reason>` — reviewer: met
- **met** — There are 12 settings rows, each with all 10 columns filled. — evidence: `tests/issue 2211 chunk 13 ledger scaffold.rs::region 13a settings rows cover the twelve variables with filled cells (passes on the branch)` — reviewer: met
- **met** — The probe (a) and probe (b) outcomes are recorded in the ledger. — evidence: `docs/audits/security-sweep-chunk-13-config-env.md Probe dispositions (13a) ; tests/issue 2211 chunk 13 ledger scaffold.rs::region 13a probe dispositions record probe a and probe b` — reviewer: met
- **met** — MODULE TIERING HIDDEN THRESHOLD appears under Doc drift observed. — evidence: `tests/issue 2211 chunk 13 ledger scaffold.rs::region 13a issues filed and doc drift are recorded without pending (asserts the 13a region of Doc drift observed names it)` — reviewer: met
- **met** — The #2213 and #2272-finding links appear under Issues filed, and the #2096 comment has been posted. — evidence: `tests/issue 2211 chunk 13 ledger scaffold.rs::region 13a issues filed and doc drift are recorded without pending (asserts #2438, #2213 and #2096 in the 13a Issues filed region); comment https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/2096 issuecomment-6472996686 resolves via the GitHub API` — reviewer: met
- **met** — The extended scaffold test fails on a re-inserted pending row, and passes on the finished ledger. — evidence: `tests/issue 2211 chunk 13 ledger scaffold.rs: region 13a files swept rows are decided and the settings-rows test assert !has standalone pending(..) on every 13a row and cell; cargo test --test issue 2211 chunk 13 ledger scaffold gives 12 passed, 0 failed on the finished ledger. I did not re-run the` — reviewer: met
- **partial** — ./quality.sh passes. — evidence: `tests/issue 2211 chunk 13 ledger scaffold.rs (12 passed on this tree)` — reviewer: partial — reason: I did not run ./quality.sh this turn; the PR summary records a full gate run before the final wording edits, and only the three ledger test targets were re-run after them.

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **clean** — Checked against CONTRIBUTING.md, because the repo has no CODING-STANDARDS.md: Australian English, code cited by symbol, no new dead levers or config surface, no FFI or invariant change (docs and test only), a ledger test that fails loudly with named rows and cells, and no vacuous assertion. One repo
