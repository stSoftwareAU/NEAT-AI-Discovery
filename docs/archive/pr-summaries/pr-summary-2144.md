## Summary

Audited the Parquet write side and shared cache (`writer.rs`, `shared_records.rs`, `mod.rs`) at baseline `b85a551` and recorded every probe and per-file outcome in `docs/audits/in-progress/security-sweep-chunk-07-parquet.md`. Five findings were filed as SEC-2091-03..07 (#2449-#2453). No `src/` change and no `Cargo.toml` change, as the issue requires; the only `.rs` file added is the ledger contract test `tests/issue_2144_chunk_07c_write_side_sweep.rs`. Closes #2144.

Version bump: not applicable — this change touches no code.

## Spec

### Intent and Rationale

- Audit-only sub-issue: probes that found nothing are recorded in the ledger; probes that confirmed a defect became a filed issue carrying its own failing-first test name.
- Scratch probe tests were run locally against the unfixed code and deleted before the commit.

### Essential Design Decisions

- Ledger written to `docs/audits/in-progress/` (not top-level `docs/audits/`), per the trusted path-change comment on #2144; `tests/issue_2088_sweep_ledger_contract.rs` scans only the top level.
- Fifth cache-key component: add `ctime` (unix); a footer hash was rejected because a same-shape rewrite changes only data pages. The same-coarse-tick residual is accepted.
- Missing `fsync` in `finish`/merge, default file mode, and NaN/inf accepted on write are recorded as accepted/no finding (reasons in the ledger); the NaN/inf pass-through is cross-referenced to #2143's read-side probe so no second finding is filed.

### Undiscoverable Facts

- Observed at run time: merge of a 1,653-byte hostile file grew `VmHWM` by 600,244,224 bytes with the decode budget env at 1 MB while the reader refuses it; an input named `{output}.tmp` is truncated and deleted by merge; a symlinked `{output}.tmp` clobbers its target; an in-place rewrite with restored mtime is served stale from the cache.
- The issue's `Depends on` blockers (#2215, #2279) are resolved: the ledger now exists on the branch.

## Evidence

Backend/doc-only change; no UI. Probes were run with `cargo test --test <scratch probe> -- --nocapture` (scratch file removed). Results are recorded row by row under **Defect classes probed** in the ledger, and the filed issues are #2449 (merge decode budget, medium), #2450 (merge temp path, low), #2451 (live-file truncation in `record_discovery_data`, low), #2452 (stale cache after in-place rewrite, low), #2453 (vacuous `mod.rs` writer tests, low).

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Every one of the 1,265 lines across the three files has been read; the ledger lists all three under **Files swept** with `outcome: clean` or `outcome: finding #N` — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md` (`## Files swept` rows for `mod.rs`, `writer.rs`, `shared_records.rs`) — reviewer: met
- **met** — Every defect class above has at least one recorded probe with class, `file:line`, input and outcome in the ledger; the write-destination caller trace lists every external call site with its temp-then-rename verdict — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md` (`### Write-path atomicity` through `### Public surface`) — reviewer: met
- **met** — Every surviving finding has exactly one `SEC-2091-NN` issue with `severity:*` and `confidence:high` labels and a named failing-first test path — evidence: #2449-#2453 (each created with `severity:*` and `confidence:high` labels and a `tests/issue_<n>_*.rs` path in its body) — reviewer: met — reason: the reviewer could not see the GitHub issues; verified here with `gh issue create`/`edit`
- **met** — No `src/` change and no `Cargo.toml` change in this sub-issue's PR — evidence: `git diff --name-only` lists only the ledger, this summary and the ledger contract test (added at the security-fix gate's request) — reviewer: met
- **met** — Tests and quality checks pass (`./quality.sh`) — evidence: the gate result is in the Test Plan — reviewer: partial — reason: the reviewer could not run the gate; see Test Plan for the result run on the final head

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — MD012 two consecutive blank lines in the ledger — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md:238` — reason: fixed in this diff
- **violation** — wrong citation range `mod.rs:20-29` (the `activation_summary` re-export is on L19) — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md:218` — reason: fixed in this diff
- **clean** — Australian English, no Mermaid added, line counts and `file:line` citations spot-checked against `src/`, named tests exist, no hidden paths staged (no `CODING-STANDARDS.md` exists here; `CONTRIBUTING.md` and `AGENTS.md` were used)

## Test Plan

- Added `tests/issue_2144_chunk_07c_write_side_sweep.rs` (ledger contract, no product code). Tests:
  - `tests/issue_2144_chunk_07c_write_side_sweep.rs::each_write_side_file_has_exactly_one_settled_outcome`
  - `tests/issue_2144_chunk_07c_write_side_sweep.rs::write_side_findings_name_issues_listed_under_issues_filed`
  - `tests/issue_2144_chunk_07c_write_side_sweep.rs::defect_classes_probed_records_every_write_side_class`
  - `tests/issue_2144_chunk_07c_write_side_sweep.rs::header_only_table_has_no_data_rows`
  - `tests/issue_2144_chunk_07c_write_side_sweep.rs::parser_rejects_a_missing_section_loudly`
- `cargo test --test issue_2144_chunk_07c_write_side_sweep`: 5 passed.
- Each finding's own regression test is named in its issue (#2449-#2453) and ships with that finding's fix PR.
- Original trigger closed, no trivial bypass: this change alters no runtime code path, so the audit cannot open a new one. The five defects it found stay open as #2449-#2453, and the new test fails if any write-side file loses its settled outcome or its issue listing, or if a probed class loses its rows, so the findings cannot be dropped from the ledger unnoticed.
- `cargo test --test issue_2143_chunk_07b_read_side_sweep --test issue_2088_sweep_ledger_contract`: 14 passed (ledger contract still holds).
- `./quality.sh < /dev/null`: "All quality checks passed!" (run before the final docs-only lint and citation fixes; `markdownlint-cli2` on the ledger was re-run after them: 0 issues).
- Not covered: a real 2 GB `i32::MAX` offset probe was not run; the offset budget is argued by a scaled-limit round trip and the arithmetic, and the ledger says so.
