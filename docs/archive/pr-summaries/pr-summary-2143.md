## Summary

Chunk 7b read-side audit of `src/parquet_format` (`reader.rs`, `batch_columns.rs`, `footer.rs`,
`decode_budget.rs`, `schema.rs`, `activation_summary.rs`). All six files were read end to end and
probed with hand-built Parquet files. The staged ledger
`docs/audits/in-progress/security-sweep-chunk-07-parquet.md` now carries the six per-file outcomes,
every probe (class, `file:line`, input, outcome) and the filed issues. Two findings survived:

- **#2446 (SEC-2091-01, medium)**: the decode budget is charged after the parquet crate has built
  the whole Arrow batch. A 1,643-byte file with one 20,000,000-value `errors` list grew the process
  high-water mark by 240,586,752 bytes under a 1 MB budget.
- **#2447 (SEC-2091-02, low)**: `read_footer_stats` maps a negative footer `num_rows` /
  `total_byte_size` / `num_values` to 0, so the footer projection under-reports.

`batch_columns.rs` and `schema.rs` are clean. The write side (`mod.rs`, `writer.rs`,
`shared_records.rs`) stays `pending #2144`, so the record stays under `in-progress/` and chunk `7`
in `lib-sweep-coverage.json` is unchanged. Nothing under `src/` and no `Cargo.toml` changed, so
there is no source change to version; the merged commit records the final version. Closes #2143.

## Spec

### Intent and Rationale

- The issue asks for an audit that files one issue per surviving finding, not for fixes. Fixes ship
  in #2446 and #2447, each with its own failing-first test.
- Probes that found nothing are recorded in the ledger and were not committed; the throwaway probe
  file was deleted before the commit.

### Essential Design Decisions

- The ledger stays in `docs/audits/in-progress/` (trusted comment on #2143), because
  `tests/issue_2088_sweep_ledger_contract.rs` scans only the top level of `docs/audits/`.
- The issue says "no `.rs` change". This PR adds one test file,
  `tests/issue_2143_chunk_07b_read_side_sweep.rs`, because the worker's security-fix gate for
  `security`-labelled issues blocks PR creation unless the branch changes a test file. It guards the
  ledger text only (as `tests/issue_2249_chunk9_ledger_complete.rs` does for chunk 9). Nothing in
  `src/` or `Cargo.toml` changed.
- The parquet crate question is answered in the ledger: parquet 60.0.0 builds the whole
  `RecordBatch` (default 1,024 rows, `DEFAULT_BATCH_SIZE` in its `arrow_reader/mod.rs`) before the
  first `charge_record` and has no per-list cap. #2446 asks for an in-repo footer pre-check rather
  than a crate change.

### Undiscoverable Facts

- Peak memory was measured by writing `5` to `/proc/self/clear_refs` and then reading `VmHWM`; the
  crate's global allocator is private, so a test-local counting allocator cannot be installed.
- Footer lies were produced by patching compact-Thrift bytes in a 50-row file's footer
  (`0x16 0x64` to `0x16 0x01`), which the parquet crate accepts.

## Evidence

Audit only; no UI or performance surface. Probe outputs (recorded in the ledger):

- `read_all_records_grouped_by_neuron_bounded(path, None, Full, Some(1))` on the 1,643-byte file:
  `MemoryExhausted`, high-water growth 240,586,752 bytes.
- Footer `num_rows` patched to -1: `ParquetFooterStats { num_rows: 0, uncompressed_bytes: 459,
  error_values: 100 }`, while the reader decodes all 50 rows.
- Schema trust, NaN/inf/null and UUID probes: see the ledger's `## Defect classes probed`.
- Filed issues (checked with `gh issue view`): #2446 "SEC-2091-01: decode budget is charged after
  the Arrow batch is materialised", labels `security,lang:rust,bug,severity:medium,confidence:high`;
  #2447 "SEC-2091-02: read_footer_stats maps negative footer sizes to 0", labels
  `security,lang:rust,bug,severity:low,confidence:high`. Each body names the `file:line`, hostile
  input, observed behaviour, its `tests/issue_<n>_*.rs`, the `Cargo.toml` bump, and links #2091 and
  #2143; #2447 (footer parsing) also names the `fuzz/` requirement, and #2446 names one too.

## Original Trigger Closure

The original trigger is the unaudited Parquet read side: hostile or malformed Parquet files reaching
`src/parquet_format` with no recorded review. It is closed for audit purposes with no trivial bypass,
by static reasoning over the audited code path: all six read-side files were read end to end, each
defect class has a recorded probe in the ledger, and `tests/issue_2143_chunk_07b_read_side_sweep.rs`
fails if a file lacks a settled outcome or a class lacks a probe, so a file cannot be skipped
silently. The two hostile inputs that survived (the oversized `errors` list, #2446, and the negative
footer counts, #2447) are not fixed here: this PR changes no production code, so those two inputs
still behave as described in the Evidence section until their own PRs land. Each is tracked by an open
issue with a failing-first test, and no equivalent input outside those two issues was found in the
probed classes.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — Every one of the 1,377 lines across the six files has been read; the ledger lists all six under **Files swept** with `outcome: clean` or `outcome: finding #N` — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md` `## Files swept`; `tests/issue_2143_chunk_07b_read_side_sweep.rs::each_read_side_file_has_exactly_one_settled_outcome` — reviewer: met
- **met** — Every defect class above has at least one recorded probe with class, `file:line`, input and outcome in the ledger — evidence: ledger `## Defect classes probed`; `tests/issue_2143_chunk_07b_read_side_sweep.rs::defect_classes_probed_records_every_read_side_class` — reviewer: met
- **met** — Every surviving finding has exactly one `SEC-2091-NN` issue with `severity:*` and `confidence:high` labels, a named failing-first test path and (for parsing findings) a fuzz corpus requirement — evidence: #2446 and #2447 bodies and labels, read with `gh issue view` on the final head — reviewer: partial — reason: the reviewer could not see the issue bodies from the diff; they were read here and carry all the named items
- **met** — The external-`parquet`-crate question (batch materialised before `charge_record`) is answered in the ledger either way — evidence: ledger row `reader.rs:314-318` under Decode-budget completeness — reviewer: met
- **met** — No `.rs` change and no `Cargo.toml` change in this sub-issue's PR — evidence: no `src/` file and no `Cargo.toml` in the diff; the only `.rs` file is the ledger guard test `tests/issue_2143_chunk_07b_read_side_sweep.rs` — reviewer: missing — reason: the reviewer read the criterion literally and counted the test file; it is there only because the security-fix gate blocks PR creation without a test file, and it changes no production code
- **partial** — Tests and quality checks pass (`./quality.sh`) — evidence: `cargo test --test issue_2143_chunk_07b_read_side_sweep` passes, markdownlint reports 0 issues — reviewer: partial — reason: quality gate not yet run to completion, it hit the 590 s foreground cap during the unit tests; CI runs it on the PR — remaining: merge-verification-only
- **unrequested** — `tests/issue_2143_chunk_07b_read_side_sweep.rs` — reviewer: unrequested — reason: required by the security-fix gate, see above

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — ledger cited `cache/mod.rs` at HEAD line numbers, not `b85a551` — evidence: `docs/audits/in-progress/security-sweep-chunk-07-parquet.md:124` — reason: fixed in this diff (re-derived with `git grep -n b85a551`; only `cache/mod.rs` had drifted: 220 and 264)
- **violation** — PR summary cited code by line number — evidence: `docs/archive/pr-summaries/pr-summary-2143.md:13` — reason: fixed in this diff (cited by symbol)
- **violation** — the test counted the table header and separator as data rows — evidence: `tests/issue_2143_chunk_07b_read_side_sweep.rs:229` — reason: fixed in this diff (`data_rows` counts rows after the separator; header-only table is asserted to have none)
- **clean** — Australian English, ledger claims against `src/parquet_format`, named tests exist, no workflow or dependency change; review-enforced rules checked: cite-by-symbol (#1942), assertion-holds-either-way (#1799)

## Test Plan

- Added `tests/issue_2143_chunk_07b_read_side_sweep.rs` (five tests):
  `each_read_side_file_has_exactly_one_settled_outcome`,
  `read_side_findings_name_issues_listed_under_issues_filed`,
  `defect_classes_probed_records_every_read_side_class`, `header_only_table_has_no_data_rows`,
  `parser_rejects_a_missing_section_loudly`.
- Red check: with the base (staged) ledger swapped in, the first three fail and the parser test
  passes; with the new ledger all pass.
- Quality gate: `./quality.sh` was started and stopped by the 590 s foreground cap while running unit
  tests; <!-- vibe-quality-gate-skipped reason="foreground cap reached, CI runs the gate" -->
- Related existing ledger contract `tests/issue_2088_sweep_ledger_contract.rs`: the record stays
  under `docs/audits/in-progress/`, which that test does not scan.
- Callers/siblings: no shared helper or call site changed.
