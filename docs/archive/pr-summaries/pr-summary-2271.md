## Summary

Fills region 13d (readers outside `src/config`) of the chunk-13 ledger `docs/audits/security-sweep-chunk-13-config-env.md`: the exact `rg` command with its 28 classified hits, 17 settings rows with a doc-versus-code verdict table, the probe dispositions, the doc drift (including `LIB_PATH`) and the issues filed. Adds the ledger test `tests/issue_2124_chunk_13d_ledger.rs` and one failing-first probe for the filed finding. Version bump: not made here, no `src/` or `Cargo.toml` change; the repository's `version-increment` CI job handles it. Closes #2271.

Outcome of the four candidates: (c) is filed as #2463 (low); (a) and (b) are refuted with reasons; (d) has nothing to adjudicate because the 13d-1 probes left no `#[ignore]`d case.

## Spec

### Intent and Rationale

- The 13d region was a header-only scaffold. Recording it makes the chunk-13 sweep checkable and gives #2125 (doc parity) and #2276 (silent opt-out class) their 13d inputs.
- The `rg` command in the issue cannot see the twelve `parse_env` readers or `RUST_LOG`, so two companion searches are recorded beside it and the hit count stays the `rg` command's own (28).

### Essential Design Decisions

- **Deviation from the issue text:** the issue asks for `file.rs::symbol` "alongside `file:line`" and cites line numbers (L59, L992 and so on). CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number" (Issue #1942) forbids line citations in docs, so 13d cites symbols only. The issue's lines were re-verified and are replaced by symbols; two of them had drifted (`sample_capture.rs` 355/449 are 357/451 at HEAD, `set_env_if_unset` starts at 52, not 53). The new test `the_13d_regions_cite_no_line_numbers` enforces it for 13d.
- **Deviation from the issue text:** the issue says the `NEURON_RANKING_BANDS` clamp is `[1, u32::MAX]`. At HEAD it is `[1, 1000]` (`MAX_NEURON_RANKING_RELIABILITY_BANDS`), recorded as stale, no finding.
- The verdict ("doc versus code") needs a place that the fixed ten-column schema lacks, so it is a separate four-column table inside the 13d settings region whose header starts with `reader`, not `variable`; the scaffold test counts exactly four `variable` headers.
- The "no `pending`" assertion matches the standalone word, so the identifier `env_setup_pending` (named by the issue) is allowed.

### Undiscoverable Facts

- The three GitHub comments (#2096, #2276, #2125) and issue #2463 were created in this run; their URLs are in the ledger.
- #2096 (the tracker) is already closed; earlier sub-issues commented on it anyway.

## Evidence

Backend/docs only. No UI file is touched.

- `cargo test --test issue_2096_env_readers_outside_config`: 12 passed, 0 ignored at HEAD, so finding (d) has no case to adjudicate. `cargo test --lib analysis::utils::platform`: 16 passed.
- `tests/issue_2463_constant_source_threshold_explicit_default.rs`: 4 controls pass; `cargo test ... -- --ignored` fails the probe with `get_constant_source_threshold returned Some(1e-6), expected Some(1e-7) (its twin ... returned Some(1e-7))`.
- Issues and comments: #2463 (labels `security`, `lang:rust`, `severity:low`, `confidence:high`), comments on #2276, #2125 and #2096.

## Acceptance Criteria

<!-- vibe-spec-review inputs="diff+issue-body" -->

- **met** — The 13d region records the exact `rg` command, the hit count, and one classification per hit. There are no unclassified hits. — evidence: `tests/issue_2124_chunk_13d_ledger.rs::the_inventory_records_the_exact_rg_command_hit_count_and_one_class_per_hit` — reviewer: met
- **met** — There is a settings row, with a doc-vs-code verdict, for every runtime reader in item 2. — evidence: `tests/issue_2124_chunk_13d_ledger.rs::every_named_reader_has_one_ten_column_settings_row_and_a_verdict` — reviewer: met
- **met** — The dispositions for `NEURON_RANKING_BANDS`, `thresholds.rs:176`, `XDG_RUNTIME_DIR`, `RUST_LOG`, `temp_dir` (#2118), `utils/mod.rs:95`, `env!`/`option_env!` and `LIB_PATH` are each recorded. — evidence: `tests/issue_2124_chunk_13d_ledger.rs::the_named_dispositions_are_recorded` — reviewer: met — reason: the two line-number headings are recorded as `samples/thresholds.rs::get_constant_source_threshold` and `utils/mod.rs::verbose_enabled` per CONTRIBUTING #1942
- **met** — Candidates (a)–(d) are each either filed in #2078 shape, linked in the ledger and commented on #2096, or recorded as refuted with a reason. — evidence: #2463 filed; (a), (b) refuted and (d) none under `## Probe dispositions (13d)`; `tests/issue_2124_chunk_13d_ledger.rs::the_filed_finding_is_linked_and_has_its_probe` — reviewer: met
- **met** — The ledger test exists and fails if a 13d region contains `pending` or is missing a named variable. — evidence: `tests/issue_2124_chunk_13d_ledger.rs`; with `pending` inserted into a 13d disposition, with the 13d `LIB_PATH` entry renamed, and with the `XDG_RUNTIME_DIR` settings row removed, the matching test went red — reviewer: met
- **met** — `./quality.sh` passes. — evidence: `./quality.sh` run in full on the head before the final symbol-citation edit, "All quality checks passed!"; re-run on the final head (after the symbol-citation edit) also printed "All quality checks passed!" — reviewer: met

## Standards Review

<!-- vibe-standards-review inputs="diff+CODING-STANDARDS.md" -->

- **violation** — CONTRIBUTING.md "Cite Code by Symbol, Never by Line Number" (Issue #1942): the first draft cited `file.rs:line` and `L<n>` throughout 13d — evidence: `docs/audits/security-sweep-chunk-13-config-env.md` — reason: fixed in this diff (symbols only; `the_13d_regions_cite_no_line_numbers` enforces it)
- **violation** — the ledger test pinned the line-number headings `thresholds.rs:176` and `utils/mod.rs:95` — evidence: `tests/issue_2124_chunk_13d_ledger.rs` — reason: fixed in this diff (needles changed to the symbol forms)
- **clean** — Australian English; named tests exist; the ledger test reads the Markdown record and `platform.rs` only to confirm cited `fn` names; imports; `#[serial]` on env-mutating tests; the `#[ignore = "finding: …"]` probe matches the convention in `tests/issue_2096_env_readers_outside_config.rs`. The standards reviewer used CONTRIBUTING.md and AGENTS.md as the repository has no CODING-STANDARDS.md.

## Test Plan

- Added `tests/issue_2124_chunk_13d_ledger.rs` (10 tests) and `tests/issue_2463_constant_source_threshold_explicit_default.rs` (4 controls plus 1 `#[ignore]`d probe). No existing test was edited, so no assertion was removed.
- Red checks on the ledger test, doc mutated then restored: standalone `pending` added to disposition (6) → `region_13d_is_non_empty_in_every_section_and_has_no_pending` red; `NEAT_AI_DISCOVERY_LIB_PATH` renamed in the 13d drift text → `lib_path_is_recorded_under_doc_drift_for_2125` red; `XDG_RUNTIME_DIR` settings row removed → `every_named_reader_has_one_ten_column_settings_row_and_a_verdict` red ("found 0"). Run before the symbol-citation edit.
- `the_13d_regions_cite_no_line_numbers` and the scanner test are covered by `the_line_number_scanner_flags_citations_and_ignores_identifiers`; the 13d text before the edit had 85 matching lines.
- Probe `explicit_default_override_is_honoured_like_its_twin` goes red against production (see Evidence); the controls pin the rest of the behaviour table so only the default-valued override fails.
- `cargo test --test issue_2124_chunk_13d_ledger --test issue_2211_chunk_13_ledger_scaffold --test issue_2088_sweep_ledger_contract`: 25 passed.
- `./quality.sh` on the final head: All quality checks passed.
- Not checked: the exact lines of `docs/CONFIGURATION.md` are no longer cited, so a later edit to that file cannot stale them; the `set_var` thread race in `platform.rs` is not re-audited (chunk 9 owns it).
