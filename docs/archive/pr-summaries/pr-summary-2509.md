## Summary

The decode preflight (`projected_bytes_for_columns` / `projected_bytes_for_row_groups`) now refuses a negative row-group `num_rows`, selected-column `uncompressed_size` or `errors` column `num_values` instead of counting it as 0. `DecodeBudget::preflight_columns` and `preflight_row_groups` return that error, wrapped with the file path. `read_footer_stats` shares the same checked conversion (`footer_u64`), so all footer readers produce one field-named message. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version. Closes #2509.

## Spec

### Intent and Rationale

- `u64::try_from(..).unwrap_or(0)` let a hostile footer drop a projection term, so the preflight could approve a batch it should refuse. Failing loud with the field name matches what #2447 did for `read_footer_stats`.

### Essential Design Decisions

- The two projection helpers now return `Result<u64>` (callers: only `DecodeBudget::preflight_columns` / `preflight_row_groups`), so a caller cannot ignore the error.
- Only fields the projection reads are checked: `uncompressed_size` of selected roots, `num_values` of `errors` when selected. Unselected negative values are not part of the projection and remain accepted by the preflight (still rejected by `read_footer_stats`).

### Undiscoverable Facts

- The parquet crate (v60.1) parses footers carrying `-1` for all three fields; the new test patches real files and the footer still loads.

## Evidence

Tests: `tests/issue_2509_preflight_negative_footer.rs` patches each field to -1 in a real file (located by parsing the patched footer, so the right field is hit), then asserts both preflights fail with `Invalid Parquet footer <field>: -1` under an unlimited budget (`DecodeBudget::new(None)`), so the refusal cannot come from the limit.

## Test Plan

- Added `tests/issue_2509_preflight_negative_footer.rs`: `valid_footer_passes_both_preflights`, `negative_row_group_num_rows_is_refused`, `negative_selected_uncompressed_size_is_refused`, `negative_errors_num_values_is_refused`, `negative_values_outside_the_selection_are_not_charged` (covers the `selected_roots` and `include_errors` operands: unselected negatives are accepted).
- Red check: with the base `footer.rs` and `decode_budget.rs` restored and the new test kept, the three refusal tests fail and the two control tests pass; they pass on the fix.
- Passed on the head: `cargo test --test issue_2509_preflight_negative_footer`, `--test issue_2447_parquet_footer_stats`, `--test issue_2005_shared_batch_decoder`, `--test issue_2446_parquet_predecode_budget`, `cargo test --lib parquet_format`; `cargo fmt --check`; `cargo clippy --lib --tests -- -D warnings`.
- No existing test assertion was removed or edited.
- Docs: grepped `README.md` and `docs/` (excluding archive) for `projected_bytes_for`, `footer preflight`, `read_footer_stats`; only the historical audit `docs/audits/security-sweep-chunk-07-parquet.md` matches, left as a record.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeds the 590 s foreground cap on this host (as in #2496/#2497/#2508); targeted tests, fmt and clippy were run; CI runs the full gate" -->

Not checked: the full `./quality.sh` locally; fuzz targets were not run.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
