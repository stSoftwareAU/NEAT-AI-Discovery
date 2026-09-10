## Summary

The three public modules under `src/parquet_format/` opened with a header fully
derivable from the file name — "Parquet reading and deserialisation", "Schema
definitions and validation", "Parquet writing and serialisation" — so a reader
orienting themselves by the module index learnt nothing about the on-disk
format's actual contracts. Each header now states the one property a caller
most needs and cannot infer from the path. Closes #2066.

- **`schema.rs`** — the five fixed columns in order (`obs_index`,
  `neuron_uuid`, `value`, `activation`, `errors`) with their Arrow types and
  the fact that `value` is the only nullable one; that `create_schema` is the
  single declaration both reader and writer are checked against; and the two
  bounds it owns (`1..=100`-byte UUIDs, `MAX_ARROW_OFFSET`).
- **`reader.rs`** — reads are column-projected via `ColumnProfile`, and a
  profile that omits `errors` returns an **empty `errors` vec**, so empty means
  "not read", not "no errors". Also names the two pre-decode guards
  (schema/nullability validation, `.parquet.tmp` rejection) and the decode
  budget.
- **`writer.rs`** — the atomicity boundary, stated accurately rather than as
  the issue's suggested wording. `ParquetRecordWriter` appends into its
  destination as records arrive and the file is **not** valid Parquet until
  `finish()` writes the footer, so publish-atomicity is the caller's job
  (`src/streaming.rs`'s `.parquet.tmp` + rename); `merge_parquet_files` is the
  only function here that publishes atomically. Per-record field atomicity
  (AGENTS.md "Atomic Record Writes") is a caller obligation the writer cannot
  detect a violation of.

`mod.rs`'s three-line sub-module index carried the same paraphrases and was
updated in step, since it is the first surface a reader hits.

Deliberately **not** documented as the issue suggested: writer.rs has no
"atomic-per-observation write guarantee" at the file level — `ParquetRecordWriter`
writes straight into its destination path (`src/parquet_format/writer.rs:111`).
Documenting a guarantee the code does not make would have been worse than the
paraphrase it replaced, so the header states where the boundary actually sits.

## Evidence

Backend/library change with no web interface to screenshot. The evidence is the
new behavioural test file, which drives each documented contract through the
public API so the prose cannot drift into fiction:

```
running 4 tests
test schema_declares_five_columns_with_value_the_only_nullable_one ... ok
test module_headers_are_no_longer_the_file_name_paraphrase ... ok
test file_is_only_readable_after_finish_writes_the_footer ... ok
test without_errors_profile_returns_empty_errors_while_full_returns_the_values ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

`merge_parquet_files`' atomic temp-then-rename publish — the other claim the new
`writer.rs` header makes — is already covered by the existing unit tests in
`src/parquet_format/writer.rs` (which assert no `{destination}.tmp` survives a
merge, successful or failed), so it is not duplicated here.

## Test Plan

- Added `tests/issue_2066_parquet_format_module_docs.rs`:
  - `schema_declares_five_columns_with_value_the_only_nullable_one` — asserts
    the documented column order, Arrow types and nullability of
    `create_schema()`.
  - `without_errors_profile_returns_empty_errors_while_full_returns_the_values`
    — writes a file, reads it under `ColumnProfile::Full` and
    `ColumnProfile::WithoutErrors`, and asserts the projected read returns the
    same rows with an empty `errors` vec.
  - `file_is_only_readable_after_finish_writes_the_footer` — asserts a
    footer-less file fails to read and the same path reads back every record
    once `finish()` returns.
  - `module_headers_are_no_longer_the_file_name_paraphrase` — the regression
    guard: none of the three headers may re-open with its old paraphrase.
- `./quality.sh` run in full (includes `RUSTDOCFLAGS="-D warnings" cargo doc
  --no-deps`, which is what validates the new intra-doc links).
- Patch version bumped `0.74.237` → `0.74.238` per AGENTS.md.
