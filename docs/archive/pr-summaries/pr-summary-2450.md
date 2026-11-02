## Summary

Merge inputs are protected from collisions with the merge temporary path. Each
merge now writes through an exclusively created unique sibling file and cleans
up only the file it created. The crate version is incremented for this change.
Closes #2450.

Version bump: `Cargo.toml` updated for this change; the merged commit records
the final version.

## Spec

### Intent and Rationale

- Prevent an input named `{output_file}.tmp` or a pre-existing symlink there
  from being truncated, overwritten, or deleted by a merge.
- Use exclusive creation with bounded collision retries so concurrent merges
  can use separate sibling files without opening existing paths.

### Essential Design Decisions

- Keep `{output_file}.tmp` reserved as an input alias and allocate actual merge
  files as `{output_file}.tmp.<process-id>.<counter>`.
- Pass the exclusively opened `File` into the writer, preserving schema checks,
  writer closure, and rename-on-success behaviour.

### Undiscoverable Facts

- Before the fix, the input-at-temp-path test failed with `EOF: Parquet file
  too small. Size is 0 but need 8`; the input had been truncated before read.
- Symlink regression tests use `#[cfg(unix)]`; this run exercised them on Linux.

## Evidence

Backend-only change. Regression tests use real Parquet recordings in isolated
temporary directories. The original trigger is closed: canonical alias checks
reject inputs resolving to the reserved `.tmp` path, while the actual temporary
file is opened with `create_new` and its already-open handle is passed to the
writer. No existing file or symlink can be opened through that pathname, and
cleanup is reached only after this call created its unique file.

```mermaid
flowchart TD
    A[Merge request] --> B{Input aliases output or reserved output.tmp?}
    B -- yes --> C[Reject before writing]
    B -- no --> D[Create unique sibling with create_new]
    D -- collision --> D
    D -- created --> E[Write and validate through open File]
    E -- failure --> F[Remove this call's temporary file]
    E -- success --> G[Rename temporary file to output]
```

## Reproduction

- **symptom** — an input at `{output_file}.tmp` was truncated to zero bytes
  before reading, and a pre-existing symlink there redirected writes to its
  target.
- **status** — `verified` — the regression test failed against the unfixed
  implementation and passed after the fix.
- **regression test** — `tests/issue_2450_merge_temp_path.rs::input_at_legacy_temp_path_is_refused_without_changing_bytes`

## Test Plan

- `tests/issue_2450_merge_temp_path.rs::input_at_legacy_temp_path_is_refused_without_changing_bytes` — failed against the base implementation, then passed with the fix. Removing the legacy alias guard made it fail again.
- `tests/issue_2450_merge_temp_path.rs::pre_existing_temp_symlink_and_target_are_unchanged` — passed; the symlink and its target remain byte-identical.
- `tests/issue_2450_merge_temp_path.rs::occupied_regular_temp_path_is_preserved_during_successful_merge` — passed; the merge succeeds without modifying the occupied path.
- `tests/issue_2450_merge_temp_path.rs::failed_merge_preserves_occupied_temp_path_and_cleans_its_own_temp` — passed; the merge error is reported, the occupied path remains unchanged, and this call's temp is removed.
- `tests/issue_2450_merge_temp_path.rs::resolved_alias_to_legacy_temp_path_is_refused_unchanged` — passed; a symlink alias to the reserved temp path is rejected without changing the target.
- `src/parquet_format/writer.rs::tests::merge_temp_creation_does_not_open_an_existing_path` — passed; replacing exclusive creation with truncating creation made it fail.
- `src/parquet_format/writer.rs::tests::merge_temp_allocation_uses_distinct_sibling_paths` — passed.
- `src/parquet_format/writer.rs::tests::merge_preserves_destination_when_input_missing`, `src/parquet_format/writer.rs::tests::merge_rejects_foreign_schema_and_leaves_destination_untouched`, `src/parquet_format/writer.rs::tests::merge_rejects_output_aliasing_input`, and `src/parquet_format/writer.rs::tests::merge_writes_all_inputs_and_leaves_no_temp_file` passed.
- `cargo fmt --all -- --check` passed.
- `./quality.sh < /dev/null` passed on retry: dependency audit, build, Clippy, type checks, all tests, documentation build, and release library build passed. The suite reported 1,757 library tests passed and 2 ignored.
- No existing test assertions were removed.
- Not checked: macOS and Windows execution. The symlink regression is limited to Unix; no run was performed on macOS.
