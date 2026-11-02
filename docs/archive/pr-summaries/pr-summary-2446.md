## Summary

Parquet readers reject footer-reported oversized decoded selections before
Arrow constructs batches, including compact files with huge error lists.
Closes #2446.

The grouped/eager reader checks the whole selected file. UUID-filtered,
observation-limited, activation-summary, streaming-index, streaming-block and
merge readers instead check each row group and decode it with a separate
reader. Arrow batches therefore cannot cross the boundary checked by the
preflight, while large files with individually bounded groups remain usable.

## Design

- Eager estimates match the preload planner: the maximum of selected
  uncompressed bytes and record overhead plus decoded error values. Error
  values are counted once.
- Non-eager estimates cover selected Arrow columns and decoded error values,
  without charging DiscoverRecord overhead to folding and indexing readers.
- All row groups are checked before decoding. A filter cannot hide an
  oversized group. Parsed metadata and the open file are reused for separate
  row-group readers.
- Retained records still consume the cumulative record budget. Merge closes an
  output row group only after 2 MiB of decoded Arrow data has been buffered
  (`MERGE_OUTPUT_ROW_GROUP_FLUSH_BYTES`), so the encoder stays bounded without
  writing one tiny output row group per batch. The encoder's own
  `in_progress_size` was not used: it undercounts highly compressible data.
- Projections match the selected columns used by the estimate. The UUID index
  does not decode errors or unrelated columns.
- Footer under-reporting remains a limitation, tracked in #2447.
- Cargo.toml, Cargo.lock and fuzz/Cargo.lock record version 0.74.322.

## Evidence

The Issue #2446 suite contains 11 tests, including:

- A 20,000,000-value hostile list, refused before Arrow allocation.
- A legal file with distinct errors and retained charge above 90% of 1 MB.
- Whole-file rejection when combined groups exceed the eager budget.
  This combined-size refusal applies to the grouped reader only.
- Large files accepted by lazy, limited, streaming-block and merge readers
  when their individual row groups fit.
- 128 one-row groups, 121 holding 200,000 errors each: lazy, limited, streaming-block
  and merge readers succeed with a bounded Linux VmHWM increase, rather than
  constructing a batch containing all groups.
- Projection, schema validation and merge-destination preservation.
- A 3,000-row input in six row groups merges to a single output row group.

The shared-decoder suite contains 6 tests. The original 20,000-row UUID index
fixture succeeds without retained-record overhead. A 30,000-row group exceeds
the UUID-column estimate and is refused. The separate 7,000-row long-identifier
fixture still requires record charging to reject a streaming block after its
preflights fit.

The activation-summary/index fixture now uses one ordinary 12,000-row group,
rather than splitting those rows to conceal record-overhead rejection.

## Test Plan

Run on this head (Linux aarch64, Rust 1.99, warnings denied), so the Linux
VmHWM assertions are exercised:

- `cargo fmt --all -- --check`: passed.
- `cargo clippy --locked --all-targets --all-features -- -D warnings`: passed.
- `cargo test --locked --all-features --test issue_2446_parquet_predecode_budget
  --test issue_2005_shared_batch_decoder -- --test-threads=2`: 17 passed
  (`tests/issue_2446_parquet_predecode_budget.rs` 11 tests,
  `tests/issue_2005_shared_batch_decoder.rs` 6 tests), including
  `lazy_and_streaming_reads_accept_large_files_with_bounded_row_groups` and
  `small_row_groups_cannot_combine_into_an_oversized_arrow_batch` and
  `merge_keeps_few_large_output_row_groups`.
- Red check: with `MERGE_OUTPUT_ROW_GROUP_FLUSH_BYTES` set to 0 (a flush after
  every batch), `merge_keeps_few_large_output_row_groups` failed with
  `left: 6, right: 1`. The constant was restored.
- `cargo test --locked --all-features --test issue_1869_parquet_decode_bound
  --test issue_1901_parquet_schema_nullability
  --test issue_1923_activation_summary`: 25 passed (11, 7 and 7 tests).
- Red check: with `row_group_batches` in
  `src/parquet_format/decode_budget.rs` changed to give every group's reader all
  row groups (so one Arrow batch spans groups),
  `small_row_groups_cannot_combine_into_an_oversized_arrow_batch` failed at its
  first assertion: `many-group lazy read high-water mark grew by 196296 KiB
  during the read`. The change was reverted; the other three readers in that
  test share `row_group_batches` but were not individually red-checked.
- Not run on this head: `./quality.sh` and the full `cargo test` run (the
  Issue #1865 quality-gate test invokes the script), the recording suite, and
  `cargo deny`. CI runs them.
