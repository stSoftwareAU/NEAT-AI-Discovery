## Summary

The Parquet decode preflight now charges the real decoded `neuron_uuid` string bytes instead of a flat 36 bytes per row, so a small file whose identifier column is dictionary-encoded with a long repeated value is refused before Arrow builds a batch. Closes #2507.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- A dictionary page stores each distinct string once, but Arrow copies the value into every row of a `StringArray` batch. The footer's encoded size therefore understated the decode, and `charge_record` only runs after the batch exists.
- The footer's `unencoded_byte_array_data_bytes` size statistic is the exact decoded total, so it replaces the 36-byte guess where present. This fixes every reader at once, because all of them share `projected_bytes_for_row_groups`.

### Essential Design Decisions

- When the statistic is absent (or negative, i.e. malformed), a chunk with encodings beyond PLAIN and the RLE or BIT_PACKED level encodings is bounded by `min(rows, 1024) × uncompressed size` (the longest value cannot exceed the chunk), and other chunks by their uncompressed size; both keep the old `rows × 36` floor (`ASSUMED_UUID_HEAP_BYTES`, kept with reworded docs).
- Assumption: the issue's alternative ("reject identifiers above the writer's length limit before decoding") is not used, because the length is unknown before decoding; the footer statistic is.
- A footer that misstates its own statistics is still outside the protection (documented).
- `ParquetFooterStats` gains `uuid_heap_bytes`; read by `projected_in_memory_bytes`, which feeds `estimate_parquet_in_memory_bytes` (`src/analysis/utils/memory.rs`).

- The merge with Develop retains the fallible footer projection: negative row counts, selected column sizes and error value counts are rejected before adding the UUID heap estimate (Issue #2509).

### Undiscoverable Facts

- None.

## Evidence

Library change; no UI. Readers covered, all via `DecodeBudget::preflight_columns` / `preflight_row_groups`: eager (`read_all_records_grouped_by_neuron_bounded`), lazy, limited, streaming index, streaming block load, activation summary, merge.

## Test Plan

- `src/parquet_format/footer.rs` unit tests: statistic trusted exactly; absent + dictionary (below and above one batch); absent + plain; negative statistic. The obsolete fallback treated DELTA_BYTE_ARRAY as plain; the remaining behaviour bounds its decoded batch, covered by `src/parquet_format/footer.rs::tests::absent_statistic_on_delta_byte_array_bounds_one_batch_of_copies`.
- New `tests/issue_2507_parquet_long_uuid_preflight.rs`: a 2,000-row, one-identifier (20,000 bytes) dictionary-encoded file under a 1 MB ceiling is refused by each reader with "footer preflight before Arrow batch construction", with a VmHWM bound; positive controls (36-char repeated, distinct 36-char identifiers) are accepted.
- Edited `the_streaming_block_loader_charges_the_decode_budget` in `tests/issue_2005_shared_batch_decoder.rs`: the two `preflight_columns` calls became `preflight_row_groups` (the streaming readers' real preflight; the eager one now correctly counts the record struct). No assertion removed; the per-record-charging assertion is unchanged and still reached (checked arithmetically: row-group preflight ~0.84 MB < 1 MB, charging 7,000 × (struct + 100) > 1 MB).
- Red checks (helper reverted to `rows × 36`): 4 of 6 footer unit tests and 7 of 9 integration tests go red (the two 36-char controls stay green by design). Breaking only `preflight_row_groups` reddens lazy, activation, index, block and merge; breaking only `preflight_columns` reddens eager.
- Results on the final tree: `cargo test --lib parquet_format` 49 passed; `issue_2507` 9, `issue_2446` 11, `issue_2005` 6 passed; `cargo fmt`, `cargo clippy --all-targets -- -D warnings` clean.
- Docs: `docs/CONFIGURATION.md`, `docs/CACHE_TUNING.md`, `CHANGELOG.md` updated; doc comments in `footer.rs`, `decode_budget.rs`, `memory.rs`.

Not done / risk: `read_records_from_parquet_with_limit_and_budget` shares `preflight_row_groups` but has no dedicated 2507 test; the fallback bound can over-refuse a foreign dictionary-encoded file with a large dictionary and no size statistic.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 590 s foreground cap while compiling (exit 124); fmt, clippy -D warnings and the targeted tests above were run; CI runs the full gate" -->
