## Summary

`StreamingRecordCache::build_block_index` now records a block id for a neuron only when it differs from the neuron's last recorded block, looks neurons up with `get_mut` before allocating an owned `String`, and calls `shrink_to_fit()` on each list. The per-row push, the end-of-scan `sort()` + `dedup()` and the retained over-allocation are gone. Closes #2508.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The index was built with one `usize` push per Parquet row and an owned `String` key per row; the retained `Vec` capacity stayed resident for the cache's lifetime, defeating the memory saving the streaming fallback exists for.

### Essential Design Decisions

- `current_block` only increases while scanning, so "last entry differs" is equivalent to the old sort + dedup and keeps each list ascending and unique (lookups in `get` rely on this order only as a set of block ids; behaviour is unchanged).
- `shrink_to_fit()` is applied once after the scan because the map is held for the cache's lifetime; it is best-effort per the std docs.

### Undiscoverable Facts

- None.

## Evidence

- Test `block_index_holds_one_entry_per_distinct_block_per_neuron_without_slack` in `src/analysis/streaming.rs` (100 observations x 3 neurons plus a neuron present in only the first and last observation, block size 30).
- Red against the unfixed code: `neuron-a: retained capacity 128 for 10 distinct blocks` (the block-list equality passed first, so only the slack assertion fails). Green after the fix.
- `cargo test --lib streaming`: 28 passed. `cargo test --test recording issue_193_streaming_parquet`: 13 passed. `cargo test --test issue_2005_shared_batch_decoder`: 6 passed.
- `cargo fmt --check` and `cargo clippy --lib --tests -- -D warnings`: clean.
- No docs under `docs/` (outside `docs/archive/`) describe the block-index construction, so no doc change was needed.

## Test Plan

- Added `block_index_holds_one_entry_per_distinct_block_per_neuron_without_slack` (`src/analysis/streaming.rs`): asserts exact ascending distinct block ids per neuron (derived from row index / block size), `neuron-rare == [0, 10]`, `capacity < 2 * len`, and `num_blocks == ceil(rows / block_size)`.
- Red check: with the old code the capacity assertion fails (above); the new `get_mut`/last-block push and `shrink_to_fit` make it pass. The separate "push only on change" operand is covered by the exact block-list equality (a duplicate push would fail it).
- No existing test was edited or had assertions removed.

<!-- vibe-quality-gate-skipped reason="./quality.sh exceeded the 590 s foreground cap in this environment (same as earlier runs); fmt, clippy and the touched test targets were run separately. CI runs the full gate." -->

## Not checked

- Full `./quality.sh` did not complete locally; CI is its first full run.
- No measurement of the memory saving on a large recording; the test pins the retained capacity only.
