## Summary

The streaming recording session's temporary file (`<temp_dir>/discovery_data.parquet.tmp`) is now created with `create_new` (`O_EXCL`) instead of `File::create`. A new `ParquetRecordWriter::new_exclusive` does this, and `RecordingSession::new` uses it. `start_session` therefore fails, with an error naming the path, when anything already sits there, and leaves it untouched: a regular file, a symlink (including a dangling one), a recording retained by the #1902 guard, or another live session's file. `ParquetRecordWriter::new` keeps its truncating behaviour for its other callers. Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merge retains the PR version `0.74.349` in both files, ahead of Develop’s `0.74.348`. Closes #2515.

## Spec

### Intent and Rationale

- `File::create` truncates and follows symlinks, so `start_session` could write through a planted link, destroy a retained #1902 recording, or corrupt a live session. The issue offers "fail clearly" or "unique name"; I chose fail clearly because existing tests, `Drop`, `finish_session` and the guide all rely on the fixed name `discovery_data.parquet.tmp`.

### Essential Design Decisions

- A separate `new_exclusive` constructor (sharing a private `open`) rather than changing `new`, because `record_discovery_data` (`src/record/mod.rs`) and tests reuse `new` on a path they own; that path is outside this issue.
- Trade-off accepted: a crash orphan or retained recording in a `temp_dir` now blocks a new session there until the directory is cleaned (documented in `docs/STREAMING_GUIDE.md` and the `streaming.rs` module docs). A failed `RecordingSession::new` never runs `Drop`, so the occupier is not deleted.
- If `ArrowWriter::try_new` fails after an exclusive create, the file this call created is removed.

### Undiscoverable Facts

- Sessions live in the static `SESSIONS` map, which is not dropped on crash, so the old guide claim that `Drop` cleans up after a crash was wrong; corrected.

## Evidence

`tests/issue_2515_streaming_exclusive_temp.rs` drives the public streaming API; each refusal test also removes the occupier and asserts `start_session` then succeeds (positive control). Unit tests `new_exclusive_refuses_existing_file_and_leaves_bytes` and `new_exclusive_creates_fresh_file` are in `src/parquet_format/writer.rs`.

Callers of `ParquetRecordWriter::new` checked (`src/record/mod.rs`, `src/parquet_format/mod.rs` tests, `tests/issue_2453_*`, `tests/issue_2066_*`): unchanged behaviour. The only production caller moved to `new_exclusive` is `RecordingSession::new`.

The coverage repair replaces the obsolete line-number reference to the CI documentation check with its stable workflow and step name. The named check remains required in both the workflow and the documentation, covered by `tests/issue_1685_doc_link_integrity.rs::ci_doc_build_step_cites_the_real_check_documentation_step`.

## Test Plan

- Added `tests/issue_2515_streaming_exclusive_temp.rs`: `existing_regular_file_is_preserved_and_start_session_fails`, `symlink_at_tmp_path_is_not_written_through`, `dangling_symlink_is_not_created_through`, `second_session_on_same_temp_dir_does_not_corrupt_the_first`, `retained_recording_survives_a_second_start_session`.
- Red check: with `src/streaming.rs` still calling `ParquetRecordWriter::new`, all 5 tests fail (`start_session` returned `Ok`); with `new_exclusive` all 5 pass.
- Passed on the head: `cargo test --test issue_2515_streaming_exclusive_temp` (5), `cargo test --lib parquet_format::writer` (21), `cargo test --lib streaming` (28), `cargo test --test issue_1751_streaming_session_locks --test issue_1876_cancelled_session_append --test issue_2451_record_atomic_write`, `cargo test --test ffi` (245); `cargo fmt --check`; `cargo clippy --lib --tests -- -D warnings`.
- The documentation pointer test now checks the stable workflow and step name instead of a line range that changes when earlier steps move.
- Docs updated: `docs/STREAMING_GUIDE.md` (cancellation, crash recovery, failed-finalisation sections), `docs/FFI_API.md` (`start_discovery_session` row), `src/streaming.rs` module docs and `start_session` doc comment.

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeds the 590 s foreground cap on this host (as in #2496/#2497/#2508/#2509); targeted tests, fmt and clippy were run; CI runs the full gate" -->

Not checked: the full `./quality.sh` locally; the other tests in the main `tests/` suite beyond those listed; `record_discovery_data`'s own `.tmp` creation (same pattern, out of this issue's scope).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
