## Summary

`record_discovery_data` now writes to `discovery_data.parquet.tmp` and publishes
it only after Parquet finalisation succeeds. Processing, construction,
finalisation, and rename failures clean up the temporary file while retaining
the original error and reporting any cleanup failure. Closes #2451.

Version bump: the `Cargo.toml` patch version was bumped manually above Develop,
with `Cargo.lock` updated to match.

## Spec

### Intent and Rationale

- Prevent a failed recording from truncating the previously published Parquet
  file.
- Publish the completed file with a rename after the writer closes, following
  the existing temporary-file publication pattern.

### Essential Design Decisions

- Use a `.tmp` sibling so the writer never opens the live destination.
- Remove the temporary file on each failure path; if cleanup also fails, its
  error is added as context while the original error remains in the chain.
- Leave streaming's retry and temporary-file retention behaviour unchanged.

### Undiscoverable Facts

- The new regression test failed against the original code because the live
  file became the four-byte `PAR1` prefix instead of remaining byte-identical.
- Removing temporary-file cleanup made the no-existing-recording test fail on
  its `.tmp` residue assertion.
- Finalisation I/O failure was not injected directly; it follows the same
  cleanup path as other write-phase errors.

## Evidence

Backend-only library change; no browser surface is involved. The recording
function opens only the `.tmp` path and calls `rename` after `finish`, closing
the original truncation trigger without changing the streaming call path.

The failure trigger was `training_data[1].neuron_data = None`, after the first
observation had been written. The regression assertion observed the original
file change from a valid Parquet recording to `[80, 65, 82, 49]` (`PAR1`), and
then passed after the fix. The named test also checks exact bytes, successful
Parquet readability, the processing error, and absence of the `.tmp` file.

## Reproduction

- **symptom** — a processing error after writer creation truncates the existing
  `discovery_data.parquet` and leaves an incomplete recording.
- **status** — `verified` — the regression test failed against the unfixed code
  and passed after the change.
- **regression test** —
  `tests/issue_2451_record_atomic_write.rs::processing_error_preserves_existing_recording_and_cleans_temporary_file`

## Test Plan

- Added
  `tests/issue_2451_record_atomic_write.rs::processing_error_preserves_existing_recording_and_cleans_temporary_file`:
  failed against the original production code on the byte-preservation
  assertion; passed after the fix.
- Added
  `tests/issue_2451_record_atomic_write.rs::successful_recording_replaces_existing_file_with_readable_data`:
  passed; replacement rows were readable and the old row was absent.
- Added
  `tests/issue_2451_record_atomic_write.rs::processing_error_without_existing_recording_cleans_temporary_file`:
  passed; no final file or temporary residue remained. With cleanup removed,
  this test failed on the `.tmp` residue assertion.
- Added
  `tests/issue_2451_record_atomic_write.rs::rename_error_cleans_temporary_file_and_preserves_destination_directory`:
  passed; rename failure preserved the occupied directory and removed `.tmp`.
- Added
  `tests/issue_2451_record_atomic_write.rs::writer_construction_error_reports_cleanup_failure`:
  passed; the writer construction cause and cleanup failure were both surfaced.
- Existing
  `tests/issue_2067_record_module_docs.rs` — passed (5 tests).
- `rustfmt --edition 2024 --check src/record/mod.rs src/record/processing.rs tests/issue_2451_record_atomic_write.rs` — passed.
- `cargo clippy --lib --test issue_2451_record_atomic_write -- -D warnings` — passed.
- `./quality.sh < /dev/null` — passed on the pre-merge head. After merging `origin/Develop`, a re-run was cut off by the 590s foreground cap while still in the lib unit tests (no failure seen); the final-head gate is left to CI.

<!-- vibe-quality-gate-skipped reason="foreground command cap (600s) is shorter than the ~15m gate; formatter, clippy and the targeted tests passed on the final head; CI runs the full gate" -->

The original trigger is closed: `record_discovery_data` writes only to the
temporary sibling and the final path is touched only by the rename after
successful finalisation. There is no direct-write fallback in this function.

Documentation updated in `docs/FFI_API.md` under the discovery output format;
recording publication comments updated in `src/record/mod.rs` and
`src/record/processing.rs`.

The merge with Issue #2452 retains shared-decode cache invalidation after the
finished temporary file is successfully published. Failed writes and failed
renames leave the existing recording and its cache intact. The successful
re-recording path is covered by
`tests/issue_2452_shared_cache_inplace_rewrite.rs::record_discovery_data_invalidates_the_shared_decode_cache`.
