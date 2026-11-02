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
- Retained records still consume the cumulative record budget. Merge flushes
  each bounded batch so its output encoder cannot accumulate many decode units.
- Projections match the selected columns used by the estimate. The UUID index
  does not decode errors or unrelated columns.
- Footer under-reporting remains a limitation, tracked in #2447.
- Cargo.toml, Cargo.lock and fuzz/Cargo.lock record version 0.74.322.

## Evidence

The Issue #2446 suite contains 10 tests, including:

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

The shared-decoder suite contains 6 tests. The original 20,000-row UUID index
fixture succeeds without retained-record overhead. A 30,000-row group exceeds
the UUID-column estimate and is refused. The separate 7,000-row long-identifier
fixture still requires record charging to reject a streaming block after its
preflights fit.

The activation-summary/index fixture now uses one ordinary 12,000-row group,
rather than splitting those rows to conceal record-overhead rejection.

## Test Plan

Validation results for this revision are recorded below. Historical base
reproduction and mutation results are not claimed as runs of this revision.
Linux allocation assertions require Linux and are not exercised on macOS.

- Passed on this macOS host with Rust 1.99 and warnings denied:
  `cargo fmt --all -- --check`, fuzz formatting,
  `cargo clippy --locked --all-targets --all-features -- -D warnings`,
  `cargo check --locked --all-targets --all-features`,
  `cargo deny check` and `RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps`.
- Focused tests passed with `--all-features -- --test-threads=2`:
  Issue #2446 (10), shared decoder (6), decode bound (11), schema nullability
  (7), activation summary (7) and recording (81), 122 tests total.
  This includes `lazy_and_streaming_reads_accept_large_files_with_bounded_row_groups`
  and the new many-group regression. Linux VmHWM assertions remain for CI.
- `./quality.sh` passed Bash syntax, ShellCheck and cargo-install pinning,
  then its summary-layout script could not create a temporary file because
  macOS `mktemp` selected a sandbox-denied system temporary directory.
  The remaining Rust gates were invoked separately using a toolchain and
  Cargo cache under the permitted TMPDIR.

- Debug and release library builds passed with warnings denied.

- The full `cargo test --locked --lib --tests --all-features -- --test-threads=2`
  run stopped at
  `issue_1865_quality_gate_no_dep_mutation::quality_gate_never_mutates_the_dependency_graph`.
  It invokes the real quality script and encounters the same sandbox-denied
  macOS `mktemp` path. This existing test and the gate remain unchanged.
- After adding an explicit greater-than-64-MB fixture precondition, the new
  many-group test passed again, and all-target Clippy passed again.
