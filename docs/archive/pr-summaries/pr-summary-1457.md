# Document undocumented crate-root public API items

## Summary

Added `///` doc comments to the public API items that were re-exported from the
crate root but carried no documentation, closing the isolated gaps flagged by
the `rust` bucket's "doc comments on public API" check and the Rust API
Guidelines (C-EXAMPLE / C-FAILURE). Closes #1457.

Items documented:

- `focus::ranking::rank_focus_neurons` (`src/focus/ranking/mod.rs`) — added a
  summary, a `# Errors` section, and a `# Examples` block (the primary public
  focus-selection entry point).
- `focus::ranking::RankFocusStats` (`src/focus/ranking/mod.rs`) — struct summary.
- `focus::ranking::RankedNeuron` (`src/focus/ranking/score_calculation.rs`) —
  struct summary.
- `parquet_format::ParquetRecordWriter::write_records` and `::finish`
  (`src/parquet_format/writer.rs`) — summaries plus `# Errors` sections for the
  fallible I/O methods.
- `ffi_types::RankFocusNeuronsInput` and `ffi_types::MergeParquetInput`
  (`src/ffi_types/requests.rs`) — struct summaries.
- `ffi_types::CandidateNeuronJson` and `ffi_types::RankedNeuronJson`
  (`src/ffi_types/candidates.rs`) — struct summaries.

The example uses the crate's existing `rust,ignore` doctest convention (the
crate requires a GPU and real Parquet/creature fixtures, so executing the
example is not feasible in `cargo test`).

The issue's optional suggestion to enable `#![warn(missing_docs)]` was **not**
adopted: turning it on would flag many other pre-existing public items across
the crate and is out of scope for this targeted documentation fix. This is
noted here so the suggestion is not lost.

This is a documentation-only change — no behaviour was modified. The patch
version was bumped (`0.74.102` → `0.74.103`) per the repository's version
invariant.

## Evidence

Backend/library change with no web interface to screenshot. Verification is via
the documentation build with warnings treated as errors:

```
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

builds cleanly, confirming every new doc comment (including the intra-doc links
such as `[`RankFocusStats`]` and `[`rank_focus_neurons_with_descriptor`]`)
resolves. The full `./quality.sh` gate (Clippy `-D warnings`, `cargo check`,
tests, doc build, release build) passes.

## Test Plan

- `./quality.sh` — full quality gate passes (fmt, Clippy with `-D warnings`,
  `cargo check`, `cargo test --lib --tests --all-features`, doc build with
  `RUSTDOCFLAGS="-D warnings"`, release build). The doc-build step is the
  executable verification that the new doc comments and their intra-doc links
  are valid.
- No unit-test changes are applicable: the change adds only `///` documentation
  and introduces no new behaviour to assert against. Per the repository testing
  guidelines, source-text-grepping "tests" are intentionally avoided.
