## Summary

Parallelise `compress_identity_candidates` and `compress_nonlinear_candidates` in
`src/analysis/orchestration.rs` using `rayon::join`. Both functions take immutable
references to the same data, return independent results, and are CPU-bound with no
side effects — a textbook case for concurrent execution. Closes #1003.

## Evidence

Both compression functions operate on `&syn.helpful_synapses` and `&input.creature`
(shared immutable references) and produce independent `Vec<CoordinatedStructuralCandidateJson>`
results. The `rayon::join` call allows them to run on separate threads when a rayon
worker is available, reducing wall-clock time for the compression phase.

## Test Plan

- Added `tests/analysis/issue_1003_parallel_candidate_compression.rs` with 3 tests:
  - `test_parallel_compression_matches_sequential` — verifies parallel results match sequential
  - `test_concurrent_compression_no_data_race` — runs 10 iterations to confirm no data races
  - `test_parallel_combined_output_matches_sequential` — verifies combined output is identical
- All existing tests pass with `cargo test --test-threads=1`
- `quality.sh` passes cleanly
