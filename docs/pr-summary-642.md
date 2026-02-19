## Summary

Add a hard sample clustering detector that identifies groups of observations (`obs_index` values) with consistently high error across all output neurons, indicating systematic structural gaps. The module produces targeted `addNeuron` + `addSynapse` candidates connecting dominant inputs to all outputs, increasing network capacity for the hard sample region. Closes #642.

## Changes

- **New detection module**: `src/analysis/detection/hard_sample_cluster.rs`
  - Aggregates per-observation mean error across all output neurons
  - Uses statistical threshold (mean + 1 std dev) to classify hard observations
  - Identifies dominant input neurons whose activations discriminate hard vs easy groups
  - Produces coordinated structural candidates (addNeuron + addSynapse)
  - Handles single-output networks as a degenerate case
- **Wired into dispatch pipeline**: Added to `structural_specs.rs` for parallel execution
- **Module declarations**: Registered in `detection/mod.rs` and re-exported via `analysis/mod.rs`
- **Scenario documentation**: `docs/discoveries/hard-sample-cluster.md`

## Evidence

This is a backend detection module with no UI changes. Correctness is verified by the test suite below.

## Test Plan

- `tests/issue_642_hard_sample_cluster_detection.rs` — 10 integration tests:
  1. Detects hard sample cluster across multiple output neurons
  2. Single-output network works as degenerate case
  3. Uniform error produces no clusters
  4. Insufficient samples do not trigger detection
  5. Hard clusters produce coordinated structural candidates (addNeuron + addSynapse)
  6. Empty records produce no clusters
  7. Dominant inputs are identified correctly
  8. Estimated improvement is positive for detected clusters
  9. Multiple error values per record are handled
  10. Hard-to-easy ratio correctly reflects error disparity
- All existing tests pass (`./quality.sh` clean)
