## Summary

Implements sample-weighted discovery (Issue #423) to prioritise high-error samples during analysis. Previously, all samples were treated equally. Now, samples are weighted by absolute error magnitude, and neurons with disproportionately high weighted error are identified as candidates for improvement.

### What changed

- **New module `src/analysis/sample_weighted.rs`**: Implements error-weighted sample analysis with three core functions:
  - `compute_sample_weights()` — normalised importance weights proportional to absolute error
  - `stratify_samples()` — separates samples into easy (low-error) and hard (high-error) strata based on median error
  - `detect_high_error_neurons()` — identifies neurons with weighted mean error exceeding a configurable threshold
  - `high_error_neurons_to_coordinated_candidates()` — converts detections to `setBias` coordinated candidates

- **Pipeline integration in `src/analysis/mod.rs`**: Registered as a standard discovery dispatch module, running on all neuron records

- **Configurable via `SampleWeightedConfig`**: `min_weighted_error` (default: 0.25) and `min_samples` (default: 10) thresholds

### Design decisions

- Uses absolute error magnitude for weighting (simpler and more robust than squared error)
- Stratification uses median split rather than percentile-based, avoiding arbitrary threshold choices
- Generates `setBias` candidates since high-error neurons often benefit from operating point shifts
- Estimated improvement scales with both weighted error and hard-to-easy ratio, capped at 0.1

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

Added 17 integration tests in `tests/issue_423_sample_weighted_discovery.rs`:

1. **Sample weighting**: Weights proportional to error, uniform errors produce equal weights, empty records handled
2. **Detection**: High-error neurons detected, low-error neurons not flagged, insufficient samples return empty
3. **Stratification**: Easy/hard separation verified, uniform errors produce balanced ratio, empty records handled
4. **Candidate conversion**: Positive improvement, comments present, operations non-empty, sorted by improvement
5. **Weighted improvement**: Higher error produces higher estimated improvement
6. **Edge cases**: Single sample, NaN/infinity errors, zero errors, bimodal error patterns
7. **Configuration**: Custom thresholds affect detection sensitivity

All 17 tests pass. Full `quality.sh` passes (fmt, clippy, check, test, release build).
