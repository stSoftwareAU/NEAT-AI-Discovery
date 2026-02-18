## Summary

Implement error distribution computation for neuron analysis, completing the long-standing TODO from Issue #192. Closes #486.

The `error_distribution` field in `NeuronAnalysisMetadata` was always set to `None`. This PR wires it up by collecting error values from target neuron records during the parallel analysis loop and computing `ErrorDistribution::from_errors()` from the aggregated samples — the same pattern already used by synapse analysis.

### Changes

- **`src/analysis/neuron.rs`**: Collect error values from each focus target neuron's records during the parallel loop via a shared `Mutex<Vec<f32>>`, then compute `ErrorDistribution::from_errors()` and populate the metadata field.

## Evidence

This is a backend-only change with no UI impact. Verified by:
- 3 new integration tests covering the main scenarios
- Full `quality.sh` pass (fmt, clippy, check, tests, release build)

## Test Plan

- `test_neuron_analysis_populates_error_distribution` — verifies error distribution is populated with correct statistical properties (mean, std_dev, skewness, percentiles) for a known outlier pattern
- `test_neuron_analysis_no_errors_returns_none` — verifies `None` is returned when target records have empty error vectors
- `test_neuron_analysis_error_distribution_multiple_targets` — verifies error samples are aggregated across multiple focus target neurons
