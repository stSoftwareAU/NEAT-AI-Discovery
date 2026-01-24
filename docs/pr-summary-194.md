## Summary

This PR adds confidence intervals to discovery predictions, addressing Issue #194. The new feature helps callers:
- Prioritise high-confidence candidates
- Understand prediction uncertainty
- Filter out unreliable predictions

### New Fields

Synapse and neuron candidates now include:
- `predictionConfidence` (f32): Overall confidence score (0.0 to 1.0)
- `expectedScoreGainConfidenceInterval` ([f32; 2]): 95% confidence interval [lower, upper] for the expected score gain

### Confidence Calculation

The overall confidence is computed as a geometric mean of three factors:

1. **Sample confidence**: `min(1.0, sample_count / 100)` - More samples = higher confidence
2. **Variance confidence**: `min(1.0, source_std_dev / 0.05)` - Higher source variance = more reliable correlation
3. **Model fit confidence**: R² when available (optional)

The confidence interval width is inversely proportional to sample count and source variance, meaning:
- More samples → narrower interval
- Higher source variance → narrower interval

### Implementation Details

- Created new `src/analysis/confidence.rs` module with:
  - `compute_confidence_metrics()` function
  - `PredictionConfidenceMetrics` struct
- Added confidence fields to `CandidateSynapseJson` and `CandidateNeuronJson` in `lib.rs`
- Integrated confidence calculation into:
  - Synapse analysis (`implementation.rs`)
  - Neuron analysis (`synapse.rs`, `samples.rs`)

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface.

The feature is demonstrated through comprehensive tests that verify:
- Fields exist in JSON output
- Confidence values are valid probabilities (0.0 to 1.0)
- Confidence intervals contain the point estimate
- Higher sample counts increase confidence
- Higher source variance increases confidence
- Confidence interval narrows with more samples

## Test Plan

### New Tests Added (tests/issue_194_confidence_intervals.rs)
- `test_prediction_confidence_field_exists_on_synapse_candidates`
- `test_score_gain_confidence_interval_field_exists_on_synapse_candidates`
- `test_prediction_confidence_field_exists_on_neuron_candidates`
- `test_score_gain_confidence_interval_field_exists_on_neuron_candidates`
- `test_prediction_confidence_is_valid_probability`
- `test_confidence_interval_lower_bound_le_expected`
- `test_confidence_interval_upper_bound_ge_expected`
- `test_more_samples_higher_confidence`
- `test_low_variance_source_lower_confidence`
- `test_confidence_interval_narrows_with_more_samples`

### Unit Tests (src/analysis/confidence.rs)
- `test_sample_confidence_*` - Sample size factor tests
- `test_variance_confidence_*` - Source variance factor tests
- `test_model_fit_confidence` - R² factor test
- `test_overall_confidence_*` - Combined confidence tests
- `test_confidence_interval_*` - Interval bounds tests

### Documentation
- Updated README.md with new section documenting the confidence interval feature

All existing tests continue to pass. The `quality.sh` script passes cleanly.
