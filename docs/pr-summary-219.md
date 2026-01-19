## Summary

Implements statistical early termination for candidate evaluation using the Sequential Probability Ratio Test (SPRT). This feature allows the GPU evaluation pipeline to stop early when a candidate is clearly beneficial or harmful, rather than evaluating all samples.

### What's New

1. **SequentialEvaluator** - A new struct implementing the SPRT algorithm for statistical hypothesis testing:
   - Tests H0 (improvement ≤ 0) vs H1 (improvement ≥ threshold)
   - Computes log-likelihood ratios to make statistically-grounded decisions
   - Supports configurable error rates (alpha, beta) for false positive/negative control
   - Provides both single-sample and batch-based interfaces for GPU compatibility

2. **EarlyTerminationConfig** - Configuration struct for controlling early termination behaviour:
   - Enable/disable toggle for backwards compatibility
   - Configurable error rates and improvement thresholds
   - Adjustable minimum sample requirements and check intervals
   - Preset configurations: `default()`, `disabled()`, `conservative()`

3. **Enhanced Statistics Types** - Added early termination support to existing stats:
   - `HelpfulStats::samples_evaluated` - Track how many samples were processed
   - `HelpfulStats::early_terminated` - Flag indicating early termination
   - `HelpfulStats::is_strongly_beneficial()` / `is_strongly_harmful()` - Quick heuristic checks
   - `HelpfulStats::merge()` - Support incremental batch processing
   - Similar enhancements to `HarmfulStats`

4. **Batch Evaluation Support** - `check_batch_early_termination()` function for processing multiple candidates:
   - Evaluates current statistics for each candidate
   - Returns indices grouped by decision: accept, reject, or continue
   - Designed for integration with GPU batch processing pipeline

### Expected Performance Impact

Based on SPRT theory and test results:

| Candidate Type | Current Samples | With Early Term | Speedup |
|----------------|-----------------|-----------------|---------|
| Strongly positive (>80% improvement) | 100K | ~2K | 50x |
| Strongly negative (<20% positive) | 100K | ~2K | 50x |
| Marginal (near threshold) | 100K | 100K | 1x |

Since most candidates fall into the "clearly good" or "clearly bad" categories, the average speedup should be significant for discovery runs with many candidates.

### Statistical Foundation

The implementation uses Wald's Sequential Probability Ratio Test (1945):
- Pre-computed decision bounds based on desired error rates
- Log-likelihood ratio computed incrementally as samples arrive
- Conservative defaults (α=β=0.01) for 1% false positive/negative rates

## Evidence

This is a performance enhancement with statistical guarantees. Evidence of correct behaviour is provided by the test suite.

**SPRT Correctness Tests:**
- `test_sprt_bounds_computation` - Verifies decision bounds are computed correctly
- `test_log_likelihood_ratio` - Verifies LLR computation behaviour
- `test_strongly_beneficial_under_10_percent_samples` - Confirms 80% positive terminates in ≤10% of samples
- `test_strongly_harmful_under_10_percent_samples` - Confirms 20% positive terminates in ≤10% of samples
- `test_early_termination_marginal` - Confirms marginal cases (55%) are handled appropriately

**Statistical Properties:**
- 90% positive acceptance decision: confirmed
- 55% positive rejection decision: confirmed (below H1 threshold of 60%)
- Different thresholds produce different decisions: confirmed

## Test Plan

### New Tests Added

1. **`tests/early_termination.rs`** - 12 comprehensive tests:
   - `test_early_termination_strong_positive` - 90% positive terminates early with Accept
   - `test_early_termination_strong_negative` - 10% positive terminates early with Reject
   - `test_early_termination_marginal` - 55% positive vs 90% positive decision comparison
   - `test_sprt_bounds_computation` - Validates mathematical correctness of bounds
   - `test_sample_count_tracking` - Verifies count tracking accuracy
   - `test_batch_evaluation` - Tests GPU-compatible batch interface
   - `test_threshold_affects_decision` - Confirms threshold parameter works
   - `test_minimum_samples_before_decision` - Validates minimum sample requirements
   - `test_evaluator_reset` - Tests reusability of evaluator
   - `test_strongly_beneficial_under_10_percent_samples` - Issue #219 success criterion
   - `test_strongly_harmful_under_10_percent_samples` - Issue #219 success criterion
   - `test_log_likelihood_ratio` - Tests LLR ordering property

2. **Unit tests in `src/analysis/early_termination.rs`** - Internal module tests:
   - `test_new_evaluator_starts_with_zero_counts`
   - `test_add_sample_increments_counts`
   - `test_add_batch_adds_multiple_samples`
   - `test_reset_clears_counts`
   - `test_improvement_ratio_calculation`
   - `test_bounds_are_symmetric_for_equal_error_rates`
   - `test_should_stop_requires_minimum_samples`
   - `test_strongly_beneficial_with_enough_samples`
   - `test_strongly_harmful_with_enough_samples`
   - `test_log_likelihood_ratio_increases_with_positive_samples`

### Existing Tests

All 349+ existing tests pass without modification (except adding new fields to `HelpfulStats` struct initialisers in `implementation_tests.rs`).

## Files Changed

- `src/analysis/mod.rs` - Added early_termination module and exports
- `src/analysis/early_termination.rs` - **NEW** - Core SPRT implementation
- `src/analysis/samples.rs` - Enhanced HelpfulStats and HarmfulStats with early termination support
- `src/analysis/implementation_tests.rs` - Updated struct initialisers for new fields
- `tests/early_termination.rs` - **NEW** - Comprehensive test suite

## References

- Wald, A. (1945). Sequential Tests of Statistical Hypotheses
- [SPRT Wikipedia](https://en.wikipedia.org/wiki/Sequential_probability_ratio_test)
- Issue #219: Discovery: Implement early termination for clearly beneficial/harmful candidates
