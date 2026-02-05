## Summary

Implements cross-validation consistency scoring for the "Brilliant but Brittle" initiative (Issue #436). This feature detects candidates that perform well on discovery samples but may overfit, providing early detection of brittle predictions.

### Key Features

1. **K-fold cross-validation**: Splits samples into configurable folds and evaluates candidate performance across each fold to detect overfitting signatures.

2. **Brittleness penalty**: Computes a penalty based on performance variance across folds:
   ```
   penalty = min(1.0, variance / variance_threshold)
   adjusted_confidence = original_confidence × (1.0 - penalty)
   ```

3. **Configurable parameters**:
   - `fold_count`: Number of folds (default: 5)
   - `min_samples_per_fold`: Minimum samples required per fold (default: 15)
   - `variance_threshold`: Threshold for determining brittleness (default: 0.04)
   - `brittleness_penalty_max`: Maximum penalty to apply (default: 0.5)

4. **Integration with SPRT infrastructure**: Uses the existing `SequentialEvaluator` concepts from `early_termination.rs` for fold evaluation.

### New Files

- `src/analysis/cross_validation.rs` - Core module with types and functions
- `tests/cross_validation_test.rs` - Comprehensive test coverage (21 tests)

### API

```rust
use neat_ai_discovery::analysis::cross_validation::{
    CrossValidationConfig,
    CrossValidationResult,
    FoldResult,
    PerformanceVariance,
    compute_cross_validation_score,
    apply_brittleness_penalty,
};

// Configure cross-validation
let config = CrossValidationConfig::default();

// Compute cross-validation score
if let Some(cv_result) = compute_cross_validation_score(&samples, &config) {
    // Check if candidate is consistent across folds
    if cv_result.is_consistent(&config) {
        // Candidate generalises well
    } else {
        // Apply brittleness penalty to confidence
        let adjusted = apply_brittleness_penalty(confidence, cv_result.brittleness_penalty);
    }
}
```

## Evidence

Unable to generate screenshot: This is a library with no visual interface. The implementation is verified through comprehensive unit tests.

## Test Plan

Added 21 new tests in `tests/cross_validation_test.rs`:

### Configuration Tests
- `test_cross_validation_config_default` - Verifies default config values
- `test_cross_validation_config_with_custom_folds` - Tests custom configuration
- `test_cross_validation_config_validation_split` - Validates split ratios

### FoldResult Tests
- `test_fold_result_improvement_ratio` - Tests ratio calculation
- `test_fold_result_empty` - Tests empty fold handling

### PerformanceVariance Tests
- `test_performance_variance_calculation` - Tests variance computation
- `test_performance_variance_consistent_folds` - Tests identical folds
- `test_performance_variance_high_variance` - Tests high variance detection

### CrossValidationResult Tests
- `test_cross_validation_result_is_consistent` - Tests consistency detection
- `test_cross_validation_result_is_brittle` - Tests brittleness detection

### Core Function Tests
- `test_compute_cross_validation_score_insufficient_samples` - Tests sample validation
- `test_compute_cross_validation_score_consistent_candidate` - Tests consistent samples
- `test_compute_cross_validation_score_overfitting_candidate` - Tests overfitting scenario

### Penalty Tests
- `test_apply_brittleness_penalty_no_penalty` - Tests zero penalty
- `test_apply_brittleness_penalty_partial_penalty` - Tests partial penalty
- `test_apply_brittleness_penalty_max_penalty` - Tests maximum penalty
- `test_apply_brittleness_penalty_bounds` - Tests result bounds

### Integration Tests
- `test_cross_validation_with_sprt_early_termination` - Tests SPRT integration
- `test_brittleness_penalty_integrates_with_confidence` - Tests confidence integration

### Edge Cases
- `test_cross_validation_single_fold` - Tests single fold edge case
- `test_cross_validation_empty_samples` - Tests empty samples

All 21 tests pass with `./quality.sh`.
