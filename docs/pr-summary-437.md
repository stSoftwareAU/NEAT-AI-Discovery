## Summary

Implements weight coherence validation for the "Brilliant but Brittle" initiative (Issue #432). This module validates that proposed weight configurations are coherent with network structure and won't create brittleness.

### Changes

1. **New module `src/analysis/weight_coherence.rs`** implementing three coherence checks:
   - **Incoherent weight ratios**: Detects neurons with very large incoming weights but tiny outgoing weights (ratio > 100x by default). Such patterns create inefficient amplification/attenuation that makes the network brittle.
   - **Near-constant output paths**: Detects neurons producing near-constant output regardless of input variation (variance < 0.01 by default). This indicates the neuron adds no meaningful information.
   - **Symmetric weight cancellation**: Detects opposite-sign weights from highly correlated inputs (correlation > 0.8) that cancel meaningful signal, causing instability when correlations shift.

2. **Configurable thresholds** via `WeightCoherenceConfig`:
   - `max_weight_ratio`: Maximum allowed incoming/outgoing weight ratio (default: 100.0)
   - `min_activation_variance`: Minimum variance to consider output non-constant (default: 0.01)
   - `min_correlation_for_cancellation`: Minimum correlation to flag symmetric cancellation (default: 0.8)
   - `min_samples`: Minimum samples required for detection (default: 20)

3. **Integration with existing weight validation**:
   - Builds on existing `weights.rs` guard rails (MAX_OUTGOING_WEIGHT, MIN_WEIGHT_RATIO)
   - Integrates with `utils/mod.rs` sensible range filtering
   - Follows the discovery module dispatch pattern from Issue #375

4. **Candidate generation** producing coordinated structural candidates:
   - `setWeight` candidates to rebalance incoherent ratios
   - `setBias` candidates to shift operating point away from saturation
   - `setWeight` candidates to reduce symmetric cancellation

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. The feature is validated through comprehensive unit tests.

## Test Plan

Added 16 tests in `tests/issue_437_weight_coherence_validation.rs`:

1. `test_detects_large_incoming_tiny_outgoing` - Verifies detection of incoherent weight ratios
2. `test_balanced_weight_ratio_not_flagged` - Ensures balanced ratios are not flagged
3. `test_detects_near_constant_output_path` - Verifies detection of near-constant outputs
4. `test_variable_output_not_flagged_as_constant` - Ensures variable outputs are not flagged
5. `test_detects_symmetric_weight_cancellation` - Verifies detection of correlated input cancellation
6. `test_uncorrelated_opposite_weights_not_flagged` - Ensures uncorrelated inputs are not flagged
7. `test_minimum_samples_required` - Validates minimum sample requirement
8. `test_incoherent_ratio_produces_setweight_candidate` - Tests setWeight candidate generation
9. `test_near_constant_path_produces_setbias_candidate` - Tests setBias candidate generation
10. `test_symmetric_cancellation_produces_coordinated_candidate` - Tests coordinated candidate output
11. `test_configurable_ratio_threshold` - Tests threshold configuration
12. `test_candidate_includes_diagnostic_comment` - Ensures diagnostic comments are included
13. `test_input_output_excluded_from_ratio_check` - Verifies only hidden neurons are checked
14. `test_empty_records_handled` - Ensures empty records don't cause panic
15. `test_multiple_incoherent_sorted_by_improvement` - Tests sorting by improvement
16. `test_detects_amplification_attenuation_pattern` - Tests detection with multiple incoming weights

Also added 4 unit tests in `src/analysis/weight_coherence.rs`:
- `test_config_default_values` - Default configuration values
- `test_is_saturating_squash` - Saturating activation function detection
- `test_correlation_identical_signals` - Correlation calculation for identical signals
- `test_correlation_opposite_signals` - Correlation calculation for opposite signals
- `test_correlation_insufficient_samples` - Insufficient samples handling

All tests pass with `./quality.sh`.
