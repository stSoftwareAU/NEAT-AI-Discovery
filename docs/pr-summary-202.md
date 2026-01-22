# PR Summary: Pre-detect epistatic neuron pairs during analysis (#202)

## Summary

This PR adds proactive detection of **epistatic neuron pairs** during synapse analysis. Epistatic changes are structural modifications where no single operation improves the score, but a group of operations does. Previously, these relationships were only discovered post-hoc through weight adjustments. Now they are pre-detected during analysis.

### Key Changes

1. **New module: `src/analysis/epistatic.rs`**
   - Implements complementary pattern detection to find neuron pairs with non-overlapping firing patterns
   - Computes combined improvement estimates for identified pairs
   - Converts detected pairs to coordinated structural candidates

2. **Integration in `src/analysis/implementation.rs`**
   - Tracks source contributions during helpful synapse processing
   - Detects epistatic pairs after processing each focus target
   - Adds detected pairs to `coordinatedStructuralCandidates` output

3. **Detection Strategy**
   - Identifies source neurons where one fires on samples the other doesn't (complementarity >= 70%)
   - Estimates combined improvement as roughly the sum of individual improvements
   - Only reports pairs where combined improvement exceeds best individual improvement

### Example

If `input-0` correlates with positive error on the first half of samples and `input-1` correlates on the second half:
- Individual improvements: ~27% each (each helps only half the samples)
- Combined improvement: ~55% (both together help all samples)
- Result: Epistatic pair candidate with two `addSynapse` operations

## Evidence

Unable to generate screenshot: This is a library with no visual interface. The feature is validated through unit tests.

## Test Plan

Added comprehensive test coverage in `tests/issue_202_epistatic_neuron_pairs.rs`:

1. **`epistatic_pair_detected_for_complementary_error_patterns`**
   - Tests detection of perfectly complementary patterns (first half / second half split)
   - Verifies both inputs appear in coordinated candidate

2. **`epistatic_pair_detected_via_correlation_analysis`**
   - Tests detection with partial overlap patterns (~75% complementarity)
   - Validates combined improvement exceeds individual

3. **`no_false_positive_epistatic_detection_for_independent_neurons`**
   - Tests that neurons with identical firing patterns are NOT grouped as epistatic
   - Both should appear as independent helpful synapse candidates

4. **`epistatic_pair_candidate_has_expected_structure`**
   - Validates JSON structure of epistatic candidates
   - Checks for required fields: operations, expectedCreatureScoreGain, comment

5. **`epistatic_detection_integrates_with_focus_ranking`**
   - Tests epistatic detection works correctly with multiple focus neurons
   - Verifies analysis completes for all targets

### Unit Tests in `src/analysis/epistatic.rs`

- `test_compute_complementarity_*`: Validates complementarity calculation
- `test_compute_firing_indices`: Validates firing index computation
- `test_detect_epistatic_pairs_insufficient_sources`: Tests edge case handling
- `test_epistatic_pairs_to_coordinated_candidates`: Tests JSON conversion

All tests pass with `./quality.sh`.
