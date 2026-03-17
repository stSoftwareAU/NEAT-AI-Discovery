## Summary

Add cluster coherence validation to bimodal neuron detection to reduce false positives from skewed unimodal distributions. Closes #751.

After the existing gap-based detection finds a candidate split point, a new validation step checks that both resulting clusters have variance below 50% of the overall variance (`MAX_CLUSTER_VARIANCE_RATIO = 0.5`). This follows the Hartigan Dip Test principle: true bimodal distributions have clusters with variance far below the overall (typically < 10%), while false positives from scattered outlier groups or heavy-tailed distributions have cluster variance near or above 50% of overall.

## Evidence

No UI changes — backend detection logic only. Verified via test results:

- All 14 existing bimodal tests continue to pass (no regression)
- 5 new tests added covering the issue's TDD plan
- `quality.sh` passes cleanly.

## Test Plan

New test file: `tests/issue_751_bimodal_cluster_coherence.rs`

- `test_uniform_distribution_not_bimodal` — uniform distribution correctly rejected
- `test_bimodal_equal_modes_detected` — equal-sized bimodal correctly detected
- `test_bimodal_unequal_modes_detected` — 70/30 split bimodal correctly detected
- `test_unimodal_with_scattered_outlier_group_not_bimodal` — scattered outlier group (80 points in [0,1] + 20 in [10,50]) correctly rejected by coherence check
- `test_heavy_tail_skewed_not_bimodal` — heavy-tailed distribution (80 points in [0,2] + 20 in [15,300]) correctly rejected by coherence check
