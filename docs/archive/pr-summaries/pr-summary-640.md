## Summary

Add pre-activation distribution shape detector that identifies hidden neurons with bimodal or multi-modal pre-activation (`value`) distributions. A bimodal distribution indicates the neuron is serving two distinct input regimes and should be split into two specialised neurons.

This is distinct from oscillating neuron detection (Issue #358, post-activation sign changes) and activation mismatch detection (Issue #543, RELU negative fraction). Bimodal detection examines the **shape** of the pre-activation distribution using a gap-based approach.

Closes #640.

## Detection Algorithm

Uses a gap-based bimodality detection:
1. Sort pre-activation values
2. Compute gaps between consecutive sorted values
3. Find the largest gap respecting minimum cluster size constraints
4. If the largest gap exceeds 5× the median gap, the distribution is bimodal

This approach is robust against false positives on uniform and unimodal distributions.

## Candidates Produced

- `addNeuron` coordinated structural candidates to split bimodal neurons
- New neuron is biased toward the lower mode, inserted before the original neuron

## Evidence

This is a backend detection module with no UI changes. Verified via 14 integration tests covering:
- Bimodal detection (two well-separated clusters)
- Unimodal rejection (uniform/linear distributions)
- None value handling (graceful skip)
- Insufficient sample rejection
- Coordinated candidate generation (`addNeuron` operations)
- Multi-modal detection (3+ modes)
- Overlapping cluster rejection
- Mixed neuron filtering
- Sorting by estimated improvement
- Edge cases (empty records, missing neurons, partial None values)

## Test Plan

- Added `tests/issue_640_bimodal_neuron_detection.rs` with 14 tests
- All existing tests continue to pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
