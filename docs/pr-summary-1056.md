## Summary

Recalibrate prediction scoring to match production success rates from GRQ-sampler discovery cache (30+ creatures). Closes #1056.

The existing linear calibration factors and pessimism discount parameters were insufficient to bridge the neuron-level → creature-level prediction gap. This PR:

- **Updates calibration factors** based on empirical data: synapse 0.001→0.0003, neuron 0.01→0.003, coordinated 0.0001→0.00005
- **Adds non-linear (logistic) calibration** that modulates the base calibration factor using a sigmoid of the improved ratio, providing better correction than a flat multiplier — moderate improved ratios (0.3–0.6) are far more overestimated than high ratios (>0.8)
- **Updates pessimism discount parameters** to match production reality: neuron floor 0.10→0.08 with exponent 0.75→0.80, synapse floor 0.05→0.03 with exponent 0.85→0.90

## Evidence

GRQ-sampler production data:

| Candidate Type | Actual Success Rate | Previous Overestimation |
|---|---|---|
| add-neurons | ~2.7% (28/1028) | ~18× |
| coordinated-structural | ~1.1% (6/525) | orders of magnitude |
| add-synapses | ~0.1% (3/1001) | massive |

The logistic calibration function uses `sigmoid(steepness × (ratio - midpoint))` to produce:
- Low ratios (<0.3): modulator ≈ 0.12 → heavy additional reduction
- Moderate ratios (~0.5): modulator ≈ 0.37 → substantial reduction
- High ratios (>0.8): modulator ≈ 0.9+ → near-full base calibration

## Test Plan

- Added `tests/scoring/issue_1056_logistic_prediction_calibration.rs` with 12 tests:
  - Compile-time validation of constant ordering and ranges
  - Logistic calibration more aggressive than linear at moderate ratios
  - Logistic calibration approaches linear at high ratios
  - Heavy reduction at low ratios
  - Ordering preservation across different improved ratios
  - Edge cases: zero total count, negative gain, zero gain
  - Cross-type ranking (neuron above synapse)
  - Updated pessimism discount validation
  - End-to-end pipeline tests for neuron and synapse candidates
- Updated `tests/analysis/issue_938_constants_submodule_organisation.rs` to match new constant values
- All 260 scoring tests pass, all 171 synapse tests pass, full quality gate passes
