## Summary

Add a new low-impact neuron detection module that expands remove-low-impact candidate generation by catching neurons in the activation range between the dead threshold (1e-6) and a new low-impact ceiling (1e-3). Closes #793.

The existing dead neuron detector uses a very tight threshold (1e-6). Neurons with activations slightly above this (e.g. 1e-5 to 1e-4) still have negligible impact on the network but were previously missed. The new module uses a tiered confidence approach based on activation magnitude, variance consistency, and sample count to identify these low-impact neurons as removal candidates.

## Changes

- **New module**: `src/analysis/detection/low_impact_neuron.rs` — detects hidden neurons with mean absolute activation between 1e-6 and 1e-3 with low variance, converts them to `RemoveNeuron` coordinated structural candidates
- **Dispatch wiring**: Added to `neuron_specs.rs` as a new discovery module spec (`low_impact_neuron_detection`)
- **Module count**: Updated from 44 to 45 discovery modules

## Evidence

- All 10 new unit tests pass covering: detection of low-impact neurons, exclusion of dead/active/input/output neurons, confidence scaling with sample count and activation magnitude, coordinated candidate conversion, insufficient samples, and high-variance exclusion
- `quality.sh` passes cleanly (fmt, clippy, check, tests, docs, release build)

## Test Plan

- Added `tests/issue_793_low_impact_neuron_detection.rs` with 10 tests:
  1. Detects neuron with activation in low-impact range (1e-5)
  2. Does not detect truly dead neurons (below 1e-6)
  3. Does not detect active neurons (above 1e-3)
  4. Confidence increases with more samples
  5. Confidence increases with lower activation
  6. Output and input neurons excluded
  7. Candidates produce coordinated RemoveNeuron operations
  8. Mixed network: only low-impact neurons detected
  9. Insufficient samples not detected
  10. Low mean but high variance not detected
