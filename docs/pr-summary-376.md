## Summary

Added unit tests for all 9 discovery detection modules, which previously had zero test coverage. Each module now has tests covering positive detection, negative detection (exclusion criteria), edge cases, and conversion to coordinated structural candidates.

Closes #376

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. All changes are verified through unit tests.

## Test Plan

### New test files (9 modules × 3–11 tests each = 76 total tests):

- **`src/analysis/saturation_tests.rs`** (11 tests)
  - TANH saturated near positive bound is detected
  - LOGISTIC saturated near zero is detected
  - RELU dead-zone neuron is detected
  - TANH in active region is not detected
  - IDENTITY neuron is never detected
  - High activation variance prevents detection
  - Insufficient samples prevents detection
  - Empty neuron list returns empty
  - Neuron with no matching records is skipped
  - Conversion produces ChangeSquash and SetBias operations
  - Conversion empty candidates returns empty

- **`src/analysis/bottleneck_tests.rs`** (5 tests)
  - Hidden neuron with high fan-in/low fan-out is detected
  - Balanced fan-in/fan-out is not detected
  - Output neuron is not detected as bottleneck
  - Insufficient samples prevents detection
  - Conversion produces AddNeuron and AddSynapse operations

- **`src/analysis/dead_neuron_tests.rs`** (7 tests)
  - Neuron with zero activation across all samples is detected
  - Neuron with meaningful activation is not detected
  - Output neuron is not detected as dead
  - Neuron active on small fraction is not detected
  - Insufficient samples prevents detection
  - Connected outputs are identified via BFS
  - Conversion produces RemoveNeuron operation

- **`src/analysis/correlated_error_tests.rs`** (5 tests)
  - Two outputs with highly correlated errors are grouped
  - Uncorrelated outputs are not grouped
  - Single output neuron returns empty
  - Insufficient samples prevents detection
  - Conversion produces AddNeuron and AddSynapse operations

- **`src/analysis/multi_hop_tests.rs`** (7 tests)
  - Unconnected neuron correlating with target error is detected
  - Already-connected neuron is excluded
  - Uncorrelated neuron is not detected
  - Empty records returns empty
  - Insufficient samples prevents detection
  - Two-hop conversion produces AddSynapse
  - Three-hop conversion produces AddNeuron and synapses

- **`src/analysis/oscillating_neuron_tests.rs`** (8 tests)
  - Neuron with frequent sign changes is detected
  - LOGISTIC neuron recommends RELU
  - Consistent sign is not detected
  - Near-dead neuron is not detected as oscillating
  - Heavily biased sign is not detected
  - Insufficient samples prevents detection
  - Bias recommendation shifts toward minority sign
  - Conversion produces ChangeSquash operation

- **`src/analysis/dormant_synapse_tests.rs`** (6 tests)
  - Near-zero weight with low contribution is detected
  - Significant weight is not detected
  - Sole input synapse is not detected (would be destructive)
  - Insufficient samples prevents detection
  - Empty creature returns empty
  - Conversion produces RemoveSynapse operation

- **`src/analysis/opposing_synapse_tests.rs`** (7 tests)
  - Contribution positively correlated to error is detected
  - Negative correlation is not detected
  - Synapse to hidden neuron is not detected
  - Dormant synapse is not detected as opposing
  - Insufficient aligned samples prevents detection
  - High correlation recommends removal
  - Moderate correlation recommends weight flip

- **`src/analysis/output_bias_drift_tests.rs`** (8 tests)
  - Consistently positive errors detected
  - Consistently negative errors detected
  - Balanced errors not detected
  - Hidden neuron not detected
  - Small error magnitude not detected
  - Insufficient samples prevents detection
  - Records without errors are skipped
  - Conversion produces SetBias with adjusted value

### Verification
- `./quality.sh` passes (fmt, clippy, check, tests, release build)
