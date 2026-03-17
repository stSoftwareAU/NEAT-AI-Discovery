## Summary

Add a fan-in weight polarity conflict detector that identifies hidden neurons
whose incoming synapses have sharply conflicting polarities — strongly positive
weights fighting strongly negative weights of similar magnitude. When this
occurs, most of the input signal cancels out, wasting representational capacity.

The detector produces coordinated structural candidates (`addNeuron` +
`addSynapse` + `removeSynapse`) to split the minority-polarity pathway into a
separate neuron, eliminating the internal cancellation.

This is distinct from:
- `opposing_synapse.rs` — detects same-source opposing pairs (narrow case)
- `weight_coherence.rs` — checks ratio consistency and correlated-source cancellation

Closes #641.

## Evidence

This is a backend detection module with no UI component. Correctness is verified
by the integration test suite below.

## Test Plan

- Added `tests/issue_641_fanin_polarity_conflict.rs` with 8 tests:
  1. Detects clear polarity conflict (balanced positive/negative fan-in)
  2. Healthy same-sign fan-in produces no candidates
  3. Tiny opposing weight is not flagged (below threshold)
  4. Candidate proposes `addNeuron` to split positive/negative pathways
  5. Insufficient samples produce no candidates
  6. Empty records produce no candidates
  7. Only hidden neurons are flagged (input/output excluded)
  8. Candidates sorted by conflict score (worst first)
- All existing tests continue to pass
- `./quality.sh` passes cleanly
