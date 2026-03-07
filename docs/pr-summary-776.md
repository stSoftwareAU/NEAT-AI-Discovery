## Summary

Remove HashMap collections used solely for iteration in `weight_coherence.rs`. Closes #776.

Two changes:
1. **`hidden_squash` HashMap eliminated** — `detect_near_constant_paths` previously built a `HashMap<&str, &str>` mapping neuron UUID to squash function, then iterated hidden UUIDs and looked up each one. Replaced with direct iteration over `creature.neurons` filtered by `topo.hidden_uuids.contains()`, accessing the squash field directly from the neuron.
2. **`synapses_by_target` HashMap eliminated** — `detect_symmetric_cancellation` previously built a `HashMap<&str, Vec<(&str, f32)>>` grouping synapses by target, then iterated it. Replaced with direct iteration over `topo.fan_in`, building the incoming synapses vec inline per target.

Both changes avoid unnecessary hash computation during construction of collections that were only iterated.

## Evidence

This is a backend refactoring with no visual changes. All existing tests pass, plus 4 new tests verify the refactored functions produce correct results.

## Test Plan

- Added `tests/issue_776_hashset_hashmap_iteration.rs` with 4 tests:
  - `test_near_constant_paths_detects_saturated_hidden_neuron` — verifies detection of saturated TANH neurons
  - `test_near_constant_paths_skips_variable_neurons` — verifies neurons with sufficient variance are not flagged
  - `test_symmetric_cancellation_detects_opposite_correlated_inputs` — verifies detection of correlated opposite-weight inputs
  - `test_symmetric_cancellation_skips_uncorrelated_inputs` — verifies uncorrelated inputs are not flagged
- All 16 existing weight coherence tests continue to pass
- `quality.sh` passes cleanly
