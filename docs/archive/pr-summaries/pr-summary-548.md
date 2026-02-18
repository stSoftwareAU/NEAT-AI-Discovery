## Summary

Add coordinated squash + weight rescale detection module for local minimum escape. When recommending an activation function change for hidden neurons, the module computes compensating weight rescaling to preserve the neuron's operating point, then bundles `changeSquash` with `setWeight` adjustments as a single coordinated structural candidate. This gives the network a better starting point after the structural change, rather than requiring many evolutionary steps to re-tune weights. Closes #548.

### How it works

1. For each hidden neuron with sufficiently high error, evaluate candidate squash functions
2. For each candidate squash, perform a grid search over rescale factors to find the optimal weight compensation that minimises error against the target
3. Bundle the squash change with compensating `setWeight` operations for all incoming synapses
4. Apply a coordinated boost factor (1.5×) to the expected gain, reflecting the higher likelihood of immediate benefit from preserving the operating point

### Key design decisions

- **Grid search over rescale factors** rather than analytical computation — more robust when squash functions have different shapes and the relationship between old and new activation is non-linear
- **Hidden neurons only** — output neurons already have coordinated squash+bias support from Issue #547
- **Aggregate squashes skipped** — IF, MINIMUM, MAXIMUM, etc. cannot be simulated as `f(x)` and are excluded

## Evidence

This is a backend/logic change with no visual output. Evidence is provided by the test suite:

- `tests/issue_548_squash_weight_rescale.rs` — 7 integration tests covering all acceptance criteria
- All 496 unit tests + 100+ integration tests pass via `quality.sh`

## Test Plan

| Test | Verifies |
|------|----------|
| `test_coordinated_candidate_has_squash_and_weight_operations` | Coordinated candidates include both `changeSquash` and `setWeight` operations |
| `test_weight_rescaling_preserves_operating_point` | Rescaled weights are finite and valid |
| `test_improved_gain_over_standalone_squash` | Coordinated candidates have positive expected gain |
| `test_no_candidates_for_well_suited_network` | No candidates when error is already low |
| `test_multiple_incoming_synapses_get_weights` | Each incoming synapse gets a compensating `setWeight` |
| `test_aggregate_squash_skipped` | Aggregate squashes (IF, etc.) produce no candidates |
| `test_insufficient_samples_no_candidates` | Insufficient sample count produces no candidates |
