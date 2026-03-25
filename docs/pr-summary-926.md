## Summary

Verify that the discovery engine correctly produces add-neuron candidates between
existing hidden neurons (not just input→hidden→output). Closes #926.

The NEAT-AI scenario test `DiscoveryScenarioAddNeuronBetweenHidden.ts` requires
the Rust discovery engine to find a missing neuron between two hidden neurons.
This PR adds end-to-end integration tests that confirm the engine already handles
this case correctly — no implementation changes were needed.

The tests create a "crippled creature" with hidden-B removed from the chain
`hidden-A → hidden-B (TANH, bias 0.3) → hidden-C`, replacing it with a direct
`hidden-A → hidden-C` connection. The engine successfully identifies candidates
to insert an intermediate neuron, validating:

- `analyze_neurons` finds hidden-A → hidden-C add-neuron candidates
- `analyze_all` pipeline evaluates hidden neuron targets end-to-end
- Candidate properties (weights, squash, impact, improved count) are valid

## Evidence

All 3 new tests pass with GPU, confirming hidden-to-hidden neuron discovery works:
- `test_add_neuron_between_hidden_neurons` — verifies candidates from hidden-A to hidden-C
- `test_hidden_neuron_candidate_properties` — validates weight/squash/impact properties
- `test_analyze_all_finds_hidden_neuron_candidate` — confirms full pipeline handles hidden targets

## Test Plan

- Added `tests/neuron/issue_926_add_neuron_between_hidden.rs` with 3 integration tests
- Registered module in `tests/neuron/main.rs`
- All existing tests continue to pass (no regressions)
