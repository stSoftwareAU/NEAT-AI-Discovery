## Summary

Verify change-squash discovery for suboptimal activation functions end-to-end. Closes #930.

Adds integration tests that construct a crippled creature where hidden-A's squash has been
degraded from TANH to IDENTITY (keeping all weights and biases identical). The tests verify
that the discovery engine's detection modules (high-error squash exploration, activation
mismatch, activation recommendation, squash weight rescale) correctly identify the suboptimal
IDENTITY activation and recommend a non-linear replacement via a `ChangeSquash` coordinated
structural candidate.

This enables the corresponding NEAT-AI scenario test (`DiscoveryScenarioChangeSquash.ts`) to
be un-ignored and pass end-to-end.

## Evidence

All 3 new tests pass on a GPU-equipped machine:

- `test_change_squash_discovery_for_suboptimal_activation` — verifies a `ChangeSquash`
  candidate is produced recommending a non-linear squash for hidden-A
- `test_change_squash_candidate_properties` — validates the candidate targets the correct
  neuron, recommends a different squash from IDENTITY, and has positive expected score gain
- `test_analyze_all_finds_change_squash_candidate` — verifies the full `analyze_all` pipeline
  (synapse + neuron analysis) also produces the change-squash candidate

No implementation changes were required — the existing detection modules already handle this
scenario correctly. This PR adds the verification tests that prove it works end-to-end.

## Test Plan

- Added `tests/analysis/issue_930_change_squash_suboptimal_activation.rs` with 3 tests
- Registered the module in `tests/analysis/main.rs`
- All existing tests continue to pass (no regressions)
- `./quality.sh` passes cleanly (fmt, clippy, check, tests, doc build, release build)
