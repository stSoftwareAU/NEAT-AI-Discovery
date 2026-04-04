## Summary

Add end-to-end scenario test verifying the discovery engine can find a missing
synapse between two hidden neurons. Closes #925.

The test constructs a crippled creature with the hidden-A → hidden-B synapse
removed and verifies that the discovery engine identifies this missing
hidden-to-hidden connection. The engine correctly discovers the missing synapse
via multi-hop analysis, returning it as a coordinated structural candidate with
an `addSynapse` operation from hidden-A to hidden-B with positive expected
score gain.

## Evidence

The discovery engine already supports hidden-to-hidden synapse discovery through
multiple pathways:

- **Multi-hop analysis** detects the high correlation (0.90) between hidden-A's
  activation and hidden-B's error, producing an `addSynapse` coordinated
  structural candidate
- **Hidden source interleaving** (Issue #907) ensures hidden neurons are
  evaluated as sources even under deadline pressure
- **Hidden source boost** (Issue #910) gives hidden-sourced candidates a 1.2×
  boost to compete with input-sourced candidates

No code changes were needed to the discovery engine — the capability was already
present but not verified end-to-end.

## Test Plan

- Added `tests/analysis/issue_925_add_synapse_between_hidden_neurons.rs` with 3 tests:
  - `issue_925_discovers_missing_hidden_to_hidden_synapse` — verifies the engine
    finds the hidden-A → hidden-B synapse in either `helpfulSynapses` or
    `coordinatedStructuralCandidates`
  - `issue_925_candidate_identifies_correct_source_and_target` — verifies the
    candidate has the correct direction (hidden-A as source, hidden-B as target)
  - `issue_925_all_candidates_have_positive_improvement` — verifies all returned
    candidates have positive expected score gain
- All existing tests continue to pass (384 tests, 0 regressions)
- Full quality gate passes (`./quality.sh`)
