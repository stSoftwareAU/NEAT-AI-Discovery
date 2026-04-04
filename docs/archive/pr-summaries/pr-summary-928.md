## Summary

Verify fan-in synapse pattern discovery end-to-end. Adds a scenario test that
builds a crippled creature with a missing hidden-B → hidden-A fan-in synapse
(reducing fan-in from 3 to 2) and verifies the discovery engine identifies the
missing connection. Closes #928.

## Evidence

All 3 new tests pass against the existing discovery engine with no code changes
required — the fan-in candidate generation (Issue #908) already handles this
scenario correctly:

```
test issue_928_verify_fan_in_synapse_discovery::issue_928_discovers_missing_fan_in_synapse ... ok
test issue_928_verify_fan_in_synapse_discovery::issue_928_candidate_identifies_correct_source_and_target ... ok
test issue_928_verify_fan_in_synapse_discovery::issue_928_all_candidates_have_positive_improvement ... ok
```

## Test Plan

- `issue_928_discovers_missing_fan_in_synapse` — verifies the engine returns an addSynapse candidate for hidden-B → hidden-A
- `issue_928_candidate_identifies_correct_source_and_target` — verifies correct neuron direction (hidden-B as source, hidden-A as target)
- `issue_928_all_candidates_have_positive_improvement` — verifies all returned candidates have positive expected score gain
