## Summary

Verify that the Rust discovery engine correctly identifies a harmful synapse
(hidden-A -> hidden-B, weight -2.0) for removal in a scenario test matching
NEAT-AI's `DiscoveryScenarioRemoveHarmfulSynapse.ts`. Closes #927.

The test builds a "crippled" creature where a harmful cross-connection from
hidden-A to hidden-B suppresses hidden-B's activation via a strong negative
weight. Discovery records use the `actual - target` error convention so the
GPU harmful shader correctly flags the synapse (signal and error have the
same sign). The existing harmful synapse evaluation pipeline handles this
scenario without any production code changes.

## Evidence

All 3 new tests pass (`cargo test --test synapse issue_927`):
- `test_remove_harmful_synapse_discovery` -- verifies the harmful synapse
  appears in either `harmful_synapses` or `coordinated_structural_candidates`
  with positive `expected_creature_score_gain`
- `test_harmful_synapse_identifies_correct_neurons` -- verifies the candidate
  identifies hidden-A as source and hidden-B as target
- `test_analyze_all_finds_harmful_synapse` -- verifies the full
  `analyze_all` pipeline also detects the harmful synapse

No production code changes; no regressions (154 synapse tests pass).

## Test Plan

- Added `tests/synapse/issue_927_remove_harmful_synapse.rs` with 3 tests
- Registered module in `tests/synapse/main.rs`
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
