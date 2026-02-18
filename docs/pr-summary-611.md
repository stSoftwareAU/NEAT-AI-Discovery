## Summary

Add end-to-end integration tests that exercise the full discovery pipeline from
recording through analysis to candidate output. These tests use realistic but
small synthetic creature data and verify that the pipeline produces sensible
candidates with positive expected improvement. Closes #611.

Five test scenarios are covered:

1. **Dead neuron** — a ReLU neuron with large negative bias (always zero
   activation) should be detected for removal
2. **Opposing synapses** — two paths with opposite weights that cancel out
   should produce removeSynapse, setWeight, or harmful synapse candidates
3. **Saturated neuron** — a TANH neuron stuck near +1 should trigger
   changeSquash, setBias, or new synapse/neuron candidates
4. **Minimal creature** — a single output neuron with no synapses should be
   handled gracefully without panicking
5. **Realistic multi-layer creature** — 3 hidden neurons, 5 synapses, varied
   activations (TANH, ReLU, LOGISTIC) exercises the full pipeline and verifies
   all returned candidates have positive expected improvement

## Evidence

This is a backend/CLI change with no visual output. Evidence is the test
results themselves — all 5 tests pass and `quality.sh` completes cleanly:

```
running 5 tests
test e2e_dead_neuron_produces_remove_neuron_candidate ... ok
test e2e_minimal_creature_handles_gracefully ... ok
test e2e_opposing_synapses_produce_removal_or_weight_candidate ... ok
test e2e_realistic_creature_produces_candidates_with_positive_improvement ... ok
test e2e_saturated_neuron_produces_squash_or_bias_candidate ... ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Test Plan

- Added `tests/issue_611_end_to_end_discovery_pipeline.rs` with 5 end-to-end
  integration tests
- All tests require GPU (use `GpuAnalyzer::gpu_is_available()` guard)
- Each test exercises the full pipeline: create creature → write parquet →
  `analyze_parallel_internal` → assert on candidates
- All existing tests continue to pass
- `cargo clippy` and `./quality.sh` pass cleanly
