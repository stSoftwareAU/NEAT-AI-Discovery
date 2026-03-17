## Summary

Add stress tests for large creature analysis (1,000+ neurons, 5,000+ synapses) to verify the pipeline handles scale without panics, OOM errors, or unbounded memory growth. All tests are marked `#[ignore]` so they do not slow CI but can be run on demand. Closes #612.

## Evidence

This is a backend/test-only change with no visual output. Evidence:

- All 5 stress tests are detected by cargo: `cargo test --test issue_612_stress_tests_large_creature_analysis -- --list`
- The 1,000-neuron stress test (`stress_1000_neurons_pipeline_completes_without_panic`) was executed and passed in ~658s
- `./quality.sh` passes cleanly with all existing tests still passing

## Tests Added

File: `tests/issue_612_stress_tests_large_creature_analysis.rs`

| Test | Description |
|------|-------------|
| `stress_1000_neurons_pipeline_completes_without_panic` | 1,000 hidden neurons (~3,000 synapses), verifies pipeline completes |
| `stress_2000_neurons_5000_plus_synapses` | 2,000 hidden neurons (5,000+ synapses), exceeds issue threshold |
| `stress_memory_no_unbounded_growth` | Runs pipeline 3 times, asserts memory growth stays under 100 MB |
| `stress_gpu_buffers_handle_large_creature` | 1,500 neurons with 100 observations, verifies GPU was used |
| `stress_mixed_activations_at_scale` | 1,000 neurons with 5 activation types, exercises all GPU shader paths |

### Running the stress tests

```sh
cargo test --test issue_612_stress_tests_large_creature_analysis -- --ignored --test-threads=1
```

## Test Plan

- [x] All 5 stress tests compile and are detected by cargo
- [x] `stress_1000_neurons_pipeline_completes_without_panic` executed and passed
- [x] All existing tests pass (`./quality.sh` clean)
- [x] Tests are marked `#[ignore]` — not included in normal CI runs
