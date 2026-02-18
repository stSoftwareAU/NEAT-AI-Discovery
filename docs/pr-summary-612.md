## Summary

Add stress tests for large creature analysis with 1,000+ neurons and 5,000+ synapses to verify the full discovery pipeline handles scale correctly without panics, OOM errors, or unbounded memory growth. Closes #612.

All five stress tests are marked `#[ignore]` so they do not slow CI but can be run on demand with:

```bash
cargo test --release -- --ignored --test-threads=1
```

## Tests Added

| Test | Neurons | Synapses | Observations | What it verifies |
|------|---------|----------|--------------|------------------|
| `stress_1000_neurons_pipeline_completes_without_panic` | 1,001 | 25,000+ | 50 | Full pipeline at 1K neuron scale |
| `stress_2000_neurons_pipeline_completes_without_panic` | 2,001 | 50,000+ | 30 | GPU buffer allocation at 2K scale |
| `stress_memory_stable_across_repeated_analyses` | 1,001 | 25,000+ | 40–60 | No unbounded memory growth (5 iterations) |
| `stress_dense_connectivity_no_panics` | 501 | 7,500+ | 40 | High fan-in/fan-out synapse handling |
| `stress_many_observations_no_panics` | 201 | 2,000+ | 500 | Large parquet record volumes |

## Evidence

This is a backend/test-only change with no visual output. Evidence:

- All tests compile and pass `cargo clippy` cleanly
- `./quality.sh` passes with all existing tests (the new stress tests are `#[ignore]` and excluded from normal CI)
- `stress_many_observations_no_panics` verified passing in release mode (122s)
- `stress_1000_neurons_pipeline_completes_without_panic` verified passing in release mode

## Test Plan

- Run `cargo test --release -- --ignored --test-threads=1` to execute all stress tests
- Run `./quality.sh` to confirm no regressions in existing tests
