## Summary

Organise 217 integration test files from a flat `tests/` directory into 15 feature-based
subdirectories that mirror the source module structure. Closes #817.

The flat directory made it difficult to navigate, find related tests, or run tests by
feature area. Tests are now grouped into meaningful categories:

| Directory | Files | Responsibility |
|-----------|-------|----------------|
| `detection/` | 40 | Pattern detection (neuron, synapse, structural) |
| `analysis/` | 43 | General analysis pipeline (orchestration, dispatch) |
| `synapse/` | 23 | Synapse analysis pipeline |
| `recommendation/` | 16 | Candidate recommendation |
| `scoring/` | 15 | Scoring, confidence, statistics |
| `infrastructure/` | 14 | Caching, config, observability |
| `activation/` | 13 | Activation function tests |
| `neuron/` | 11 | Neuron analysis |
| `focus/` | 10 | Focus and ranking system |
| `recording/` | 8 | Recording, parquet, history |
| `gpu/` | 6 | GPU functionality |
| `proptest/` | 6 | Property-based tests |
| `regression/` | 5 | Versioned regression tests |
| `export/` | 4 | Visualisation and export |
| `ffi/` | 2 | FFI boundary tests |

Files that remain at the top level:
- `integration.rs` — top-level integration entry point
- `gpu_timing.rs` — requires its own process due to cached GPU env var state
- `common/mod.rs` — shared test helpers (accessible from all subdirectories)
- `unit/` — existing unit test wrappers

Each subdirectory has a `main.rs` entry point that imports `common/mod.rs` via
`#[path = "../common/mod.rs"]` and declares all submodules.

## Evidence

- `quality.sh` passes cleanly (fmt, clippy, check, tests, docs, release build)
- All 2625 tests pass across 18 test binaries with `--test-threads=2`
- No test files were removed or modified beyond import path updates

## Test Plan

- All existing tests continue to pass from their new locations
- `cargo test --lib --tests --all-features -- --test-threads=2` — 0 failures
- `gpu_timing.rs` kept as standalone test to avoid GPU initialisation order issues
- Each subdirectory compiles as its own test crate
