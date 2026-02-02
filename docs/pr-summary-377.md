## Summary

Extract shared test utilities module to eliminate duplicated helper patterns across
integration tests. Closes #377.

The `tests/common/mod.rs` module is expanded with reusable builders for neurons,
synapses, creatures, and discovery records. Four test files are refactored to
import these shared helpers instead of redefining identical functions locally.

### Changes

- **`tests/common/mod.rs`** — Added `neuron()`, `hidden()`, `hidden_with_bias()`,
  `output()`, `make_creature()`, and `record()` helpers alongside the existing
  `synapse()`, `synapse_typed()`, `test_data_dir()`, and `skip_without_gpu!` macro.
- **`tests/issue_341_dead_neuron_detection.rs`** — Removed 4 local helpers
  (`record`, `make_creature`, `neuron`, `synapse`); now imports from `common`.
- **`tests/issue_356_dormant_synapse_detection.rs`** — Removed 3 local helpers
  (`make_creature`, `neuron`, `synapse`); now imports from `common`. Retains a
  local `record()` with a different signature (3 args, value derived from activation).
- **`tests/impact_calculation_production.rs`** — Removed 3 local helpers
  (`synapse`, `hidden`, `output`); now imports from `common`.
- **`tests/saturation_detection.rs`** — Removed 3 local helpers
  (`hidden`, `output`, `synapse`); now imports from `common` including
  `hidden_with_bias()` for the non-zero-bias case.

## Evidence

Unable to generate screenshot: This is a CLI-only Rust library with no visual interface.

## Test Plan

- All 4 refactored test files pass with identical behaviour (no test logic changed)
- `./quality.sh` passes cleanly (build, fmt, clippy, check, tests, release build)
- No tests were added, removed, or modified — only the helper function source changed
