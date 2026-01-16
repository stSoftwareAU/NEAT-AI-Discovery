## Summary

This PR extracts synapse evaluation functions from `implementation.rs` (~2000 lines) to the dedicated `src/analysis/synapse.rs` module as part of the ongoing refactoring effort (parent issue #185).

### Key Changes

**src/analysis/synapse.rs** (expanded from 26 lines to ~1990 lines):
- Sample locality grouping functions (Issue #221 optimisation):
  - `SampleLocalityGroup` struct for grouping sources with similar obs_indices
  - `extract_obs_indices()`, `compute_obs_index_overlap()`, `group_sources_by_locality()`
  - `build_samples_for_locality_group()` - efficient batched sample building
- Target simulation mode types and functions:
  - `TargetSimulationMode` enum (None, Full, ApproximateValueFromActivation)
  - `get_target_activation_fn()`, `get_target_simulation_fn()`, `get_target_simulation_mode()`
- Synapse improvement calculation functions:
  - `weight_sign()`, `upsert_candidate()`, `SplitReluResult` struct
  - `has_sufficient_output_variance()` - saturation detection (Issue #123)
  - `compute_relu_improvement_and_count()`, `compute_activation_improvement_and_count()`
  - `compute_synapse_improvement_and_count()` - main improvement calculation
- ReLU and activation candidate evaluation:
  - `evaluate_relu_candidates_split()` - split-error ReLU evaluation
  - `evaluate_activation_for_subset()`, `evaluate_activation_candidate()`
- Coordinated structural helpers and public API entry points

**src/analysis/samples.rs**:
- Moved `ReluStats::evaluate()` method from implementation.rs

**src/analysis/implementation.rs**:
- Renamed `analyze_synapses_with_cache` to `analyze_synapses_with_cache_impl`
- Updated test imports to use moved functions
- Removed duplicate functions (now in synapse.rs)

**src/analysis/mod.rs**:
- Updated `analyze_all` to call `synapse::analyze_synapses_with_cache`

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

This is a refactoring change with no functional modifications. The evidence is:
- All 343+ existing tests pass
- `./quality.sh` passes completely
- No public API changes

## Test Plan

- All existing synapse analysis tests continue to pass
- New unit tests added in `synapse.rs`:
  - `test_weight_sign` - verifies weight sign calculation
  - `test_compute_obs_index_overlap` - verifies obs_index overlap calculation
  - `test_target_simulation_mode_none_without_data` - verifies mode selection without target data
  - `test_target_simulation_mode_full_with_data` - verifies full simulation mode
  - `test_target_simulation_mode_approximate` - verifies approximate mode selection
- Ran `./quality.sh` which includes:
  - Debug build
  - Clippy linting
  - All unit tests (343 passed)
  - All integration tests
  - Release build

## Notes

- Some functions are extracted to synapse.rs but still have copies in implementation.rs. This is intentional - implementation.rs needs these for neuron analysis until that is also extracted (Issue #277).
- The sample locality optimisation (Issue #221) is preserved - sources with ≥80% overlap are grouped to reduce sample building overhead by up to 100x.
