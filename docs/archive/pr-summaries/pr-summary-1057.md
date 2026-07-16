## Summary

Add-synapse candidates have a near-zero success rate (~0.1-0.3%) across GRQ-sampler creatures, wasting compute on candidates that almost never pass ablation testing. This PR adds two gating mechanisms to skip add-synapse candidate generation when historical data or structural properties indicate near-zero likelihood of success. Closes #1057.

### Changes

- **New `add_synapse_gating` module** (`src/analysis/synapse/add_synapse_gating.rs`): Provides two independent gates:
  1. **Outcome tracker gate**: Skips add-synapse generation when `ModuleOutcomeTracker` shows <1% historical success rate (configurable via `DEFAULT_ADD_SYNAPSE_SUCCESS_THRESHOLD`) for the `"add-synapse"` module, requiring at least `MIN_BOOST_SAMPLES` (10) attempts before activating.
  2. **Synapse density gate**: Skips generation when synapse-to-neuron ratio exceeds 14.0 (configurable via `DEFAULT_SYNAPSE_DENSITY_THRESHOLD`), since adding one synapse to a very dense network has negligible structural impact.

- **Orchestration integration** (`src/analysis/orchestration.rs`): The gate is applied in `analyze_all()` after GPU analysis completes but before post-processing, clearing `helpful_synapses` when either gate triggers while preserving harmful synapse and coordinated structural candidates.

## Evidence

- 19 unit tests in `src/analysis/synapse/add_synapse_gating.rs` validate outcome tracker gating, density gating, combined gating, and candidate filtering
- 17 integration tests in `tests/analysis/issue_1057_add_synapse_gating.rs` validate end-to-end gating behaviour
- All existing tests pass (`./quality.sh` clean)

## Test Plan

- Added `src/analysis/synapse/add_synapse_gating.rs` (19 unit tests):
  - Outcome tracker: insufficient data, empty tracker, low/high success rate, borderline, custom threshold, module isolation
  - Density: sparse/dense networks, threshold boundary, empty creature, custom threshold, input neuron counting
  - Combined: both inactive, outcome-only, density-only triggers
  - Integration: gate clears/preserves helpful synapses

- Added `tests/analysis/issue_1057_add_synapse_gating.rs` (17 integration tests):
  - Outcome gate activation/inactivation scenarios
  - Density gate with boundary conditions
  - Combined gate logic
  - End-to-end candidate filtering
  - Module name stability for cross-run persistence
