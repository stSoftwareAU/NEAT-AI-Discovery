## Summary

Gate add-synapse candidate generation behind historical success rate and synapse
density checks to reduce wasted compute. Closes #1057.

GRQ-sampler discovery cache shows add-synapse candidates have a near-zero success
rate (~0.1-0.3%) across most creatures — only 3 of 43 creatures have any successes.
This change skips add-synapse generation when:

1. **ModuleOutcomeTracker shows <1% historical success rate** (configurable via
   `ADD_SYNAPSE_MIN_SUCCESS_RATE`): When sufficient history exists (>=10 attempts)
   and the Bayesian success rate falls below the threshold, helpful add-synapse
   candidates are cleared before post-processing.

2. **Synapse density ratio exceeds threshold** (configurable via
   `ADD_SYNAPSE_MAX_DENSITY_RATIO`): When synapse_count/neuron_count > 14,
   adding one synapse has minimal structural impact and generation is skipped.

Both thresholds are configurable via override parameters and exposed as public
constants for external use.

### Changes

- **New module**: `src/analysis/synapse/add_synapse_gating.rs` — contains
  `should_skip_add_synapse_candidates()` and `compute_synapse_density_ratio()`
- **New constants**: `ADD_SYNAPSE_MODULE_NAME`, `ADD_SYNAPSE_MIN_SUCCESS_RATE`,
  `ADD_SYNAPSE_MAX_DENSITY_RATIO` in `src/analysis/constants/candidate_scoring.rs`
- **Extended `AnalyzeSynapsesInput`** with optional `module_outcome_tracker` field,
  passed from `analyze_all` orchestration
- **Gating wired into `results.rs`**: Checks run before post-processing; when
  triggered, helpful candidates are cleared with diagnostic logging

## Evidence

All 13 new tests pass, covering:
- Density ratio computation (basic, zero neurons, no synapses)
- Success rate gating (low rate, acceptable rate, no data, insufficient data, no tracker)
- Density gating (high density, below threshold)
- Custom threshold overrides
- Constant range validation

## Test Plan

- Added `tests/analysis/issue_1057_add_synapse_gating.rs` with 13 unit tests
- All 528+ existing tests continue to pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
