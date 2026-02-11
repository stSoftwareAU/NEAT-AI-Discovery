## Summary

Adds per-discovery-module success rate tracking and reporting to enable adaptive weighting of discovery modules. Closes #485.

### What Changed

1. **New module `src/analysis/module_weights.rs`** — `ModuleOutcomeTracker` tracks per-module accept/reject outcomes with Bayesian smoothing (Beta(1,1) prior). Provides:
   - Per-module success rate computation
   - Boost factors for deadline-constrained prioritisation (clamped to [0.5, 2.0])
   - Candidate production counts per module
   - Serialisation/deserialisation for cross-run persistence

2. **Per-module stats in discovery dispatch** — `run_discovery_modules_parallel()` now records how many candidates each module produced and populates `discovery_module_stats` in the synapse metadata.

3. **JSON metadata output** — The `synapseMetadata` section of the JSON response now includes a `discoveryModuleStats` array reporting per-module effectiveness:
   ```json
   "discoveryModuleStats": [
     {
       "moduleName": "saturation detection",
       "candidatesProduced": 3,
       "attempts": 50,
       "successes": 18,
       "successRate": 0.36
     }
   ]
   ```

### Design Decisions

- **Bayesian smoothing**: Uses Beta(1,1) prior consistent with existing `CandidateOutcomeCache` (Issue #465), ensuring robust estimates with low sample counts.
- **No module starvation**: Minimum boost is 0.5 (not 0.0), ensuring all modules continue to receive execution budget.
- **No changes to NEAT-AI calling interface**: Module stats are added to existing metadata — the JSON shape extends but doesn't break.
- **Neutral until proven**: Modules with fewer than `MIN_BOOST_SAMPLES` (10) attempts receive neutral boost (1.0).

## Evidence

This is a backend/library change with no UI component. Evidence is provided via tests.

## Test Plan

- Added `tests/issue_485_adaptive_module_weighting.rs` with 14 tests covering:
  - Basic recording and lookup (empty tracker, record/retrieve, independent tracking)
  - Bayesian success rate computation (prior, convergence, never 0 or 1)
  - Boost factor computation (insufficient data, reflects rate, clamped bounds, unknown module)
  - Batch candidate recording
  - All-stats summary
  - Serialisation round-trip
  - Integration test: `discovery_module_stats` populated after `run_discovery_modules_parallel`
- All existing tests continue to pass (verified via `./quality.sh`)
