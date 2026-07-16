# PR Summary: Enhance ModuleOutcomeTracker to learn from pre-filtering failures (#1060)

Closes #1060

## Summary

Enhances the `ModuleOutcomeTracker` to learn from pre-filtering failures
(budget truncation and non-positive gain filtering) by recording them as
soft failures with configurable weight. Modules whose Bayesian success rate
drops below a configurable threshold are gated (skipped) during detection,
saving compute on consistently failing modules.

## Changes

### Core logic (`src/analysis/module_weights.rs`)
- Added `soft_failures: f64` field to `ModuleStats` with `#[serde(default)]`
  for backwards-compatible deserialisation.
- Updated `success_rate()` to incorporate soft failures into the Bayesian
  Beta(1,1) posterior: `beta = real_failures + soft_failures + 1.0`.
- Added `record_soft_failures()` method with weight clamped to [0, 1].
- Added `is_gated()` and `is_gated_default()` methods requiring
  `MIN_BOOST_SAMPLES` attempts before gating activates.
- Decay now also applies to `soft_failures`.
- Added `soft_failures` and `gated` fields to `DiscoveryModuleStatsJson`.

### Constants (`src/analysis/constants/candidate_scoring.rs`)
- `MODULE_GATE_THRESHOLD = 0.005` — success rate below which modules are skipped.
- `SOFT_FAILURE_WEIGHT = 0.5` — weight per filtered candidate (half a real failure).

### Discovery dispatch (`src/analysis/discovery_dispatch.rs`)
- `detect_discovery_modules_parallel` accepts optional tracker for gating.
- Detection phase skips modules gated by low success rate.
- Merge phase records soft failures for budget-truncated and negative-gain
  filtered candidates.
- Per-module stats now include `soft_failures` and `gated` in metadata.

### Tests (`tests/analysis/issue_1060_prefilter_learning.rs`)
- 24 unit tests covering soft failure recording, success rate integration,
  module gating, decay, dispatch gating, merge recording, and metadata exposure.

## Test plan

- [x] All 24 new tests pass
- [x] All existing tests pass (no regressions)
- [x] `quality.sh` passes (clippy, tests, docs, release build)
