## Summary

Remove all ~80 backward-compatibility `pub use` re-exports from `analysis/mod.rs` and update
all callers to use the correct module paths directly. Closes #676.

The `analysis/mod.rs` file previously re-exported every detection, recommendation, and scoring
sub-module at the `analysis::` level, plus dozens of individual types and functions from
`shared`, `activation`, `samples`, `system`, `utils`, `weights`, `error_distribution`,
`early_termination`, and `cross_validation`. This made the module boundary leaky and tightly
coupled consumers to internal structure.

### Changes

**`src/analysis/mod.rs`** reduced from ~284 lines to 87 lines:
- Removed 36 detection module re-exports
- Removed 6 recommendation module re-exports
- Removed 4 scoring module re-exports
- Removed ~35 individual type/function re-exports (shared, activation, samples, system, utils,
  weights, error_distribution, early_termination, cross_validation)
- Retained only 5 core API entry points: `GpuAnalyzer`, `supports_unified_memory`,
  `analyze_neurons`, `analyze_synapses`, `analyze_synapses_with_cache_and_gpu_queue`,
  and `analyze_all`

**71 files updated** across `src/`, `tests/`, and `benches/` to use direct module paths:
- Detection modules: `analysis::MODULE` → `analysis::detection::MODULE`
- Recommendation modules: `analysis::MODULE` → `analysis::recommendation::MODULE`
- Scoring modules: `analysis::MODULE` → `analysis::scoring::MODULE`
- Shared types: `analysis::TYPE` → `analysis::shared::TYPE`
- Activation items: `analysis::ITEM` → `analysis::activation::ITEM`
- Utils items: `analysis::ITEM` → `analysis::utils::ITEM`
- Error distribution: `analysis::ITEM` → `analysis::scoring::error_distribution::ITEM`

## Evidence

This is a pure refactoring change with no behavioural modifications — only import paths changed.
All 531 unit tests and all integration tests pass. The full quality gate (`./quality.sh`) passes
cleanly including fmt, clippy, check, tests, and release build.

## Test Plan

- No new tests required — this is a refactoring of import paths only
- All existing 531 unit tests pass
- All existing integration tests pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
