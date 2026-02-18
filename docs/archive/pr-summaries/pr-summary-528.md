## Summary

Group discovery analysis modules into thematic subdirectories to improve navigability of the `src/analysis/` directory. Closes #528.

28 modules previously at `src/analysis/` root level are now organised into three thematic subdirectories:

- **`detection/`** (18 modules) — Pattern detection modules (saturation, bottleneck, dead neuron, dormant synapse, opposing synapse, oscillating neuron, correlated error, redundant path, bounded range, observation range, sentinel gating, restricted range, operating point, unbounded capping, noise signal, input sensitivity, topology, weight coherence)
- **`recommendation/`** (6 modules) — Candidate recommendation engines (activation recommendation, output bias drift, epistatic, multi-hop, gradient discovery, sample weighted)
- **`scoring/`** (4 modules) — Scoring, confidence, and validation (confidence, weights, error distribution, cross-validation)

All modules are re-exported at the `analysis::` level for full backward compatibility — no public API changes and no changes needed to test or benchmark imports.

## Changes

- Created `src/analysis/detection/mod.rs`, `src/analysis/recommendation/mod.rs`, `src/analysis/scoring/mod.rs`
- Moved 28 module files into their respective subdirectories via `git mv`
- Updated `super::` references in moved files to `crate::analysis::` (since `super` now refers to subdirectory, not `analysis/`)
- Updated `src/analysis/mod.rs` with new subdirectory declarations and backward-compatible re-exports
- Updated `AGENTS.md` source layout documentation to reflect new structure
- Updated `tests/discovery_types_doc_consistency.rs` file paths to match new locations

## Evidence

This is a pure structural refactoring with no behavioural changes. All 489 unit tests and all integration tests pass. `./quality.sh` passes cleanly (fmt, clippy, check, test, release build).

## Test Plan

- All existing tests pass without modification (backward-compatible re-exports maintain `analysis::module_name` paths)
- Updated `tests/discovery_types_doc_consistency.rs` to use new file paths for moved modules
- Verified via `./quality.sh`: fmt, clippy, check, all tests, release build
