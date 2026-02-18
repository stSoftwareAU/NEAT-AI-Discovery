## Summary

Complete the `implementation.rs` refactoring (Issue #185) by moving the sole remaining function `analyze_synapses_with_cache_impl` (~1,920 lines) into `synapse.rs` and deleting `implementation.rs` entirely.

### Changes

- **Deleted `src/analysis/implementation.rs`** (2,123 lines) — the legacy monolith is gone
- **Moved `analyze_synapses_with_cache_impl`** into `src/analysis/synapse.rs` where it belongs alongside the public API functions that call it
- **Removed `mod implementation;`** from `src/analysis/mod.rs`
- **Removed file-level `#![allow(dead_code)]`** from `synapse.rs` — no longer needed since the migration is complete
- **Updated `implementation_tests`** to be owned by `synapse.rs` via `#[path = ...]` attribute
- **Updated `AGENTS.md`** source layout to reflect the removal

### What this achieves

- `implementation.rs` is reduced from 2,123 lines to **0 lines** (deleted)
- No duplicate function definitions remain
- The `#![allow(dead_code)]` annotation in `synapse.rs` is removed
- The delegation chain (`synapse.rs` → `implementation.rs`) is eliminated — `synapse.rs` now owns its full pipeline

### No functional changes

This is a purely structural refactoring. The function body was moved verbatim. All existing tests pass without modification.

## Evidence

This is a pure refactoring with no UI or performance changes. Evidence is provided by the test suite:

- All 10 implementation test modules pass (cache, GPU batch, diagnostics, improvement model, optimal weight, prediction accuracy, ReLU evaluation, sample matching, synapse analysis, bias calculation)
- All ~97 integration tests pass
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- No new tests added — this is a refactoring that preserves all existing behaviour
- All existing tests continue to pass unchanged, confirming no functional regression
- The `implementation_tests/` directory is now referenced from `synapse.rs` instead of the deleted `implementation.rs`
