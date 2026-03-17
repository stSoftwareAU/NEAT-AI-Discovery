## Summary

Consolidate duplicated activation function classification helpers (`is_bounded_squash`, `is_saturating_squash`, `can_have_dead_zone`) into a shared `activation_properties` module. Closes #768.

Previously, `saturation.rs` and `weight_coherence.rs` each had their own private copies of these helpers. This refactoring creates a single source of truth in `src/analysis/detection/activation_properties.rs` and replaces all private implementations with imports from the shared module.

## Changes

- **New**: `src/analysis/detection/activation_properties.rs` — shared module with three well-documented public helpers
- **Modified**: `src/analysis/detection/saturation.rs` — removed private `is_bounded_squash` and `can_have_dead_zone`, replaced with shared imports
- **Modified**: `src/analysis/detection/weight_coherence.rs` — removed private `is_saturating_squash`, replaced with shared import
- **Modified**: `src/analysis/detection/mod.rs` — registered new `activation_properties` module
- **New**: `tests/issue_768_activation_properties.rs` — 9 integration tests covering all helpers

## Evidence

This is a backend refactoring with no UI changes. All existing tests continue to pass, and 9 new integration tests verify the shared helpers. `quality.sh` passes cleanly.

## Test Plan

- `tests/issue_768_activation_properties.rs` — 9 tests covering:
  - `is_bounded_squash` recognises all bounded and rejects unbounded functions
  - `is_saturating_squash` recognises saturating functions, is case-insensitive, rejects non-saturating
  - `can_have_dead_zone` recognises ReLU family, rejects non-ReLU functions
  - Cross-classification: all saturating functions are also bounded; dead-zone functions are not bounded
- Existing `weight_coherence::tests::test_is_saturating_squash` continues to pass via the shared import
