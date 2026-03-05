## Summary

Fix four American English spelling violations in source comments, update an outdated
module-level doc comment in `analysis/mod.rs`, and improve the `PESSIMISM_CURVE_EXPONENT`
documentation with fully worked examples showing `raw_gain x discount`. Closes #756.

## Changes

### American English fixes
- `src/record/processing.rs:31`: "behavior" → "behaviour"
- `src/record/mod.rs:393`: "behavior" → "behaviour"
- `src/analysis/implementation_tests/synapse_analysis_tests.rs:1265`: "behavior" → "behaviour"
- `src/ffi/analysis.rs:98`: "Analyze" → "Analyse"

### Outdated module comment
- `src/analysis/mod.rs:16`: Updated `neuron.rs` reference to `neuron/` subdirectory
  with its sub-modules (evaluation, post-processing, preparation)

### Constants documentation
- `src/analysis/constants.rs`: Added fully worked examples to `PESSIMISM_CURVE_EXPONENT`
  showing `raw_gain x discount` for each ratio tier

## Evidence

This is a documentation/comments-only change with no behavioural impact. All existing
tests continue to pass. `quality.sh` passes cleanly.

## Test Plan

- No new tests required (comment-only changes)
- All existing tests pass unchanged
- `quality.sh` passes (fmt, clippy, check, test, doc build, release build)
