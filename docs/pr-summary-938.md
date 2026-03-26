## Summary

Split `src/analysis/constants.rs` (874 lines) into a `constants/` directory with six thematic sub-modules, improving maintainability while preserving backward compatibility. All constants remain accessible via the same `crate::analysis::constants::CONSTANT_NAME` paths through re-exports in `mod.rs`. Closes #938.

## Sub-module Organisation

| Sub-module | Category |
|------------|----------|
| `sample_thresholds.rs` | Sample count thresholds, hold-out validation |
| `sentinel_detection.rs` | Sentinel values and clustering thresholds |
| `source_variance.rs` | Source variance filtering thresholds |
| `candidate_scoring.rs` | Scoring boosts, pessimism discounts, calibration, NaN-safe comparisons |
| `compression.rs` | Candidate compression thresholds |
| `detection_thresholds.rs` | Detection filtering (removal, weight constraints, improved ratios) |

## Evidence
- No changes to any consuming source files — purely internal reorganisation
- `./quality.sh` passes with no warnings
- All 158 library tests + 14 new tests pass
- All existing tests (including `issue_424_consolidate_discovery_constants`) continue to pass unchanged

## Test Plan
- Added `tests/analysis/issue_938_constants_submodule_organisation.rs` with 14 tests covering:
  - All constants accessible via re-exports from each sub-module
  - `activation_neuron_boost()` function accessible and returns correct values
  - NaN-safe comparison functions accessible and correct
  - Every constant value matches expected defaults
