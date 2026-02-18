## Summary

Consolidates discovery constants into a central `src/analysis/constants.rs` module,
eliminating duplication across 25+ analysis modules. This is a pure refactoring change
with no functional modifications.

### Constants Consolidated

| Constant | Previous Locations | Value |
|---|---|---|
| `MIN_NEURON_SAMPLE_COUNT` | weights.rs, synapse.rs, implementation.rs, samples.rs, gpu/shaders.rs | `10` |
| `MIN_DISCOVERY_SAMPLE_COUNT` | 20 modules with various names (`MIN_SAMPLES_FOR_*`) | `20` |
| `CANDIDATE_SENTINELS` | observation_range.rs, bounded_range.rs, sentinel_gating.rs | `[-1.0, 0.0, 1.0]` |
| `MIN_SENTINEL_FRACTION` | observation_range.rs, sentinel_gating.rs | `0.15` |
| `SENTINEL_TOLERANCE` | observation_range.rs, sentinel_gating.rs, weights.rs (`DEFAULT_SENTINEL_TOLERANCE`) | `0.02` |
| `MIN_SENTINEL_GAP` | observation_range.rs, bounded_range.rs, sentinel_gating.rs (`MIN_GAP`) | `0.05` |
| `MIN_SOURCE_STD_DEV` | weights.rs, samples.rs (2 locations + reference variant) | `0.05` |
| `DIVERSIFY_TOP_K` | implementation.rs, neuron.rs | `64` |

### Approach

- Created `src/analysis/constants.rs` as the single source of truth
- Each consuming module uses `use super::constants::CONSTANT_NAME` (with aliases where needed to preserve local naming)
- `AGENTS.md` updated with the new file in the source layout
- All existing tests pass — zero functional changes

## Evidence

This is a backend refactoring with no UI changes. Verified by:
- `./quality.sh` passes (fmt, clippy, check, tests, release build)
- All 300+ existing tests pass with `--test-threads=1`
- New integration tests verify constants are accessible and detection modules work correctly

## Test Plan

- Added `tests/issue_424_consolidate_discovery_constants.rs` with 7 tests:
  - `test_min_neuron_sample_count_accessible` — verifies constant value from central module
  - `test_min_discovery_sample_count_accessible` — verifies constant value from central module
  - `test_sentinel_constants_accessible` — verifies all sentinel detection constants
  - `test_min_source_std_dev_accessible` — verifies source variance threshold
  - `test_diversify_top_k_accessible` — verifies diversification parameter
  - `test_saturation_detection_uses_centralised_constants` — functional test with < MIN_NEURON_SAMPLE_COUNT samples
  - `test_dead_neuron_detection_uses_centralised_constants` — functional test with < MIN_DISCOVERY_SAMPLE_COUNT samples
