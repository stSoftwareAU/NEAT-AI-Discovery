## Summary

Comprehensive improvements to the discovery module addressing Issue #481. This PR implements three concrete improvements identified during codebase analysis:

1. **NaN-safe floating-point sorting across all modules** (Issue #483) — Replaced 78 instances of the unsafe `partial_cmp().unwrap_or(Ordering::Equal)` pattern with `total_cmp()` across 33 source files. Added `cmp_f32_desc`, `cmp_f32_asc`, and `cmp_f64_desc` helpers to `constants.rs` as the canonical NaN-safe sort comparators.

2. **Record-loading helper to eliminate boilerplate** (Issue #493) — Added `load_records_for_uuids()` and `load_records_for_hidden()` methods to `RecordCache`, then replaced 25 identical record-loading blocks in `analyze_all()`. Net reduction of **308 lines** (174 added, 482 removed).

3. **Created GitHub issue #493** for the record-loading DRY violation, documenting the problem and solution for future reference.

### Issues Created

| # | Title | Category |
|---|-------|----------|
| [#493](https://github.com/stSoftwareAU/NEAT-AI-Discovery/issues/493) | Extract record-loading helper to eliminate 25x boilerplate in mod.rs | DRY / Code organisation |

### Design Constraints

- NEAT-AI does not need to change — all improvements are internal to the discovery library
- All existing candidate types are reused unchanged
- All 463+ existing tests continue to pass without modification

## Evidence

This is a backend/CLI change with no visual UI. Evidence is provided via:
- 12 new integration tests across 2 test files that exercise real library functions
- All 463 existing unit tests continue to pass
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)
- 78 NaN-unsafe sort patterns eliminated across 33 files
- 25 duplicated record-loading blocks replaced with 2 helper methods

## Test Plan

- `tests/issue_481_nan_safe_sorting.rs` — 8 new tests:
  - `test_cmp_f32_desc_basic_ordering` — Descending sort correctness
  - `test_cmp_f32_asc_basic_ordering` — Ascending sort correctness
  - `test_cmp_f32_desc_nan_deterministic` — NaN values sort deterministically (descending)
  - `test_cmp_f32_asc_nan_deterministic` — NaN values sort deterministically (ascending)
  - `test_cmp_f32_handles_negative_zero` — Negative zero edge case
  - `test_cmp_f32_handles_infinity` — Infinity edge case
  - `test_cmp_f32_all_nan_no_panic` — All-NaN slice doesn't panic
  - `test_cmp_f32_empty_slice_no_panic` — Empty slice doesn't panic

- `tests/issue_481_record_loading_helper.rs` — 4 new tests:
  - `test_load_records_returns_all_requested_uuids` — Returns records for requested UUIDs
  - `test_load_records_skips_missing_uuids` — Missing UUIDs silently skipped
  - `test_load_records_empty_uuids` — Empty UUID list returns empty
  - `test_load_records_preserves_content` — Record content preserved exactly

- `tests/issue_481_suggest_improvements.rs` — 9 existing tests (unchanged):
  - `test_bias_values_are_sorted_for_all_activations`
  - `test_bias_values_span_negative_and_positive`
  - `test_boost_constants_within_valid_ranges`
  - `test_sentinel_constants_ordering_invariants`
  - `test_saturation_detection_produces_valid_candidates`
  - `test_dead_neuron_detection_produces_valid_candidates`
  - `test_dormant_synapse_detection_produces_valid_candidates`
  - `test_oscillating_detection_handles_constant_activation`
  - `test_diversify_top_k_is_practical`
