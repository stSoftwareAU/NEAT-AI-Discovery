## Summary

Fix combo-successful module's 0% success rate caused by false complementarity detection in epistatic pair analysis. Closes #731.

Four root causes addressed:

1. **Strict pre-screen** (`MAX_INDIVIDUAL_HARM_FOR_PAIRING`: -0.01 → 0.0): Harmful individual sources no longer participate in pairing. Production data showed even mildly negative sources (-0.005) consistently caused combo failures.

2. **Sample-wise harm check**: New `has_cross_sample_harm()` function verifies that source A does not hurt samples where source B fires, and vice versa. This replaces the naive complementarity-only metric that assumed non-overlapping firing patterns imply synergy.

3. **Super-additivity requirement**: Combined improvement must now exceed the **sum** of individual improvements (not just the max). This filters out merely additive pairs that provide no genuine epistatic benefit.

4. **Cross-validation**: `cross_validate_pair()` splits samples in half and verifies the combo benefit holds on both halves, preventing overfitting to observation patterns.

5. **Sample-level combined improvement**: `compute_combined_improvement_from_samples()` replaces the old naive estimator that just summed pre-computed individual improvements. The new version computes actual error reduction from sample-level data.

## Evidence

This is a backend logic change with no UI component. Verified by:
- 6 new unit tests covering all four improvements
- All 557 existing library tests pass
- All 15 existing integration tests for epistatic/combo/pre-screen modules pass
- `quality.sh` passes cleanly

## Test Plan

New tests in `tests/issue_731_combo_successful_filtering.rs`:
- `prescreen_rejects_mildly_harmful_source` — verifies sources with improvement < 0 are excluded
- `prescreen_allows_zero_improvement_source` — verifies zero-improvement sources pass the pre-screen
- `rejects_pair_where_source_hurts_others_samples` — verifies cross-harm detection
- `rejects_pair_without_super_additivity` — verifies super-additivity requirement
- `cross_validation_rejects_overfit_combo` — verifies cross-validation filtering
- `genuine_epistatic_pair_survives_strict_filters` — regression test for true epistatic pairs

Modified test in `src/analysis/recommendation/epistatic/scoring.rs`:
- `test_prescreen_allows_mildly_negative_source` → renamed to `test_prescreen_rejects_mildly_negative_source` (business logic change: threshold tightened from -0.01 to 0.0)
