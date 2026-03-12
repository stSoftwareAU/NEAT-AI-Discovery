## Summary

Wire up adaptive module weighting from `ModuleOutcomeTracker` to candidate scoring. Closes #792.

Previously, the `ModuleOutcomeTracker` infrastructure (Issue #485) was fully implemented but not
connected end-to-end. The ensemble scoring used an empty default tracker, and per-module stats in
the response metadata were hardcoded to zeroes. This change completes the wiring:

- **Input/output**: Added `moduleOutcomeTracker` field to `AnalyzeParallelInput`/`AnalyzeAllInput`
  (input) and `AnalyzeParallelOutput` (output) for persistence across discovery runs.
- **Pipeline threading**: The tracker flows from input through `analyze_all` →
  `dispatch_and_merge_discovery_modules` → `run_discovery_modules_parallel` and
  `apply_ensemble_scoring`.
- **Historical stats in metadata**: `run_discovery_modules_parallel` now populates per-module
  `attempts`, `successes`, and `success_rate` from the tracker instead of hardcoded zeroes.
- **Module boost applied to candidates**: `apply_module_boost_to_candidates()` multiplies each
  coordinated structural candidate's `expected_creature_score_gain` by its module's boost factor
  (0.5–2.0), so modules with poor track records produce lower-ranked candidates.
- **Ensemble scoring uses real tracker**: `apply_ensemble_scoring` now receives the actual tracker
  instead of `ModuleOutcomeTracker::default()`.

## Evidence

All 6 new tests pass, verifying:
- Tracker stats populate metadata in parallel dispatch with historical data
- Module boost is applied to coordinated candidate gains (high-success > low-success)
- Empty tracker produces neutral boost (no change to gains)
- Tracker serialisation round-trip works for input JSON
- Ensemble scoring uses provided tracker instead of default
- Module stats success rate is correct for known data

All 14 existing adaptive module weighting tests (Issue #485) continue to pass.
All existing parallel dispatch tests continue to pass.
`quality.sh` passes cleanly.

## Test Plan

- Added `tests/issue_792_adaptive_module_wiring.rs` with 6 tests:
  - `tracker_stats_populate_metadata_in_parallel_dispatch`
  - `module_boost_applied_to_coordinated_candidates`
  - `empty_tracker_produces_neutral_boost`
  - `tracker_serialises_in_input_json`
  - `ensemble_scoring_uses_real_tracker`
  - `module_stats_success_rate_for_known_data`
- Updated existing test `discovery_module_stats_populated_after_parallel_dispatch` in
  `tests/issue_485_adaptive_module_weighting.rs` to pass the new tracker parameter
- Updated existing test `analyze_parallel_output_contains_all_candidate_fields` in
  `tests/issue_337_candidate_type_contract.rs` to include the new field
