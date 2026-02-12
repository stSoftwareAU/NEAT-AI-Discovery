## Summary

Pre-screen individual operations before forming coordinated-structural (epistatic/synergistic) pairs. Closes #508.

Production analysis showed all 10 coordinated-structural candidates in one run shared the same harmful operation (neuron e8480883 → output-0, weight 0.1), which degraded the score by ~-0.042 when applied. The partner neuron varied across 9 different inputs but could never overcome the dominant damage.

The fix adds a `MAX_INDIVIDUAL_HARM_FOR_PAIRING` threshold (-0.01) to filter out sources with strongly harmful individual improvement before they can participate in epistatic or synergistic pairing:

- **`detect_epistatic_pairs`**: filters `valid_sources` to exclude sources with `individual_improvement < -0.01`
- **`detect_synergistic_candidates`**: same filter on `valid_sources`, preventing harmful sources from being selected as either primary or complement

Sources that are only mildly negative (>= -0.01) are still allowed — these are the genuine epistatic/synergistic candidates the system is designed to find.

## Evidence

This is a backend logic change with no UI impact. Verified via unit and integration tests.

## Test Plan

- Added integration test `tests/issue_508_prescreen_individual_operations.rs` with 4 tests:
  - `synergistic_candidate_rejects_strongly_harmful_complement` — verifies synergistic path rejects harmful complements
  - `synergistic_candidate_allows_mildly_negative_complement` — verifies mildly negative sources are not over-filtered
  - `epistatic_prescreen_filters_strongly_harmful_sources_early` — verifies epistatic path excludes harmful sources
  - `issue_508_scenario_all_pairs_with_harmful_source_rejected` — reproduces the issue scenario with multiple partners
- Added unit tests in `src/analysis/epistatic.rs`:
  - `test_prescreen_rejects_strongly_harmful_source` — verifies both epistatic and synergistic paths reject harmful sources
  - `test_prescreen_allows_mildly_negative_source` — verifies mildly negative sources pass the pre-screen
- All existing tests continue to pass (no regressions)
- `./quality.sh` passes cleanly
