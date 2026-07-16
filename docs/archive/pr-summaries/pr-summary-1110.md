## Summary

Add a minimum expected-gain floor (`COORDINATED_MIN_EXPECTED_GAIN = 1e-5`) for coordinated structural candidates to filter out noise-level proposals. Closes #1110.

Production failure data from GRQ-sampler (commit 50a2909) shows coordinated structural candidates with `expectedCreatureScoreGain` of ~8e-8 and ~4e-8 that produced actual error changes of -0.0008 and -0.0004 respectively -- harming the network. Expected gains at 1e-8 to 1e-7 are indistinguishable from numerical noise and should never be proposed.

## Changes

- Added `COORDINATED_MIN_EXPECTED_GAIN: f32 = 1e-5` constant in `src/analysis/constants/candidate_scoring.rs`
- Updated both discovery dispatch paths (`run_discovery_module` single-module and `merge_discovery_module_results` parallel) in `src/analysis/discovery_dispatch.rs` to use `>= COORDINATED_MIN_EXPECTED_GAIN` instead of `> 0.0`
- The filter in `src/analysis/neuron/post_processing.rs` was not changed because it handles neuron candidates, not coordinated structural candidates

## Evidence

No UI changes. Backend-only constant addition and filter update. All 32 discovery dispatch tests pass, including 5 new tests for the threshold behaviour.

## Test Plan

Added 5 unit tests in `src/analysis/discovery_dispatch_tests.rs`:
- `coordinated_min_expected_gain_constant_is_1e_minus_5` -- verifies constant value
- `noise_level_gain_8e_minus_8_is_rejected` -- confirms noise-level candidates from production failure data are rejected
- `genuine_gain_1e_minus_4_is_accepted` -- confirms genuine candidates pass through
- `gain_exactly_at_threshold_is_accepted` -- boundary: gain == 1e-5 is accepted
- `gain_just_below_threshold_is_rejected` -- boundary: gain at 9e-6 is rejected
