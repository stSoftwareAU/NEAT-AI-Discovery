## Summary

Replace the unprincipled `1 / confidence_factor` margin inflation in confidence interval calculation with a statistically sound t-distribution approach. The previous formula divided the margin by `confidence_factor` (product of sample and variance confidence, both ≤ 1.0), which caused the margin to explode at low confidence (e.g., ~196× standard error for 10 samples with low variance).

The new implementation uses `t_critical(df) × standard_error` where `df = sample_count - 1`, with a lookup table covering 20 common degrees of freedom (1–120) and linear interpolation between entries. For df > 120, the normal approximation (z = 1.960) is used. Closes #752.

## Evidence

This is a backend-only change with no visual output. All 19 unit tests pass, including 4 new tests verifying t-distribution behaviour.

## Test Plan

New tests added to `src/analysis/scoring/confidence.rs`:
- `test_t_critical_lookup_known_values` — verifies lookup table accuracy for df=1, df=9, and df=200
- `test_t_distribution_interval_ratio_10_vs_100_samples` — confirms 10-sample interval is 2–5× wider than 100-sample (t-distribution + √n effect)
- `test_nan_samples_do_not_produce_nan_intervals` — ensures NaN-valued samples produce finite intervals
- `test_interval_always_contains_point_estimate` — parametric test across 6 sample counts × 5 expected gains

All 15 existing tests continue to pass unchanged.
