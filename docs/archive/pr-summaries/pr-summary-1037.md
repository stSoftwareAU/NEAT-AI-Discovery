## Summary

Add range validation for `NEAT_AI_DISCOVERY_MH_TEMPERATURE` (0.01–5.0) and
`NEAT_AI_DISCOVERY_SOURCE_INPUT_INDEX_BIAS` (0–10.0) configuration values,
preventing silent misconfiguration. Out-of-range values now return `None`,
consistent with the existing `.filter()` pattern used by GPU batch size and
outlier percentile. Closes #1037.

## Evidence

- `mh_temperature()` now rejects values outside 0.01–5.0 (matching
  `MIN_TEMPERATURE` / `MAX_TEMPERATURE` from the temperature scheduling module)
- `source_input_index_bias()` now rejects values above 10.0
- Both functions already validated `is_finite()` and `> 0.0`; this adds upper
  bound enforcement

## Test Plan

- Added 17 tests in `tests/infrastructure/issue_1037_config_validation.rs`:
  - MH temperature: accepts lower bound (0.01), upper bound (5.0), mid-range (1.0)
  - MH temperature: rejects zero, below lower bound (0.009), above upper bound (5.1),
    negative, NaN, infinity, empty, non-numeric
  - Source input index bias: rejects above max (10.1), accepts max bound (10.0),
    rejects NaN, rejects zero
  - Constants consistency checks against documented values
- All 171+ existing tests continue to pass
- `./quality.sh` passes cleanly
