## Summary

Add per-scale success rate tracking for weight variant selection. The new
`ScaleOutcomeTracker` records outcomes keyed by `(module_name, weight_scale_tier)`
using the same Bayesian Beta(1,1) prior as the existing `ModuleOutcomeTracker`.
Per-scale boost factors (clamped to [0.5, 2.0]) can be applied to variant
`expected_multiplier` values, biasing future generation toward historically
successful scales. Includes decay factor to prevent stale data dominance and
JSON serialisation for persistence. Closes #964.

## Evidence

All 22 new unit tests pass, covering:
- Per-(module, scale) outcome recording and independent tracking
- Bayesian success rate computation with Beta(1,1) prior
- Per-scale boost factor computation with MIN_BOOST_SAMPLES threshold
- Boost clamping to [0.5, 2.0] range
- Decay factor with rounding safety (successes never exceed attempts)
- JSON serialisation round-trip
- Scale tier extraction from variant candidate comments
- `apply_scale_boosts_to_candidates` integration with coordinated structural candidates

## Test Plan

- Added `tests/analysis/issue_964_per_scale_success_rate.rs` (22 tests)
- `quality.sh` passes cleanly
