## Summary

Replace all `unwrap()` calls in production (non-test) code paths with proper error handling to prevent panics. Closes #940.

Across 9 source files, 25 production `unwrap()` calls were replaced with:
- `let Some(...) else { continue; }` for loop bodies where skipping a sample is the safe fallback
- `filter_map` / `?` operator for iterator chains where None values should be filtered out
- `is_none_or()` for option comparisons (replacing `is_none() || x.unwrap()` pattern)
- `expect()` with documented invariant reason (1 case in `ensemble_scoring.rs` where the index is guaranteed valid)

Test-only `unwrap()` calls were left as-is per the issue requirements.

### Files changed

| File | Unwraps removed | Strategy |
|------|----------------|----------|
| `src/analysis/synapse/scoring.rs` | 7 | `let Some(...) else { continue; }` — skip samples missing target data |
| `src/analysis/recommendation/epistatic/pre_screening.rs` | 6 | `filter_map` + `?` operator for activation-aware residual computation |
| `src/analysis/recommendation/epistatic/candidate_generation.rs` | 3 | `let Some(...) else { continue; }` — skip samples in combined improvement loop |
| `src/analysis/scoring/confidence.rs` | 1 | `let Some(...) else { return Z_SCORE_95; }` — compile-time constant table |
| `src/analysis/ensemble_scoring.rs` | 1 | `expect()` with documented invariant (index computed from same collection) |
| `src/analysis/detection/output_squash_mismatch.rs` | 2 | `filter_map` for pre-activation values; `is_none_or()` for best-candidate comparison |
| `src/analysis/detection/correlated_error.rs` | 3 | `let Some(...) else { continue; }` for HashMap lookups; `let Some(...) else { return; }` |
| `src/analysis/scoring/weights/calculation.rs` | 1 | `let Some(...) else { continue; }` — skip samples missing target_value |
| `src/analysis/gpu/activation_evaluation.rs` | 1 | `let Some(...) else { continue; }` — skip missing GPU buffer |

## Evidence

- `./quality.sh` passes cleanly (fmt, clippy, check, tests, doc, release build)
- All 158 existing tests pass without modification
- New unit tests verify graceful handling of None values where unwrap() was removed

## Test Plan

- Added 4 unit tests in `src/analysis/synapse/scoring.rs`:
  - `test_relu_improvement_skips_samples_with_none_target_value`
  - `test_activation_improvement_skips_samples_with_none_target_activation`
  - `test_synapse_improvement_handles_mixed_none_target_data`
  - `test_relu_improvement_correct_with_complete_data`
- Added 7 integration tests in `tests/issue_940_unwrap_removal.rs`:
  - `test_confidence_metrics_no_panic_empty_samples`
  - `test_confidence_metrics_single_sample_no_panic`
  - `test_ensemble_scoring_single_candidate_no_panic`
  - `test_ensemble_scoring_agreeing_candidates_no_panic`
  - `test_correlated_error_no_panic_with_minimal_data`
  - `test_correlated_error_to_candidates_empty_groups_no_panic`
  - `test_output_squash_mismatch_no_panic_missing_values`
