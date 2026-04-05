## Summary

Extend candidate compression to support non-linear squash functions (TANH, GELU)
with saturation-aware gain estimation. While IDENTITY compression (Issue #921) uses
simple linear summation, non-linear compression captures interaction effects between
inputs that individual candidates miss. Closes #922.

### Changes

- **`src/analysis/candidate_compression.rs`**: Added `compress_nonlinear_candidates()`,
  `estimate_nonlinear_gain()`, `select_nonlinear_squash()`, and `compress_group_nonlinear()`
  functions. Non-linear gain estimation accounts for TANH/GELU saturation — when combined
  inputs push the squash into saturation, diminished returns are applied. Benefit ratio
  check ensures combined gain exceeds best individual gain by 5% (matching fan-in module).

- **`src/analysis/constants.rs`**: Added `COMPRESSION_SATURATION_THRESHOLD` (0.9) and
  `COMPRESSION_MIN_BENEFIT_RATIO` (1.05) constants.

- **`src/analysis/orchestration.rs`**: Integrated non-linear compression alongside
  IDENTITY compression in the analysis pipeline. Both types coexist in the candidate pool.

## Evidence

All 32 unit tests and 7 integration tests pass. `quality.sh` passes cleanly including
clippy, fmt, doc build, and release build.

## Test Plan

### Unit tests (in `src/analysis/candidate_compression.rs`)
- `test_nonlinear_tanh_compression_linear_regime` — TANH compression succeeds with small weights
- `test_nonlinear_tanh_saturated_inputs_diminished` — saturated TANH inputs produce reduced gain
- `test_nonlinear_gelu_compression` — GELU compression uses target neuron's squash
- `test_nonlinear_benefit_ratio_filtering` — saturated inputs fail benefit ratio check
- `test_nonlinear_selects_target_squash` — target neuron's GELU squash is selected
- `test_nonlinear_defaults_to_tanh` — defaults to TANH when target is IDENTITY
- `test_nonlinear_output_weight_is_conservative` — output weight is 0.1 (conservative)
- `test_nonlinear_comment_mentions_squash` — comment includes squash function name
- `test_nonlinear_empty_input` — empty input produces no candidates

### Integration tests (in `tests/analysis/issue_922_nonlinear_candidate_compression.rs`)
- `test_tanh_compression_linear_regime` — end-to-end TANH compression
- `test_tanh_saturated_inputs_diminished` — saturation-aware gain estimation
- `test_gelu_compression` — GELU squash selection from target neuron
- `test_benefit_ratio_filters_marginal_gains` — heavily saturated inputs rejected
- `test_nonlinear_coexists_with_identity` — both compression types coexist in pipeline
- `test_uses_target_tanh_squash` — target's TANH squash is inherited
- `test_nonlinear_operation_count_discount` — operation discount correctly applied
