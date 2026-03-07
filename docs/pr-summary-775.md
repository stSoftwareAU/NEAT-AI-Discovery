## Summary

Promote `spearman_rank_correlation` and `compute_ranks` from private functions in `monotonicity.rs` to public shared functions in the `stats` module (`src/analysis/detection/stats.rs`). This follows the same pattern as Pearson correlation (Issue #767) and makes Spearman rank correlation available for future modules. Closes #775.

## Changes

- **`src/analysis/detection/stats.rs`**: Added `compute_ranks()` and `spearman_rank_correlation()` as public functions with documentation.
- **`src/analysis/detection/monotonicity.rs`**: Removed local copies of `spearman_rank_correlation` and `compute_ranks`, replaced with import from `super::stats::spearman_rank_correlation`.
- **`tests/issue_775_stats_spearman_correlation.rs`**: 14 integration tests covering edge cases.

## Evidence

This is a backend refactor with no UI changes. All tests pass and `quality.sh` passes cleanly.

## Test Plan

- `compute_ranks_empty_input` — empty slice returns empty ranks
- `compute_ranks_single_element` — single element gets rank 1.0
- `compute_ranks_sorted_values` — already sorted values get sequential ranks
- `compute_ranks_reverse_sorted` — reverse sorted values get descending ranks
- `compute_ranks_with_ties` — tied values receive average rank
- `compute_ranks_all_tied` — all equal values receive same average rank
- `spearman_empty_input_returns_zero` — empty input returns 0.0
- `spearman_single_element_returns_zero` — single element returns 0.0
- `spearman_perfect_positive` — perfectly monotonic increasing returns ~1.0
- `spearman_perfect_negative` — perfectly monotonic decreasing returns ~-1.0
- `spearman_monotonic_nonlinear` — nonlinear monotonic (x³) returns ~1.0
- `spearman_with_ties` — tied values still produce correct correlation
- `spearman_zero_variance_returns_zero` — constant input returns 0.0
- `spearman_no_monotonic_relationship` — U-shaped data returns near-zero
