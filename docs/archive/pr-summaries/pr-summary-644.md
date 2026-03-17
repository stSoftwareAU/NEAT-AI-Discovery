## Summary

Add a weight polarity flip detection module that identifies synapses where
the gradient sign consistently agrees with the weight sign — meaning gradient
descent must push the weight through zero to reach the optimal value. Instead
of many small delta steps, a direct `setWeight` candidate with the negated
weight skips the traversal entirely. Closes #644.

### Changes

- **New module**: `src/analysis/detection/weight_polarity_flip.rs` — detects
  gradient-weight sign agreement and proposes negated weight candidates
- **Dispatch wiring**: Added to `synapse_specs.rs` dispatch pipeline
- **Module registration**: Added to `detection/mod.rs` and `analysis/mod.rs`
  re-exports
- **Scenario doc**: `docs/discoveries/weight-polarity-flip.md`

### Detection criteria

1. Weight and gradient have the same sign (descent crosses zero)
2. High gradient consistency (|mean|/std > 0.5)
3. Significant weight magnitude (|weight| > 0.1)
4. Gradient large relative to weight (ratio > 0.05)

## Evidence

This is a backend detection module with no UI. Evidence is provided via tests:

```
running 10 tests
test test_candidate_produces_negated_weight ... ok
test test_candidates_sorted_by_improvement ... ok
test test_detects_negative_weight_negative_gradient ... ok
test test_detects_positive_weight_positive_gradient ... ok
test test_empty_records_no_candidates ... ok
test test_flip_is_distinct_from_gradient_delta ... ok
test test_inconsistent_gradient_not_flagged ... ok
test test_insufficient_samples_no_candidates ... ok
test test_near_zero_weight_not_flagged ... ok
test test_opposite_sign_weight_gradient_not_flagged ... ok

test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured
```

## Test Plan

- `tests/issue_644_weight_polarity_flip.rs` — 10 tests covering:
  - Detection of positive weight + positive gradient polarity mismatch
  - Detection of negative weight + negative gradient polarity mismatch
  - Rejection of opposite-sign weight-gradient pairs (no flip needed)
  - Rejection of inconsistent/noisy gradients
  - Candidate produces `setWeight` with negated weight
  - Rejection of near-zero weights
  - Edge cases: empty records, insufficient samples
  - Candidates distinct from gradient discovery small-delta proposals
  - Candidates sorted by estimated improvement
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
