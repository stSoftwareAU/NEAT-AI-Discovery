## Summary

Apply a pessimism discount to expected score gain predictions to address the 18,500x over-estimation observed in production (creature b2ff6e45, production discovery-cache commit a1340f8d). Closes #506.

The root cause: the improvement calculation measures the fraction of a single target neuron's squared error explained by sampled data (`(baseline_error_sq - new_error_sq) / baseline_error_sq`), but this neuron-level relative improvement was used directly as the creature-level expected score gain. In practice, sample-level improvements do not fully generalise to the full training set, causing predictions to be orders of magnitude too optimistic.

The fix applies a pessimism discount based on the `improved_count / total_count` ratio:

```
discount = PESSIMISM_DISCOUNT_FLOOR + (1 - PESSIMISM_DISCOUNT_FLOOR) x (improved_count / total_count)
discounted_gain = raw_gain x discount
```

- When all samples improve (ratio = 1.0), no discount is applied
- When fewer samples improve, the prediction is progressively reduced
- The floor (0.15) ensures candidates are never completely zeroed out

The discount is applied uniformly to:
- Helpful synapse candidates (post-processing)
- Harmful synapse candidates (post-processing)
- Neuron candidates (post-processing)

## Evidence

This is a backend scoring change with no UI component. Correctness is verified by the test suite:

- 8 unit/integration tests in `tests/issue_506_score_prediction_pessimism_discount.rs`
- All 472 existing unit tests continue to pass
- All 97 integration test files continue to pass
- Full `quality.sh` gate passes (fmt, clippy, check, test, release build)

## Test Plan

- `test_pessimism_discount_partial_improvement` — 60% improved ratio reduces prediction to ~0.033
- `test_pessimism_discount_all_improved` — 100% improved ratio preserves full prediction (no discount)
- `test_pessimism_discount_few_improved` — higher ratio produces larger gain than lower ratio
- `test_pessimism_discount_none_improved` — 0% improved applies floor discount only (~0.0075)
- `test_pessimism_discount_zero_total` — degenerate case handled gracefully
- `test_pessimism_discount_negative_gain` — negative gains preserve sign after discount
- `test_pessimism_discount_reduces_weak_signal_significantly` — weak signal (30% improved) reduces by >50%
- `test_issue_506_synapse_candidates_have_pessimism_discount` — GPU integration test verifying discount is applied in full synapse analysis pipeline
