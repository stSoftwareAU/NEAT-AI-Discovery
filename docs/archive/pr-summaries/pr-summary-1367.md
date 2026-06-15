## Summary

Candidate ranking sorts coordinated-structural candidates by
`expected_creature_score_gain` using `total_cmp`. While panic-safe, `total_cmp`
orders a positive `NaN` **above** `+∞`, so a candidate with a `NaN` (or `±∞`)
gain would sort to the **top** and be returned as the *best* candidate. There
was no finiteness guard anywhere in `candidate_aggregation.rs` or
`candidate_diversity.rs`. A non-finite gain is not a valid positive expected
improvement, and returning one wastes the controller's ablation-test budget on a
meaningless candidate — exactly the cost the mission exists to avoid.

This PR adds a finiteness filter that drops candidates whose
`expected_creature_score_gain` is not finite (`!x.is_finite()`) **before**
diversity reranking / final selection, and records the drop in the existing
rejection breakdown so it stays observable.

Changes:
- New `reject_non_finite_gains()` in `candidate_aggregation.rs` — retains only
  finite-gain candidates, preserving the relative order of survivors, and
  returns the dropped count.
- Wired into `merge_coordinated_structural_replacements` (before the sort /
  diversity rerank) and into `apply_final_coordinated_gain_floor` (the
  unconditional final selection gate before the FFI response). The latter also
  closes the `+∞ >= floor` gap, where an infinite gain would otherwise survive
  the existing noise-floor comparison.
- New stable rejection reason `non_finite_gain` (`REJECTION_NON_FINITE_GAIN`),
  added to `ALL_REJECTION_REASONS` and the friendly-summary mapping.

Closes #1367.

## Evidence

Backend/library change only — no web interface to screenshot. Verified via the
new TDD tests and the full `./quality.sh` gate (fmt, clippy `-D warnings`,
check, all tests, release build) passing cleanly.

```mermaid
flowchart LR
    A[Coordinated candidates] --> B{is_finite?}
    B -- "NaN / ±∞" --> R[Dropped\nrecord non_finite_gain]
    B -- finite --> C[Diversity rerank / final gain floor]
    C --> D[Returned candidates]
```

## Test Plan

Added `tests/issue_1367_non_finite_gain_rejection.rs`:
- `reject_non_finite_gains_drops_non_finite_and_preserves_order` — feeds a set
  containing `NaN`, `+∞` and `-∞` plus several finite candidates; asserts the
  three non-finite entries are dropped and the finite survivors keep their
  original order.
- `final_gate_rejects_non_finite_and_records_breakdown` — feeds a `NaN`-gain and
  a `+∞`-gain candidate plus finite candidates through
  `apply_final_coordinated_gain_floor`; asserts neither non-finite candidate is
  returned, all finite candidates survive, and the rejection breakdown records
  `non_finite_gain == 2`.

Full suite: `./quality.sh` passes (all unit + integration tests green).
