## Summary

Plan and initial implementation for discovering bounded ranges of observations and hidden neurons (Issue #395).

### Planning Phase

Created 5 sub-issues breaking down the bounded range discovery feature:

1. **#398 — Detect observation effective ranges** — Foundation module to identify effective ranges and sentinel values per observation
2. **#399 — Bounded range neuron detection** — Detect hidden neurons with restricted activation ranges (implemented in this PR)
3. **#400 — Sentinel value gating** — Propose gated connections using scalar (GPU-compatible) squash functions to ignore sentinel values
4. **#401 — Hidden neuron operating-point analysis** — Detect neurons whose pre-activation distribution is poorly placed for their squash function
5. **#402 — Range-aware weight optimisation** — Enhance weight computation to exclude sentinel samples

### Implementation Phase

Implemented the bounded range neuron detection module (`#399`) as the first concrete deliverable:

- New `src/analysis/bounded_range.rs` module that detects hidden neurons operating in a restricted sub-range of their activation function's output domain
- Integrated into the analysis pipeline via the DRY `run_discovery_module` dispatch pattern (Issue #375)
- Proposes `ChangeSquash` and `SetBias` coordinated structural candidates
- Only analyses hidden neurons with bounded activation functions (TANH, LOGISTIC, HARD_TANH, etc.)
- Excludes dead neurons, input/output/constant neurons, and neurons with insufficient samples

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

Added 15 integration tests in `tests/issue_395_bounded_range_detection.rs`:

- `test_empty_records_returns_empty` — no records produces no candidates
- `test_single_neuron_narrow_range_detected` — TANH neuron using 10% of range is detected
- `test_neuron_using_full_range_not_detected` — 90% utilisation is not flagged
- `test_unbounded_squash_not_flagged` — IDENTITY (unbounded) neurons excluded
- `test_dead_neuron_excluded` — near-zero activation excluded (dead neuron territory)
- `test_input_neurons_excluded` — input neurons not analysed
- `test_output_neurons_excluded` — output neurons not analysed
- `test_constant_neurons_excluded` — constant neurons not analysed
- `test_insufficient_samples_skipped` — fewer than 20 samples skipped
- `test_utilisation_ratio_computed_correctly` — verifies range/theoretical calculation
- `test_candidates_sorted_by_improvement` — candidates ordered best-first
- `test_coordinated_candidates_contain_change_squash` — output has ChangeSquash/SetBias ops
- `test_multiple_neurons_analysed` — two narrow-range neurons produce two candidates
- `test_logistic_narrow_range_detected` — LOGISTIC neuron at 10% utilisation detected
- `test_hard_tanh_narrow_range_detected` — HARD_TANH neuron at 5% utilisation detected
