## Summary

Add fan-in candidate generation that detects pairs of inputs whose activations
jointly predict a target's error and creates coordinated structural candidates
with a shared non-linear hidden neuron. Closes #908.

The new `fan_in` module in `src/analysis/recommendation/` detects complementary
input pairs via correlation analysis and two-variable least-squares regression,
then emits `CoordinatedStructuralCandidateJson` candidates containing:
- `AddNeuron` (hidden, TANH activation) to capture interaction effects
- `AddSynapse` per input to the hidden neuron (with optimised weights)
- `AddSynapse` from hidden neuron to target

Key design decisions:
- Non-linear activations only (TANH/GELU) — IDENTITY is never used for fan-in
  since it reduces to a linear combination that cannot capture interactions
- Inputs must have low mutual correlation (< 0.8) to ensure complementarity
- Combined regression improvement must exceed best individual by 5% ratio
- Conservative 0.01× scaling on estimated improvement (multi-operation discount)
- Deterministic UUID generation via FNV-1a hash of sorted input + target UUIDs

## Evidence

All 13 unit tests pass, plus the updated module count test (45 → 46).

## Test Plan

- `tests/recommendation/issue_908_fan_in_candidates.rs` — 13 tests:
  1. Detects fan-in candidates for correlated inputs
  2. No candidates for uncorrelated inputs
  3. Non-linear activations enforced (not IDENTITY)
  4. Valid coordinated structural operations produced
  5. Deterministic UUID generation
  6. Empty records produce no candidates
  7. Insufficient samples produce no candidates
  8. Single input cannot form fan-in pair
  9. Candidates sorted by estimated improvement
  10. Redundant (highly correlated) inputs filtered out
  11. Fan-in targets hidden neurons as well as outputs
  12. Estimated improvement always positive
  13. Conversion requires at least 2 inputs
