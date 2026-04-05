## Summary

Add a new `batch_successful` recommendation module that groups multiple individually
high-confidence candidates into combined operations for batch application. Unlike
epistatic pair detection (which finds synergistic pairs that individually fail), this
module batches proven winners for combined testing. Closes #965.

### What changed

- **New module** `src/analysis/recommendation/batch_successful/` with three files:
  - `mod.rs` — Types (`IndividualCandidate`, `BatchSuccessfulGroup`) and re-exports
  - `detection.rs` — Detects individually successful source→target synapse candidates
    from recorded data using least-squares error reduction
  - `grouping.rs` — Groups non-conflicting candidates into batches of 2–4, converts
    to `CoordinatedStructuralCandidateJson` with `AddSynapse` operations
- **Registered** as `"batch-successful grouping"` in `module_dispatch_specs/scoring_specs.rs`
  (now 48 total discovery modules)
- **Outcome tracking** via comment field containing `"Batch-successful:"` prefix,
  distinguishing these from epistatic and individual candidates
- Multi-operation discount (`COORDINATED_OPERATION_DISCOUNT^(N-1)`) applied automatically
  during the merge pipeline

## Evidence

All 14 new tests pass. `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build).

## Test Plan

- `tests/recommendation/issue_965_batch_successful_grouping.rs` — 14 integration tests:
  1. Detects individually successful candidates with clear improvement signals
  2. No candidates when improvement is too low (constant activations)
  3. Structural conflict detection: same source+target = conflict
  4. No conflict for different sources to same target
  5. Batch grouping produces groups of 2–4 candidates
  6. Batch respects max size (no batches > 4)
  7. Conversion produces valid coordinated candidates with `AddSynapse` operations
  8. Combined improvement is positive for all batch groups
  9. Empty records produce no candidates
  10. Insufficient samples produce no candidates
  11. Single candidate does not form a batch (minimum 2 required)
  12. Comments identify batch-successful type for outcome tracking
  13. End-to-end pipeline (`detect_batch_successful_groups` → conversion)
  14. Existing synapses are excluded from detection
- Module dispatch spec count test updated (47 → 48)
