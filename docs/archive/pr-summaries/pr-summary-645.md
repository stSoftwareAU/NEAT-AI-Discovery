## Summary

Add output neuron range compression detector that flags output neurons operating in a compressed sub-range of their activation function's theoretical range. For example, a TANH output neuron (range [-1, 1]) whose activations cluster in [0.3, 0.7] is using only 20% of the available range, reducing dynamic resolution and making weight adjustments less precise. Closes #645.

This complements the existing `restricted_range` module (hidden neurons) and `output_squash_mismatch` module (wrong function type) by detecting output neurons using the *correct* type of activation but at compressed utilisation.

### Changes

- **New detection module**: `src/analysis/detection/output_range_compression.rs`
  - Detects output neurons with <40% range utilisation of their bounded squash function
  - Excludes dead neurons, saturated neurons, unbounded activations, and hidden neurons
  - Produces two candidate types:
    - `changeSquash` to a better-fitting activation function
    - Coordinated `setBias` + `setWeight` to recentre and rescale the output pathway
- **Dispatch pipeline**: Wired into `scoring_specs.rs` for automatic execution during analysis
- **Re-exports**: Added to `detection/mod.rs` and `analysis/mod.rs` for backward compatibility
- **Scenario doc**: `docs/discoveries/output-range-compression.md`

## Evidence

This is a backend detection module with no UI changes. Correctness verified by 12 unit tests covering all detection criteria, edge cases, and candidate generation.

## Test Plan

- `tests/issue_645_output_range_compression.rs` — 12 tests:
  1. Compressed TANH output detected (20% utilisation)
  2. Full-range output NOT flagged (80% utilisation)
  3. Hidden neurons excluded
  4. Insufficient samples produce no detections
  5. Unbounded activations excluded
  6. Produces valid coordinated candidates (changeSquash + setBias/setWeight)
  7. LOGISTIC output compression detected
  8. Multiple outputs — only compressed ones flagged
  9. Near-saturated outputs excluded
  10. Dead outputs excluded
  11. Empty records produce no detections
  12. Results sorted by utilisation (worst first)
- `./quality.sh` passes cleanly (fmt, clippy, check, test, release build)
