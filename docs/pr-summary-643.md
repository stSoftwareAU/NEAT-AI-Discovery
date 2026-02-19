## Summary

Add activation-error monotonicity detector that identifies hidden neurons with
non-monotonic activation-error relationships using Spearman's rank correlation.
A well-functioning hidden neuron should have a consistent (monotonic) relationship
between its activation and output error. Neurons where this relationship is
contradictory (e.g., U-shaped or W-shaped) are encoding multiple features that
interfere with each other and should be restructured. Closes #643.

## What Changed

- **New detection module** (`src/analysis/detection/monotonicity.rs`):
  - Computes Spearman's rank correlation (rho) between activation and absolute error
  - Flags hidden neurons where |rho| < 0.3 (non-monotonic)
  - Produces `addNeuron` candidates (to split the workload) or `changeSquash`
    candidates (to try a better-fitting activation function)
- **Dispatch pipeline wiring** (`src/analysis/module_dispatch_specs/neuron_specs.rs`):
  - Registered as "monotonicity detection" phase
- **Module registration** (`src/analysis/detection/mod.rs`, `src/analysis/mod.rs`):
  - Re-exported at analysis level for backward compatibility
- **Scenario documentation** (`docs/discoveries/monotonicity.md`):
  - Explains the problem, detection method, and recommended actions
  - Clarifies distinction from `noise_signal.rs` (variance-based) and
    `gradient_discovery.rs` (synapse-level gradients)

## Evidence

This is a backend detection module with no UI changes. Evidence is provided by
the 10 integration tests listed below.

## Test Plan

Added `tests/issue_643_activation_error_monotonicity.rs` with 10 tests:

1. `test_monotonic_increasing_not_flagged` — monotonic increasing relationship is not flagged
2. `test_monotonic_decreasing_not_flagged` — monotonic decreasing relationship is not flagged
3. `test_non_monotonic_u_shape_flagged` — U-shaped activation-error is detected
4. `test_non_monotonic_inverted_u_flagged` — inverted-U activation-error is detected
5. `test_input_output_neurons_excluded` — input/output neurons are never flagged
6. `test_insufficient_samples_no_detection` — fewer than 20 samples produces no detections
7. `test_produces_valid_structural_candidates` — candidates include addNeuron or changeSquash
8. `test_multiple_neurons_only_non_monotonic_flagged` — only non-monotonic neurons flagged
9. `test_estimated_improvement_positive` — estimated improvement is always positive
10. `test_empty_records_no_detections` — empty records produce no detections
