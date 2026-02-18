## Summary

Adds a new `restricted_range` discovery module (Issue #399) that detects hidden neurons operating in a restricted sub-range of their activation function's output domain. For example, a TANH neuron consistently outputting values in [0.1, 0.3] is only using 10% of its [-1, 1] range, wasting representational capacity.

This complements existing detection modules:
- **Saturation detection** (Issue #342): neurons stuck at activation bounds.
- **Dead neuron detection** (Issue #341): neurons with near-zero activation.
- **Bounded range / sentinel detection** (Issue #395): sentinel clusters at boundary values.
- **Restricted range** (this PR): neurons active but confined to a narrow band within the theoretical bounds.

### Changes

- **New module**: `src/analysis/restricted_range.rs` — detection logic and candidate generation.
- **Integration**: Wired into `analyze_all()` via `run_discovery_module` dispatch pattern (DRY, Issue #375).
- **Module registration**: Added to `src/analysis/mod.rs`.
- **Tests**: 11 TDD tests in `tests/issue_399_restricted_range_detection.rs`.

### Detection Logic

- Computes `range_utilisation = (activation_max - activation_min) / theoretical_range` for bounded squash functions.
- Flags neurons below a configurable utilisation threshold (default: 20%).
- Excludes unbounded activations (IDENTITY, RELU, etc.), output neurons, dead neurons, and saturated neurons.

### Candidate Types

- `changeSquash` — switch to IDENTITY to remove bounding.
- `setBias` — adjust bias to centre the operating region.
- `setWeight` + `setBias` (coordinated) — scale incoming weights to expand range.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

All 11 tests in `tests/issue_399_restricted_range_detection.rs`:
- `test_detects_tanh_neuron_with_restricted_range` — TANH using 10% of range is detected
- `test_does_not_flag_neuron_using_full_range` — TANH at 80% utilisation not flagged
- `test_unbounded_activations_excluded` — IDENTITY and RELU skipped
- `test_output_neurons_excluded` — output neurons skipped
- `test_insufficient_samples_skipped` — fewer than 20 samples skipped
- `test_dead_neurons_excluded` — near-zero activation excluded
- `test_candidate_generation_includes_expected_operations` — produces changeSquash/setBias/setWeight
- `test_configurable_utilisation_threshold` — custom threshold works
- `test_multiple_neurons_only_restricted_detected` — only restricted neurons flagged
- `test_detects_logistic_neuron_with_restricted_range` — LOGISTIC at 4% utilisation detected
- `test_saturated_neurons_not_flagged_as_restricted` — saturation excluded
