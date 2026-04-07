## Summary

Replace the fixed 9-variant weight grid with an adaptive Gaussian proposal distribution for synapse weight search. The new system centres proposals on the computed optimal weight and adapts its spread (sigma) based on historical acceptance rates per target neuron type. Falls back to the fixed grid when insufficient historical data is available. Closes #1019.

## Changes

### New Module: `src/analysis/synapse/adaptive_proposal.rs`
- **`AcceptanceTracker`**: Per-target-type acceptance rate tracker with sigma adaptation. Tracks accepted vs total proposals, adapts sigma toward a target acceptance rate (0.35), and clamps sigma between configurable bounds.
- **`generate_weight_candidates()`**: Main entry point that selects between adaptive Gaussian or fixed grid based on history.
- **`generate_gaussian_candidates()`**: Deterministic Gaussian-like proposal using Box-Muller transform with hash-based pseudo-random numbers. Includes sign-flip probability for negative weight exploration.
- **`generate_fixed_grid()`**: Extracted fixed 9-variant grid for fallback compatibility.

### Modified: `src/analysis/synapse/target_analysis/evaluation.rs`
- Replaced inline fixed `weight_candidates` array with call to `generate_weight_candidates()`.
- Added acceptance counting during weight evaluation (both holdout and fallback paths).
- Records batch outcomes to `AcceptanceTracker` for sigma adaptation.

### Modified: `src/analysis/synapse/target_analysis/mod.rs`
- Added `acceptance_tracker: Arc<Mutex<AcceptanceTracker>>` field to `TargetAnalysisContext`.

### Modified: `src/analysis/synapse/orchestration.rs`
- Creates and injects `AcceptanceTracker` into the shared analysis context.

### Modified: `src/analysis/constants/candidate_scoring.rs`
- Added 8 new constants for adaptive proposal configuration: initial sigma, candidate count, minimum history, target acceptance rate, adaptation rate, sigma bounds, and sign-flip probability.

## Evidence

- Hold-out validation is retained and unchanged (`holdout_validation.rs` unmodified).
- When insufficient history is available (< 15 samples), the system falls back to the original fixed 9-variant grid, preserving existing behaviour.
- All 34 new tests pass (18 unit + 16 integration).
- Full existing test suite passes with no regressions (`./quality.sh` passes cleanly).

## Test Plan

### Unit tests (`src/analysis/synapse/adaptive_proposal.rs`)
- `acceptance_tracker_initial_sigma` -- default sigma returned
- `acceptance_tracker_insufficient_history_uses_default` -- fallback with few samples
- `acceptance_tracker_adapts_sigma_down_on_high_acceptance` -- sigma shrinks
- `acceptance_tracker_adapts_sigma_up_on_low_acceptance` -- sigma grows
- `acceptance_tracker_sigma_clamped_to_bounds` -- min/max enforcement
- `acceptance_tracker_acceptance_rate` -- rate calculation
- `fixed_grid_produces_expected_candidates` -- fixed grid values
- `gaussian_candidates_include_optimal_weight` -- optimal weight preserved
- `gaussian_candidates_correct_count` -- candidate count
- `gaussian_candidates_are_deterministic` -- reproducibility
- `gaussian_candidates_differ_for_different_inputs` -- varied proposals
- `gaussian_candidates_vary_around_optimal` -- spread verification
- `gaussian_candidates_include_some_negative` -- sign-flip exploration
- `generate_weight_candidates_uses_fixed_grid_without_history` -- fallback path
- `generate_weight_candidates_uses_adaptive_with_history` -- adaptive path
- `all_gaussian_candidates_are_finite` -- numerical safety
- `gaussian_candidates_with_zero_weight` -- edge case
- `gaussian_candidates_with_negative_weight` -- edge case

### Integration tests (`tests/issue_1019_adaptive_proposal.rs`)
- `fallback_to_fixed_grid_without_history` -- compatibility verification
- `fallback_with_insufficient_history` -- threshold check
- `adaptive_proposal_activates_with_sufficient_history` -- activation check
- `adaptive_proposal_includes_optimal_weight` -- first candidate
- `sigma_decreases_on_high_acceptance` -- adaptation direction
- `sigma_increases_on_low_acceptance` -- adaptation direction
- `sigma_remains_bounded` -- bound enforcement
- `independent_sigma_per_target_type` -- per-type tracking
- `adaptive_covers_broader_range_with_large_sigma` -- exploration quality
- `all_candidates_are_finite` -- numerical safety
- `candidates_are_deterministic` -- reproducibility
- `different_inputs_produce_different_candidates` -- diversity
- `fixed_grid_expected_values` -- grid correctness
- `fixed_grid_includes_negative_weights` -- negative exploration
- `adaptive_handles_zero_weight` -- edge case
- `adaptive_handles_negative_weight` -- edge case
