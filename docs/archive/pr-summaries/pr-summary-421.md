## Summary

Implements gradient-based discovery (Issue #421) — a new discovery module that computes
local gradients (∂error/∂weight) for each synapse and proposes weight adjustments in
the error-reducing direction.

Unlike existing correlation-based methods that measure association strength, gradient-based
discovery provides directional information: it tells you not just *which* synapses matter,
but *which direction* to adjust them for maximum error reduction.

### How it works

1. **Compute local gradients**: For each synapse targeting an output neuron, computes
   `∂error/∂weight ≈ mean(source_activation × target_error)` across all paired observations.
2. **Identify high-gradient synapses**: Filters for synapses where the gradient magnitude
   exceeds a minimum threshold (0.01) and the gradient direction is consistent (signal-to-noise
   ratio ≥ 0.3).
3. **Propose gradient-directed weight adjustments**: Suggests `SetWeight` operations in the
   gradient descent direction with a conservative learning rate (0.1).

### New files

- `src/analysis/gradient_discovery.rs` — Detection and candidate conversion
- `tests/issue_421_gradient_based_discovery.rs` — 12 integration tests

### Modified files

- `src/analysis/mod.rs` — Module registration and dispatch in `analyze_all()`
- `docs/DISCOVERY_TYPES.md` — Documentation for the new discovery type

## Evidence

This is a backend library change with no UI. Evidence is provided through tests:

- 12 integration tests covering gradient computation accuracy, directional correctness,
  edge cases (empty records, insufficient samples, non-finite values), candidate conversion,
  and improvement scaling.
- All tests pass via `cargo test --test issue_421_gradient_based_discovery -- --test-threads=1`
- Full `quality.sh` passes (fmt, clippy, check, test, release build)

## Test Plan

- `test_gradient_computation_basic` — Verifies gradient calculation matches expected values
- `test_gradient_zero_activations` — Zero activations produce zero/no gradient
- `test_detects_high_gradient_synapse` — High-gradient synapses are correctly identified
- `test_weight_adjustment_direction` — Positive gradient proposes negative delta (gradient descent)
- `test_low_gradient_not_detected` — Low-gradient synapses are filtered out
- `test_gradient_to_coordinated_candidates` — Valid coordinated candidates with positive improvement
- `test_empty_records_no_candidates` — Edge case: empty records
- `test_insufficient_samples_no_candidates` — Edge case: too few samples
- `test_non_finite_values_handled` — NaN/Infinity handled safely
- `test_gradient_sign_determines_direction` — Negative gradient proposes positive delta
- `test_improvement_scales_with_gradient` — Higher gradient = higher estimated improvement
- `test_candidates_sorted_by_improvement` — Candidates sorted best-first
