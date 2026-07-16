## Summary

Add a target saturation pre-check during neuron candidate preparation that detects when target neurons are operating near activation bounds, and adjusts candidate generation accordingly. Closes #1111.

### What changed

- **`src/analysis/neuron/preparation.rs`**: Added `TargetSaturationInfo`, `compute_target_saturation()`, `bounded_output_range()`, and `compounds_target_clipping()` functions. The saturation check computes how much of a bounded activation's output range is covered by the observed activation min/max. Targets covering >90% are flagged as near-saturated.
- **`src/analysis/neuron/evaluation.rs`**: Added `apply_target_saturation_discount()` which sets `target_saturation_factor` on candidates and discounts expected gains. Added filtering of activation specs that compound clipping with saturated targets (e.g., ABSOLUTE feeding into HARD_TANH).
- **`src/analysis/neuron/mod.rs`**: Wired the saturation pre-check into the per-target analysis loop, computing saturation info from target records before candidate evaluation.
- **`src/ffi_types/candidates.rs`**: Added `target_saturation_factor: Option<f32>` field to `CandidateNeuronJson` for downstream scoring.

## Evidence

This is a backend enhancement with no UI changes. Verified by:
- 8 new unit tests covering all acceptance criteria
- All 171 existing tests pass unchanged
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)

## Test Plan

New tests in `src/analysis/neuron/preparation.rs`:
- `test_hard_tanh_saturated_target_triggers_precheck` — HARD_TANH target at activation range [-1, 1] triggers the saturation pre-check
- `test_identity_target_not_saturated` — IDENTITY target (unbounded) does NOT trigger the pre-check
- `test_tanh_narrow_range_not_saturated` — TANH with narrow range is not flagged
- `test_logistic_near_saturated` — LOGISTIC spanning 96% of range is flagged
- `test_empty_records_not_saturated` — Empty records return non-saturated
- `test_absolute_compounds_hard_tanh_clipping` — ABSOLUTE compounding with bounded targets
- `test_bounded_output_range` — Output range lookup for all bounded activations
- `test_relu_target_not_saturated` — Unbounded RELU returns non-saturated
