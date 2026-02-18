## Summary

Split `src/analysis/scoring/weights.rs` (~37KB, 963 lines) into focused sub-modules under `src/analysis/scoring/weights/`. Closes #609.

### New structure

| File | Lines | Purpose |
|------|-------|---------|
| `weights/mod.rs` | ~70 | Public API, re-exports, constants (`MAX_OUTGOING_WEIGHT`, `MIN_WEIGHT_RATIO`) |
| `weights/calculation.rs` | ~280 | Core weight calculation (`calculate_optimal_outgoing_weight`, `calculate_optimal_identity_outgoing_and_bias`, `calculate_optimal_bias`, `hard_tanh`) |
| `weights/normalisation.rs` | ~85 | Range-aware weight computation — sentinel filtering (`compute_range_aware_sums`, `calculate_range_aware_weight`) |
| `weights/adjustment.rs` | ~70 | Dynamic weight adjustments (`clamp_weight_update_delta`, `coordinated_structural_activation_delta`) |

All public API items are re-exported from `weights/mod.rs`, so existing imports (`crate::analysis::weights::*` and `crate::analysis::*`) remain unchanged.

## Evidence

This is a pure code reorganisation with no visual or behavioural changes. All existing tests pass unchanged, and `quality.sh` passes cleanly (fmt, clippy, check, test, release build).

## Test Plan

- All 30+ existing unit tests in `weights/mod.rs` pass unchanged
- All integration tests in `tests/` pass without modification
- `cargo clippy` clean, `cargo fmt` clean, release build succeeds
