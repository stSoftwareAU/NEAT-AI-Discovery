## Summary

Replace 4 `unsafe { sample.target_value.unwrap_unchecked() }` and `unsafe { sample.target_activation.unwrap_unchecked() }` calls in `src/analysis/synapse/scoring.rs` with safe `debug_assert!` + `unwrap()` alternatives. This eliminates undefined behaviour risk in two critical scoring functions (`compute_relu_improvement_and_count` and `compute_activation_improvement_and_count`) while preserving the invariant documentation. Closes #521.

### Changes

- **`src/analysis/synapse/scoring.rs`**: Replaced all 4 `unsafe { ... unwrap_unchecked() }` calls with `debug_assert!` guards and safe `.unwrap()`. The `debug_assert!` messages document the invariant: `get_target_simulation_fn()` only returns `Some` when all samples have `target_value` and `target_activation` set.
- **`src/analysis/implementation_tests/safe_unwrap_tests.rs`** (new): 4 tests exercising the target simulation code paths in both `compute_relu_improvement_and_count` and `compute_activation_improvement_and_count`.
- **`src/analysis/implementation_tests/mod.rs`**: Registered the new test module.

### Approach

Used **Option A** from the issue: `debug_assert!` + safe `unwrap()`. This gives:
- Clean panics instead of undefined behaviour if the invariant is violated
- Debug-mode assertions that catch violations early during development
- Explicit documentation of the invariant and where it is established

## Evidence

This is a backend code safety improvement with no UI changes. All 478 unit tests and 97 integration test files pass. `./quality.sh` passes cleanly including fmt, clippy, check, tests, and release build.

## Test Plan

- `safe_unwrap_tests::relu_improvement_with_target_simulation_uses_safe_access` — exercises `compute_relu_improvement_and_count` with `target_activation_fn = Some(...)`, verifying correct results via the safe unwrap path
- `safe_unwrap_tests::relu_improvement_target_simulation_activation_domain` — verifies saturation-aware vs linear approximation near boundaries
- `safe_unwrap_tests::activation_improvement_with_target_simulation_uses_safe_access` — exercises `compute_activation_improvement_and_count` with target simulation
- `safe_unwrap_tests::activation_improvement_target_simulation_near_saturation` — verifies finite results near saturation boundaries
