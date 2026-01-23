## Summary

This PR ensures the squash (activation) functions in NEAT-AI-Discovery are consistent with the NEAT-AI WASM implementation. Inconsistencies between these implementations can cause failed candidates during discovery because predictions won't match actual outcomes from the neural network.

### Changes Made

**Fixed three key inconsistencies:**

1. **STDINVERSE epsilon**: Changed from `1e-15` to `1e-10` to match NEAT-AI WASM
   - Values with `|x| < 1e-10` are now clamped to `±1e-10` before computing `1/x`
   - Previously used a tighter epsilon of `1e-15`

2. **EXPONENTIAL cutoff and saturation value**: Changed from `x >= 88.0 → f32::MAX` to `x >= 36.0 → JS_MAX_SAFE_INTEGER (~9e15)`
   - Matches the NEAT-AI WASM implementation which uses JavaScript's safe integer boundary
   - Prevents extremely large values that could cause numerical instability

3. **SOFTPLUS cutoff and saturation value**: Changed from `x > 20.0 → x` to `x >= 709.0 → 100.0`
   - Matches the NEAT-AI WASM implementation
   - Non-finite inputs now return `1e-15` instead of computing ln(1+exp(x))

4. **MISH simplification**: Removed unnecessary cutoff in the softplus component since `tanh()` saturates anyway

### Why This Matters

The NEAT-AI project has migrated its squash functions from TypeScript to WASM/Rust. During this migration, the implementations were reconciled and corrected. This PR brings NEAT-AI-Discovery in line with those corrections to ensure:

- Discovery predictions match actual network behaviour
- Candidate evaluation is consistent between Discovery and NEAT-AI
- Reduced failed candidates due to numerical mismatches

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface. The changes are validated by comprehensive test suites.

**Test output showing consistency:**
```
running 37 tests
test test_identity_consistency ... ok
test test_relu_consistency ... ok
test test_exponential_overflow_handling ... ok
test test_softplus_extreme_values ... ok
test test_std_inverse_near_zero ... ok
... (all 37 tests pass)
```

## Test Plan

### New Tests Added

- `tests/squash_consistency_with_neat_ai.rs` - 37 tests comparing each squash function against the NEAT-AI WASM reference implementation:
  - `test_identity_consistency`
  - `test_relu_consistency`
  - `test_relu6_consistency`
  - `test_leaky_relu_consistency`
  - `test_selu_consistency`
  - `test_elu_consistency`
  - `test_elu_at_zero`
  - `test_logistic_consistency`
  - `test_tanh_consistency`
  - `test_hard_tanh_consistency`
  - `test_softsign_consistency`
  - `test_softplus_consistency`
  - `test_softplus_extreme_values` - Verifies x >= 709 returns 100.0
  - `test_swish_consistency`
  - `test_mish_consistency`
  - `test_gelu_consistency`
  - `test_sine_consistency`
  - `test_cosine_consistency`
  - `test_tan_consistency`
  - `test_arctan_consistency`
  - `test_gaussian_consistency`
  - `test_gaussian_extreme_values`
  - `test_bent_identity_consistency`
  - `test_bipolar_sigmoid_consistency`
  - `test_bipolar_consistency`
  - `test_step_consistency`
  - `test_complement_consistency`
  - `test_absolute_consistency`
  - `test_square_consistency`
  - `test_cube_consistency`
  - `test_sqrt_consistency`
  - `test_std_inverse_consistency`
  - `test_std_inverse_near_zero` - Verifies epsilon handling at 1e-10
  - `test_exponential_consistency`
  - `test_exponential_overflow_handling` - Verifies x >= 36.0 returns JS_MAX_SAFE_INTEGER
  - `test_logsigmoid_consistency`
  - `test_isru_consistency`

### Modified Tests

- `tests/activation_overflow_protection_f32.rs` - Updated to expect `JS_MAX_SAFE_INTEGER` instead of `f32::MAX` for EXPONENTIAL saturation:
  - `exponential_should_saturate_for_large_inputs`
  - `exponential_target_simulation_should_saturate_for_large_inputs`
  - Added `exponential_at_cutoff_boundary` to verify the 36.0 cutoff

### Verification

All 424 tests pass after running `./quality.sh`:
- 379 unit tests
- 45 integration tests

Closes #323
