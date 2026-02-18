## Summary

Add property-based testing with `proptest` for mathematical functions across
scoring, confidence, error distribution, cross-validation, weight calculation,
activation functions, and neuron fingerprinting. Closes #573.

This adds 53 property-based tests organised into 8 test groups that complement
the existing example-based test suite by systematically exploring edge cases,
boundary conditions, NaN/Inf propagation, and mathematical invariants.

## Changes

- **Cargo.toml**: Added `proptest = "1.6"` to `[dev-dependencies]`
- **tests/issue_573_proptest_mathematical_functions.rs**: New integration test
  file with 53 property-based tests

## Test Groups (8 mathematical function groups)

1. **Confidence metrics** (5 tests): Bounded [0, 1], CI interval ordering,
   empty-sample handling, monotonicity with sample count
2. **Error distribution** (9 tests): Non-negative variance, std_dev consistency,
   mean within [min, max], percentile ordering, IQR non-negative, constant data,
   kurtosis positivity
3. **Cross-validation / brittleness** (7 tests): Penalty bounded [0, 1],
   zero/full penalty behaviour, monotonic decrease, fold improvement ratio
   bounds, variance non-negativity
4. **Weight calculation** (5 tests): Outgoing weight clamped to MAX, zero
   activation returns None, delta consistency, coordinated delta finiteness
5. **Activation functions** (9 tests): Bounded activations stay in range,
   RELU/ABSOLUTE/SQUARE non-negative, IDENTITY invariance, LOGISTIC/TANH/RELU
   monotonicity, GAUSSIAN symmetry, COMPLEMENT identity, determinism
6. **NaN/Inf injection** (8 tests): All scalar activations NaN-free for finite
   inputs, confidence handles NaN/Inf samples, error distribution filters
   non-finite values, weight calculation NaN/Inf safety, cross-validation
   NaN/Inf safety, EXPONENTIAL/SOFTPLUS overflow protection, STDINVERSE epsilon
   protection, SQRT negative-input safety
7. **Neuron fingerprinting** (3 tests): Determinism (same creature = same
   fingerprint), sensitivity to weight changes, sensitivity to bias changes
8. **NaN-safe comparison** (3 tests): `cmp_f32_desc`/`cmp_f32_asc` total order
   with NaN, sorting determinism with mixed NaN/finite values

## Evidence

This is a purely additive testing change with no UI or performance implications.
All 53 proptest tests pass in < 1 second, and the full `quality.sh` gate passes.

```
running 53 tests
test result: ok. 53 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.81s
```

## Test Plan

- Added `tests/issue_573_proptest_mathematical_functions.rs` with 53 property
  tests across 8 mathematical function groups
- All existing tests continue to pass (`cargo test --test-threads=1`)
- `quality.sh` passes (fmt, clippy, check, test, release build)
- Proptest suite completes in < 1 second (well under the 30-second target)
- Australian English used throughout
