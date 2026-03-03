## Summary

Eliminate intermediate `String` allocation in `deterministic_coordinated_neuron_uuid()` by hashing component bytes directly using FNV-1a instead of building a format string first. Floats are hashed via `to_bits().to_le_bytes()` instead of `{:.6}` formatting, which also eliminates a second allocation and improves float sensitivity (distinct bit patterns always produce distinct hashes). Closes #743.

## Evidence — Benchmark Results

| Metric | Baseline | After | Improvement |
|--------|----------|-------|-------------|
| 10,000 iterations | 3.445 ms | 1.162 ms | **-66.2%** |
| Single call | 354.8 ns | 123.6 ns | **-65.4%** |

The optimisation delivers a consistent ~65% speedup by eliminating one heap allocation per call and avoiding float formatting overhead.

## Test Plan

- Added `tests/issue_743_uuid_hashing.rs` with 6 tests:
  - `deterministic_same_inputs_produce_same_uuid` — same inputs yield identical UUIDs
  - `deterministic_different_inputs_produce_different_uuids` — varying each input component produces distinct UUIDs
  - `deterministic_uuid_has_correct_prefix` — output starts with `coordinated-hidden-`
  - `deterministic_uuid_has_correct_length` — output is exactly 35 characters
  - `deterministic_uuid_hex_suffix_is_valid` — hex suffix contains only valid hex digits
  - `deterministic_uuid_float_sensitivity` — floats differing by `f32::EPSILON` produce different UUIDs (this test **fails** on the old `{:.6}` formatting approach and **passes** with the new `to_bits()` approach)
- Added `benches/uuid_hashing.rs` criterion benchmark suite
- Updated `tests/issue_576_benchmark_regression_tracking.rs` to include the new benchmark suite
- `quality.sh` passes cleanly
