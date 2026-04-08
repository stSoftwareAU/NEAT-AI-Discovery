## Summary

Add SIMD micro-benchmark infrastructure with baseline measurements for the six
hot numerical loops identified as SIMD candidates. Closes #1006.

**SIMD approach decision:** rely on compiler auto-vectorisation first (stable
Rust, no unsafe code, no extra crates). Explicit SIMD (`std::arch`, `wide`, or
nightly `std::simd`) will only be introduced if auto-vectorisation proves
insufficient in follow-up work — benchmarked here to measure that.

### Changes

- **New benchmark** `benches/simd_hot_paths.rs` — Criterion micro-benchmarks for
  all 6 target functions at 3 sample sizes (100, 1 000, 10 000) with realistic
  production-scale test data including edge cases (NaN/Inf, all-zero activations,
  optional target fields).
- **Visibility widened** for 5 functions so benchmarks (external to the crate)
  can call them directly:
  - `compute_synapse_improvement_and_count` — `pub(crate)` → `pub`
  - `compute_relu_improvement_and_count` — `pub(crate)` → `pub`
  - `compute_activation_improvement_and_count` — `pub(crate)` → `pub`
  - `compute_source_variance_confidence` — private → `pub`
  - `compute_error_variance` — private → `pub`
- `Cargo.toml` — added `[[bench]]` entry for `simd_hot_paths`.

## Evidence

### Baseline Benchmark Results

| Function | 100 samples | 1 000 samples | 10 000 samples |
|----------|-------------|---------------|----------------|
| `synapse_improvement` (value domain) | 111 ns | 1.24 µs | 12.6 µs |
| `synapse_improvement` (TANH sim) | 116 ns | 1.24 µs | 12.6 µs |
| `relu_improvement` (no target fn) | 93.6 ns | 1.22 µs | 12.7 µs |
| `relu_improvement` (TANH target) | 369 ns | 3.90 µs | 40.9 µs |
| `activation_improvement` (TANH, no target) | 218 ns | 2.56 µs | 27.5 µs |
| `activation_improvement` (TANH + target) | 571 ns | 5.96 µs | 62.2 µs |
| `ErrorDistribution::from_errors` | 597 ns | 7.91 µs | 85.8 µs |
| `source_variance_confidence` | 56.6 ns | 706 ns | 7.22 µs |
| `error_variance` | 51.7 ns | 671 ns | 6.94 µs |

All functions scale linearly with sample count, confirming they are bounded by
the inner loop iteration. The variance/confidence functions are the fastest
(simple sum-of-squares), while the target-simulation paths (TANH) are ~3-4×
slower due to `f32::tanh()` calls per sample — a prime target for SIMD.

## Test Plan

- `quality.sh` passes (fmt, clippy, check, test, doc, release build)
- `cargo bench --bench simd_hot_paths` runs all 6 benchmark groups successfully
- Benchmark data generated with edge cases: NaN/Inf values, all-zero activations,
  missing optional fields (~20 % None)
- No existing tests modified or removed
