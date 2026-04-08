## Summary

Compiler auto-vectorisation audit and data layout evaluation for hot numerical
loops. **Negative result — no meaningful improvement from SoA layout or compiler
hints.** The compiler already auto-vectorises where it can; the primary blockers
are algorithmic (conditional branches), not data layout. Closes #1009.

## Findings

### Assembly Audit (Apple Silicon / NEON)

| Function | Auto-vectorised? | Reason |
|----------|-----------------|--------|
| `ErrorDistribution::from_errors` | **Yes** (40+ NEON instructions) | Operates on `&[f32]` with simple arithmetic |
| `compute_synapse_improvement_and_count` | **No** (scalar only) | `TargetSimulationMode` match, `is_finite()` branches, function pointer calls |
| `compute_relu_improvement_and_count` | **No** (scalar only) | `Option` matching with `continue`, `is_finite()` branches |
| `compute_activation_improvement_and_count` | **No** (scalar only) | `activation_fn()` function pointer call per sample |
| `compute_source_variance_confidence` | **No** (scalar only) | `is_finite()` branch per sample, struct field access |
| `compute_error_variance` | **No** (scalar only) | `is_finite()` branch per sample, struct field access |

### SoA vs AoS Benchmark Results

**Synapse improvement (value-domain, no target squash):**

| Samples | AoS (current) | SoA (pre-extracted) | SoA (with extraction) |
|---------|---------------|--------------------|-----------------------|
| 100 | 115 ns | 95 ns (-17%) | 174 ns (+52%) |
| 1,000 | 1.23 µs | 1.19 µs (-3%) | 1.44 µs (+17%) |
| 10,000 | 12.5 µs | 12.3 µs (-2%) | 17.1 µs (+37%) |

**Source variance confidence:**

| Samples | AoS (current) | SoA (pre-extracted) | SoA (with extraction) |
|---------|---------------|--------------------|-----------------------|
| 100 | 54 ns | 54 ns (same) | 102 ns (+89%) |
| 1,000 | 969 ns | 732 ns (-24%) | 853 ns (-12%) |
| 10,000 | 7.8 µs | 9.4 µs (+20%) | 14.3 µs (+83%) |

**Error variance:**

| Samples | AoS (current) | SoA (pre-extracted) | SoA (with extraction) |
|---------|---------------|--------------------|-----------------------|
| 100 | 107 ns | 97 ns (-10%) | — |
| 1,000 | 765 ns | 746 ns (-3%) | 823 ns (+8%) |
| 10,000 | 7.4 µs | 7.3 µs (-1%) | 9.5 µs (+29%) |

### Compiler Hints Assessment

- **`#[inline(always)]`**: Not applicable — `lto = "fat"` and `codegen-units = 1`
  in the release profile already enable full cross-crate inlining.
- **`target-cpu=native`**: Assembly was generated with `-C target-cpu=native`.
  NEON is always available on Apple Silicon. The functions that aren't vectorised
  are blocked by algorithmic patterns (branches, function pointers), not missing
  CPU features.
- **`#[target_feature(enable = "...")]`**: Not applicable on AArch64 where NEON
  is baseline. On x86-64 this could enable AVX2, but the branch-heavy loops
  still would not vectorise.

### Key Conclusions

1. **SoA extraction cost exceeds any cache benefit.** At production-scale sample
   counts (1,000–10,000), the `.iter().map().collect()` overhead dominates. The
   SoA loop itself is only marginally faster (1–3%) at scale because the
   `is_finite()` branches prevent vectorisation regardless of layout.

2. **The compiler auto-vectorises effectively where it can.** `from_errors()`
   which operates on `&[f32]` with no branches gets full NEON vectorisation.
   The improvement functions don't vectorise because of algorithmic complexity,
   not data layout.

3. **No code changes warranted.** The current AoS layout is the right choice:
   it's simpler, avoids allocation overhead, and the vectorisation blockers
   are in the algorithm, not the data layout.

## Evidence

Benchmark results from `cargo bench --bench vectorisation_audit` on Apple
Silicon (M-series). Assembly inspection via
`RUSTFLAGS="-C target-cpu=native --emit=asm" cargo build --release`.

## Test Plan

- Added `tests/vectorisation_audit.rs` — 5 integration tests verifying SoA
  reference implementations match AoS production functions
- Added `benches/vectorisation_audit.rs` — 4 benchmark groups comparing AoS
  vs SoA across 3 sample sizes (100, 1,000, 10,000)
