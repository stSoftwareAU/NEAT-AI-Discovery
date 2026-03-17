## Summary

Add release profile optimisation with `lto = "fat"` and `codegen-units = 1` to `Cargo.toml`. This enables link-time optimisation and single codegen unit for release builds, producing better-optimised binaries at the cost of longer compile times. Closes #741.

## Evidence — Benchmark Results

Benchmark: `cargo bench --bench parallel_discovery` (full `analyze_all()` pipeline)

| Scenario | Baseline (ms) | With LTO (ms) | Change |
|----------|--------------|---------------|--------|
| 5h_100r  | 166.11       | 72.87         | **-56%** |
| 20h_200r | 113.12       | 88.19         | **-22%** |
| 50h_200r | 295.56       | 272.13        | **-8%**  |

The smaller creatures see the largest improvement (up to 56%) because CPU-bound analysis dominates over GPU work. Larger creatures are more GPU-bound, so the CPU optimisation has less impact — but still a meaningful 8% improvement.

**Trade-off**: Release compile time increased from ~12s to ~3m07s (full rebuild) and ~1m11s (incremental). This only affects release builds; debug builds are unchanged.

## Test Plan

- No new tests required — this is a build configuration change only
- `quality.sh` passes (includes fmt, clippy, check, tests, release build)
- Benchmark results above demonstrate the improvement
