## Summary

Add analysis pipeline wall-clock utilisation benchmarks to establish a baseline for measuring improvements from parallelisation work in Issue #999. Closes #1001.

New benchmark file `benches/pipeline_utilisation.rs` exercises `analyze_all()` with representative test creatures and measures:

- **Wall-clock time per phase** via Criterion and `ProfileData`
- **Rayon thread pool utilisation** (active vs idle thread-seconds)
- **GPU queue saturation** (busy vs idle time via `global_gpu_metrics()`)
- **Per-variant timing** for full pipeline, synapse-only, and neuron-only configurations

Results are printed in a formatted utilisation report suitable for before/after comparison.

## Evidence

- Benchmark compiles and passes `cargo check --bench pipeline_utilisation`
- All quality checks pass (`./quality.sh` clean)
- Benchmark is runnable with `cargo bench --bench pipeline_utilisation`

## Test Plan

- Benchmark registered in `Cargo.toml` with `harness = false`
- `benches/pipeline_utilisation.rs` exercises `analyze_all()` across three pipeline variants (full, synapse-only, neuron-only) with two creature sizes (10h/150r, 30h/200r)
- Utilisation report covers CPU thread utilisation and GPU queue saturation metrics
- Follows existing benchmark patterns from `benches/parallel_discovery.rs`
