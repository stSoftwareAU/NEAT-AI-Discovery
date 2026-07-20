# 📊 Benchmark Regression Tracking

This document describes the benchmark comparison workflow for detecting
performance regressions using Criterion.

## 🔍 Overview

The project includes 42 Criterion benchmark suites in `benches/`. The
`benchmark_compare.sh` script automates baseline saving and regression
detection by leveraging Criterion's built-in comparison features.

Baselines are machine-specific — each developer or CI runner saves their own
baseline locally in `target/criterion/`.

## 🧪 Optimisation outcomes

This section records the durable outcomes of past optimisation investigations —
including **negative results** — so the same fruitless approaches are not
re-attempted. Nothing in the code fails when a negative result is lost; the
effort is simply re-spent. These findings were folded here from the PR-summary
archive (Issue #1682).

### 🧭 SIMD / vectorisation of the hot numerical loops

- **Auto-vectorisation first (policy, #1006).** Rely on compiler
  auto-vectorisation before reaching for explicit SIMD (`std::arch`, `wide`, or
  nightly `std::simd`). Explicit SIMD stays out of the codebase unless
  auto-vectorisation is measured insufficient — it adds `unsafe`, extra crates,
  and portability cost. Baselines for the six hot loops
  (`benches/simd_hot_paths.rs`) show the target-simulation (`tanh`) paths are
  ~3–4× slower than the value-domain paths because of the `f32::tanh()` call per
  sample.

- **Negative result — Struct-of-Arrays layout and compiler hints do not help
  (#1009).** An Apple Silicon / NEON assembly audit showed the hot improvement
  functions (`compute_synapse_improvement_and_count`,
  `compute_relu_improvement_and_count`, `compute_activation_improvement_and_count`,
  `compute_source_variance_confidence`, `compute_error_variance`) are blocked
  from auto-vectorisation by **`is_finite()` branches and function-pointer calls**
  — an *algorithmic* blocker, not data layout. `ErrorDistribution::from_errors`,
  which is branch-free over `&[f32]`, already gets full NEON vectorisation.
  Converting to a Struct-of-Arrays layout gave at best 1–3 % at production scale
  and the `.iter().map().collect()` extraction cost usually made it *slower*.
  `#[inline(always)]` (redundant under `lto = "fat"` + `codegen-units = 1`),
  `target-cpu=native`, and `#[target_feature]` were all assessed **not
  applicable** (NEON is baseline on AArch64; the branch-heavy loops would not
  vectorise on x86-64 AVX2 either). **No code changes were warranted** — the
  current AoS layout is the right choice. Do not re-attempt SoA or compiler-hint
  tweaks on these loops without first removing the branch/function-pointer
  blockers.

- **Branch elimination helps only the value-domain paths (caveat, #1075).**
  Eliminating branches from the no-target ("value-domain") paths gave a real
  **30–34 % throughput improvement**. On `tanh`-dominated paths the same change
  was **within noise** (±1–6 %) because `f32::tanh()` per sample dominates the
  loop cost. Optimise the `tanh` paths by attacking the transcendental cost, not
  the surrounding branches.

### ⚙️ Release-profile link-time optimisation (#741)

The release profile pins `lto = "fat"` and `codegen-units = 1`
(`Cargo.toml`, `[profile.release]`). Measured against the full `analyze_all()`
pipeline (`cargo bench --bench parallel_discovery`):

| Scenario (creature size) | Baseline | With LTO | Change |
|--------------------------|----------|----------|--------|
| 5h_100r (small)          | 166.11 ms | 72.87 ms | **−56 %** |
| 20h_200r (medium)        | 113.12 ms | 88.19 ms | **−22 %** |
| 50h_200r (large)         | 295.56 ms | 272.13 ms | **−8 %** |

Smaller creatures gain most because CPU-bound analysis dominates; larger
creatures are more GPU-bound, so the CPU optimisation has less headroom.
**Trade-off:** full release compile time rose from ~12 s to ~3 m 07 s
(~1 m 11 s incremental). Debug builds are unaffected. Keep both settings unless a
future measurement shows the compile-time cost outweighs the runtime win.

## 🚀 Quick Start

```bash
# 1. Save a baseline (run all benchmarks and record timings)
./benchmark_compare.sh --save-baseline

# 2. Make code changes ...

# 3. Compare against the baseline
./benchmark_compare.sh
```

## ⌨️ Commands

| Command | Description |
|---------|-------------|
| `./benchmark_compare.sh --save-baseline` | Run all benchmarks and save as baseline |
| `./benchmark_compare.sh` | Compare current results against saved baseline |
| `./benchmark_compare.sh --threshold 10` | Set regression threshold to 10% (default: 5%) |
| `./benchmark_compare.sh --bench synapse_counts` | Compare a single benchmark suite |
| `./benchmark_compare.sh --list` | List all available benchmark suites |
| `./benchmark_compare.sh --help` | Show usage information |

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_THRESHOLD` | `5` | Regression threshold percentage |
| `BENCHMARK_BASELINE` | `saved` | Baseline name for Criterion |

## 🧪 Benchmark Suites

The following suites are defined in `Cargo.toml`:

| Suite | Focus |
|-------|-------|
| `analysis_pipeline_clones` | Clone reduction in analysis detection/recommendation pipeline |
| `async_pipeline` | CPU/GPU overlap pipeline throughput |
| `batched_activation` | Batched vs sequential activation evaluation |
| `bfs_allocation` | Visited-set allocation strategies in BFS |
| `cache_eviction` | Cache eviction patterns under memory pressure (Issue #1040) |
| `cache_locality` | Cache access patterns |
| `candidate_pipeline_clones` | Clone reduction in candidate pipeline operations |
| `clone_reduction` | Clone elimination in hot paths |
| `cpu_pre_reject` | CPU pre-reject screen before the helpful GPU submit (Issue #1544) |
| `error_collection` | Lock-free vs mutex-based error collection |
| `ffi_marshalling` | FFI JSON marshalling overhead across the FFI boundary (Issue #1040) |
| `gpu_buffer_transfers` | GPU buffer transfer overhead |
| `gpu_helpful_chunk_reuse` | GPU buffer reuse across helpful-batch chunks (Issue #1369) |
| `gpu_shader_workgroup` | GPU shader workgroup optimisation |
| `impact_uuid_cloning` | UUID string cloning in focus/impact hot loops |
| `input_uuid_precompute` | Input-neuron UUID construction on the recording hot path (Issue #1368) |
| `memory_streaming` | Memory streaming performance |
| `module_tiering_dispatch` | Creature-scale module tiering during discovery dispatch (Issue #1547) |
| `neuron_interning` | Neuron UUID interning |
| `neuron_squash_pruning` | Squash-aware hidden-target `ACTIVATION_SPECS` pruning (Issue #1545) |
| `parallel_discovery` | Parallel discovery throughput |
| `parquet_loading_comparison` | Parquet loading strategy comparison across cache tiers (Issue #1040) |
| `pipeline_utilisation` | Analysis pipeline wall-clock utilisation across `analyze_all()` (Issue #1001) |
| `quality_skip_dispatch` | Quality-based module skipping during dispatch merge (Issue #1074) |
| `queue_submission_copies` | GPU queue submission copy overhead |
| `record_arc_sharing` | Arc-shared `DiscoverRecord` across detection modules (Issue #1543) |
| `sample_locality` | Sample data locality |
| `simd_hot_paths` | SIMD baseline micro-benchmarks for hot numerical loops (Issue #1006) |
| `source_budget` | Per-target source budget (top-K) vs unlimited enumeration (Issue #1542) |
| `squash_normalisation` | Pre-normalised squash string lookup |
| `synapse_counts` | Synapse count pre-computation |
| `synapse_lookup` | Zero-allocation synapse existence and weight lookups |
| `synapse_preparation` | String cloning cost in synapse preparation |
| `tiered_loading` | Tiered cache loading strategies |
| `topology_cache` | Pre-computed topology cache vs repeated map building |
| `topology_traversal` | Backtracking vs HashSet clone in topology traversal |
| `upsert_candidate` | Candidate upsert operations |
| `uuid_arc_preparation` | UUID string cloning reduction in analysis preparation (Issue #1036) |
| `uuid_hashing` | Deterministic UUID generation for candidates |
| `vectorisation_audit` | Compiler auto-vectorisation audit of AoS iteration (Issue #1009) |
| `weight_coherence_cache` | Weight coherence detection with topology cache |
| `zero_copy_buffer` | Zero-copy vs traditional GPU buffer sharing (Issue #228) |

**Note:** Most benchmarks require a GPU. Suites that cannot initialise a GPU
are automatically skipped.

## 🔄 Workflow

### 🏁 Initial Setup

Save a baseline on your machine before making performance-sensitive changes:

```bash
./benchmark_compare.sh --save-baseline
```

This runs all 42 benchmark suites and stores results in `target/criterion/`.

### 🔎 Detecting Regressions

After making changes, compare against the baseline:

```bash
./benchmark_compare.sh
```

The script reports:
- **Regressions** — benchmarks that slowed by more than the threshold
- **Improvements** — benchmarks that sped up by more than the threshold
- **Within threshold** — benchmarks with changes within the acceptable range

The exit code is **1** if any regressions are detected, **0** otherwise.

### 🔄 Updating the Baseline

After intentional performance changes (e.g., trading speed for correctness),
update the baseline:

```bash
./benchmark_compare.sh --save-baseline
```

### 📈 Interpreting Results

Criterion reports the **median** change with confidence intervals. A result
like `[-2.3% -1.2% +0.1%]` means:

- The lower bound estimate is 2.3% faster
- The point estimate is 1.2% faster
- The upper bound estimate is 0.1% slower

The comparison script uses the **point estimate** (middle value) to determine
whether the threshold has been exceeded.

### 🎯 Single Suite Comparison

To focus on a specific benchmark:

```bash
./benchmark_compare.sh --bench synapse_counts
```

### 🛠️ Running Individual Benchmarks Manually

You can also run Criterion benchmarks directly:

```bash
# Run a single benchmark
cargo bench --bench synapse_counts

# Run with baseline comparison
cargo bench --bench synapse_counts -- --baseline saved

# Save a new baseline
cargo bench --bench synapse_counts -- --save-baseline saved
```

## 🔗 CI Integration

### Benchmark CI Script

The `scripts/benchmark-ci.sh` script provides CI-friendly benchmark verification
with two modes:

```bash
# Verify all benchmarks compile (fast, no GPU needed, default mode)
./scripts/benchmark-ci.sh --compile-only

# Compare against a saved baseline (requires GPU and baseline)
./scripts/benchmark-ci.sh --compare --threshold 10

# List all discovered benchmark suites
./scripts/benchmark-ci.sh --list
```

| Command | Description |
|---------|-------------|
| `./scripts/benchmark-ci.sh` | Verify benchmarks compile (default) |
| `./scripts/benchmark-ci.sh --compile-only` | Explicitly compile-only mode |
| `./scripts/benchmark-ci.sh --compare` | Compare against saved baseline |
| `./scripts/benchmark-ci.sh --threshold N` | Set regression threshold (default: 10%) |
| `./scripts/benchmark-ci.sh --list` | List discovered benchmark suites |
| `./scripts/benchmark-ci.sh --help` | Show usage information |

### CI Workflow

The recommended CI integration has two tiers:

**Tier 1 — Compilation check (every PR):**
Runs `cargo bench --no-run` for every benchmark target to verify benchmarks
compile. This catches compilation regressions without requiring a GPU or
spending time on actual benchmark execution. Runs on standard GitHub Actions
runners.

**Tier 2 — Regression detection (self-hosted runners):**
For environments with consistent hardware and GPU availability, the comparison
mode runs full benchmarks against a saved baseline and fails if any benchmark
regresses by more than the configured threshold.

### Regression Threshold

The default regression threshold is **10%**. This means a benchmark must slow
down by more than 10% compared to the baseline to be flagged as a regression.

The threshold can be configured via:
- Command-line flag: `--threshold N`
- Environment variable: `BENCHMARK_THRESHOLD=N`

A 10% default was chosen to balance sensitivity against noise from CI
environment variability. For self-hosted runners with stable hardware, consider
lowering this to 5%.

### Disk Space

Benchmark compilation requires release builds which consume significant disk
space. The CI script compiles benchmarks individually (per-suite) to limit
peak disk usage. The `target/criterion/` directory stores baseline data and
should be cached between runs for comparison mode.

### GPU Benchmarks

Most benchmarks require a GPU. On GitHub Actions runners (no GPU), the
compile-only mode verifies the code compiles correctly. Benchmarks that
cannot initialise a GPU are automatically skipped in comparison mode.

**Note:** Benchmarks produce machine-specific results. Meaningful regression
detection requires a self-hosted runner with consistent hardware. The scripts
handle GPU-less environments gracefully.
