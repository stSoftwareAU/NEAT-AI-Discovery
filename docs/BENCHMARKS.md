# 📊 Benchmark Regression Tracking

This document describes the benchmark comparison workflow for detecting
performance regressions using Criterion.

## 🔍 Overview

The project includes 22 Criterion benchmark suites in `benches/`. The
`benchmark_compare.sh` script automates baseline saving and regression
detection by leveraging Criterion's built-in comparison features.

Baselines are machine-specific — each developer or CI runner saves their own
baseline locally in `target/criterion/`.

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
| `async_pipeline` | CPU/GPU overlap pipeline throughput |
| `batched_activation` | Batched vs sequential activation evaluation |
| `bfs_allocation` | Visited-set allocation strategies in BFS |
| `cache_locality` | Cache access patterns |
| `clone_reduction` | Clone elimination in hot paths |
| `error_collection` | Lock-free vs mutex-based error collection |
| `gpu_buffer_transfers` | GPU buffer transfer overhead |
| `gpu_shader_workgroup` | GPU shader workgroup optimisation |
| `memory_streaming` | Memory streaming performance |
| `neuron_interning` | Neuron UUID interning |
| `parallel_discovery` | Parallel discovery throughput |
| `sample_locality` | Sample data locality |
| `squash_normalisation` | Pre-normalised squash string lookup |
| `synapse_counts` | Synapse count pre-computation |
| `synapse_lookup` | Zero-allocation synapse existence and weight lookups |
| `synapse_preparation` | String cloning cost in synapse preparation |
| `tiered_loading` | Tiered cache loading strategies |
| `topology_cache` | Pre-computed topology cache vs repeated map building |
| `upsert_candidate` | Candidate upsert operations |
| `uuid_hashing` | Deterministic UUID generation for candidates |
| `weight_coherence_cache` | Weight coherence detection with topology cache |
| `zero_copy_buffer` | Zero-copy buffer performance |

**Note:** Most benchmarks require a GPU. Suites that cannot initialise a GPU
are automatically skipped.

## 🔄 Workflow

### 🏁 Initial Setup

Save a baseline on your machine before making performance-sensitive changes:

```bash
./benchmark_compare.sh --save-baseline
```

This runs all 22 benchmark suites and stores results in `target/criterion/`.

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
