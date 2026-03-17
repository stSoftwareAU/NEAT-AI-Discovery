# 📊 Benchmark Regression Tracking

This document describes the benchmark comparison workflow for detecting
performance regressions using Criterion.

## 🔍 Overview

The project includes 12 Criterion benchmark suites in `benches/`. The
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
| `batched_activation` | Batched vs sequential activation evaluation |
| `cache_locality` | Cache access patterns |
| `clone_reduction` | Clone elimination in hot paths |
| `gpu_buffer_transfers` | GPU buffer transfer overhead |
| `memory_streaming` | Memory streaming performance |
| `neuron_interning` | Neuron UUID interning |
| `parallel_discovery` | Parallel discovery throughput |
| `sample_locality` | Sample data locality |
| `synapse_counts` | Synapse count pre-computation |
| `tiered_loading` | Tiered cache loading strategies |
| `upsert_candidate` | Candidate upsert operations |
| `zero_copy_buffer` | Zero-copy buffer performance |

**Note:** Most benchmarks require a GPU. Suites that cannot initialise a GPU
are automatically skipped.

## 🔄 Workflow

### 🏁 Initial Setup

Save a baseline on your machine before making performance-sensitive changes:

```bash
./benchmark_compare.sh --save-baseline
```

This runs all 12 benchmark suites and stores results in `target/criterion/`.

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

The `benchmark_compare.sh` script can be integrated into CI by:

1. Saving a baseline on a known-good commit
2. Running the comparison on each pull request
3. Failing the build if regressions exceed the threshold

**Note:** Benchmarks require a GPU and produce machine-specific results.
CI integration requires a self-hosted runner with consistent hardware for
meaningful comparisons. The script handles GPU-less environments gracefully
by skipping benchmarks that cannot initialise.
