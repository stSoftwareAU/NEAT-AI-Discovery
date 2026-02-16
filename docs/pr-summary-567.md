# PR Summary: Optimise GPU Compute Shaders with Workgroup-Level Improvements

Closes #567

## Overview

Optimises GPU compute shaders using workgroup shared memory and parallel reduction
patterns to reduce GPU→CPU data transfer and improve memory access efficiency.

## Changes

### New GPU Reduction Shaders

- **`src/shaders/activation_reduce.wgsl`** — Parallel tree reduction for ActivationOutput
  aggregation. Reduces GPU→CPU transfer by ~255× for large sample counts
  (100K samples: 2.8MB → 10.9KB).

- **`src/shaders/relu_reduce.wgsl`** — Parallel tree reduction for ReluContribution
  aggregation. Available for future use when ReLU evaluation pipeline is refactored
  to separate aggregation from per-sample weight validation.

### Bias Shader Workgroup Shared Memory Optimisation

- **`src/shaders/bias.wgsl`** — Replaced per-thread global memory reads with cooperative
  tiled sample loading into `var<workgroup>` shared memory. Each workgroup loads tiles
  of 256 samples cooperatively, then all threads in the workgroup read from shared memory
  instead of global memory. This dramatically reduces global memory bandwidth pressure
  for the bias grid search (which evaluates every sample for every bias candidate).

### Activation Evaluation GPU Reduction Integration

- **`src/analysis/gpu/activation_evaluation.rs`** — Both `evaluate_activation_gpu` and
  `evaluate_activations_batched_gpu` now use the activation reduction shader for sample
  counts ≥ `GPU_REDUCTION_THRESHOLD` (10,000). For smaller counts, falls back to direct
  CPU aggregation.

- **`src/analysis/gpu/analyzer.rs`** — Added `activation_reduce_layout` and
  `activation_reduce_pipeline` fields to `GpuAnalyzer` struct.

- **`src/analysis/gpu/shaders.rs`** — Added `RELU_REDUCE_SHADER` and
  `ACTIVATION_REDUCE_SHADER` constants with comprehensive test coverage.

### Benchmark Suite

- **`benches/gpu_shader_workgroup.rs`** — Criterion benchmark measuring helpful, harmful,
  ReLU, activation, and batched activation GPU evaluation throughput across sample sizes
  (1K, 5K, 10K, 50K, 100K).

### Design Decisions

1. **ReLU reduction not wired into pipeline**: The `evaluate_relu_gpu` method
   reconstructs per-sample `(activation, error)` pairs into `ReluStats.samples` for
   weight validation (`improved_count` calculation). Pure reduction would lose this
   per-element data. The shader is created for future use when the pipeline is
   refactored.

2. **Matching shader unchanged**: The matching shader performs binary search joins
   between target and source records — a fundamentally different access pattern that
   doesn't benefit from reduction.

3. **Threshold-based reduction**: Uses `GPU_REDUCTION_THRESHOLD` (10,000 samples) to
   avoid reduction overhead for small sample counts where CPU aggregation is faster.

## Benchmark Results

Activation evaluation shows 3–6% throughput improvement at medium sample sizes
(5K–50K). The bias shader shared memory optimisation reduces global memory bandwidth
pressure but is not directly benchmarked (bias evaluation is invoked during full
discovery, not isolated in the benchmark suite).

## Quality

- `quality.sh` passes (fmt, clippy, check, test, release build)
- All 509 unit tests pass
- All integration tests pass
- No new warnings
