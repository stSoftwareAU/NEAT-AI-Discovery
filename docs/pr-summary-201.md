## Summary

Issue #201: Implemented batched activation function evaluation to reduce GPU round-trips during neuron analysis.

### Changes

1. **New `GpuEvaluator` trait method**: Added `evaluate_activations_batched()` to evaluate multiple activation function configurations in a single GPU command buffer submission.

2. **GPU implementation** (`src/analysis/gpu/analyzer.rs`):
   - `evaluate_activations_batched_gpu()` uploads sample data once, creates all compute passes in a single command buffer, and maps all result buffers together
   - Eliminates per-config GPU round-trip overhead

3. **GpuWorkQueue support** (`src/analysis/gpu/queue.rs`):
   - Added `ActivationBatchEval` work request variant
   - Thread-safe batched evaluation via the work queue

4. **Neuron analysis integration** (`src/analysis/neuron.rs`):
   - Replaced sequential `for spec in ACTIVATION_SPECS.iter()` loop with single call to `evaluate_all_activation_specs_batched()`
   - Maintains identical results while dramatically reducing GPU calls

5. **Helper function** (`src/analysis/synapse.rs`):
   - `evaluate_all_activation_specs_batched()` collects all (activation_type, orientation, scale) combinations for all 15 activation specs
   - Falls back to sequential evaluation if batched fails

## Evidence

### Performance Results

Benchmark comparing batched vs sequential evaluation on Apple M4 Pro:

```
Issue #201 Performance Comparison:
  Sample count: 500
  Activation configs: 256 (15 specs × orientations × scales)

Results (average of 5 iterations):
  Sequential: 7,054,620µs (256 GPU calls)
  Batched:    69,103µs (1 GPU call)
  Speedup:    102.09x
  Time reduction: 99.0%
```

**Key findings:**
- **102x speedup** (far exceeding the expected 10-20% improvement)
- Reduced from 256 GPU round-trips to just 1
- Same sample data uploaded only once, shared across all activation configs

### Why the dramatic improvement?

The original implementation made separate GPU calls for each (activation_type, orientation, scale) combination:
- 15 activation specs × 2 orientations × 8-9 scales ≈ 256 GPU calls per (source, target) pair
- Each call involved: buffer creation → command submission → buffer mapping → result readback

The batched implementation:
- Creates all buffers upfront
- Submits all compute passes in a single command buffer
- Maps all staging buffers concurrently
- Processes all results in one pass

## Test Plan

### Unit Tests (`tests/issue_201_batched_activation.rs`)
- `test_batched_vs_sequential_results_identical` - Verifies batched produces identical results to sequential
- `test_all_activation_specs_in_batched_mode` - Verifies all 15 activation specs work correctly
- `test_batched_evaluation_empty_samples` - Edge case: empty sample input
- `test_batched_evaluation_single_config` - Edge case: single activation config
- `test_batched_evaluation_empty_configs` - Edge case: empty config input
- `test_batched_evaluation_edge_case_samples` - Edge case: NaN/Infinity values

### Performance Test (`tests/issue_201_benchmark.rs`)
- `test_batched_vs_sequential_performance` - Measures and reports speedup

### Benchmark (`benches/batched_activation.rs`)
- Criterion benchmark for accurate performance measurement

### All Existing Tests Pass
- `./quality.sh` passes with all 372 unit tests and 60+ integration tests
