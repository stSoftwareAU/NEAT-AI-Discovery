## Summary

Implements GPU workgroup reduction for aggregation operations to reduce GPU→CPU data transfer for large sample counts. For sample sets with 10K+ samples, instead of transferring all per-sample contributions to the CPU for aggregation, the GPU now performs parallel tree reduction within workgroups, transferring only partial sums (one per workgroup).

### Changes

- **New shaders**: Added `helpful_reduce.wgsl` and `harmful_reduce.wgsl` that perform parallel tree reduction using workgroup shared memory
- **Reduction pipelines**: Added `build_helpful_reduce_pipeline` and `build_harmful_reduce_pipeline` to `GpuAnalyzer`
- **Modified batch evaluation**: Updated `evaluate_helpful_batch` and `evaluate_harmful_batch` to use reduction when sample count >= 10,000
- **New constants**: Added `GPU_REDUCTION_THRESHOLD` (10,000) and reduction shader references to `shaders.rs`
- **New struct**: Added `ReductionUniforms` to `samples.rs`

### Data Transfer Reduction

| Samples | HelpfulContribution Transfer | Reduction |
|---------|------------------------------|-----------|
| 10K     | 480KB → 1.9KB               | ~250×     |
| 50K     | 2.4MB → 9.6KB               | ~250×     |
| 100K    | 4.8MB → 18.8KB              | ~255×     |

For `HarmfulContribution` (16 bytes vs 48 bytes), the reduction ratio is similar.

## Evidence

Unable to generate screenshot: This is a GPU compute library with no visual interface.

### Performance Characteristics

The implementation uses a threshold of 10,000 samples to ensure the overhead of a second shader pass is worthwhile. Below this threshold, the original path (transferring all contributions) is used. The threshold can be tuned via the `GPU_REDUCTION_THRESHOLD` constant.

The reduction shader uses parallel tree reduction within each 256-thread workgroup:
1. Each thread loads one contribution into shared memory
2. Tree reduction halves active threads each iteration (128, 64, 32, 16, 8, 4, 2, 1)
3. First thread writes the workgroup's partial sum to output buffer
4. CPU sums only ~(N/256) partial sums instead of N contributions

## Test Plan

Added `tests/gpu_workgroup_reduction.rs` with 9 tests:

- `test_helpful_contribution_size` - Verifies HelpfulContribution is 48 bytes
- `test_harmful_contribution_size` - Verifies HarmfulContribution is 16 bytes
- `test_helpful_reduction_correctness` - Verifies reduction produces correct stats for 10K samples
- `test_harmful_reduction_correctness` - Verifies harmful reduction produces correct stats for 10K samples
- `test_reduction_threshold_small_samples` - Verifies small batches (< threshold) work correctly
- `test_reduction_edge_cases` - Tests empty, single sample, and zero activation cases
- `test_reduction_multiple_batches` - Tests multiple batches of varying sizes
- `test_partial_sums_buffer_sizing` - Unit test for workgroup count calculation
- `test_reduction_numerical_stability` - Tests with large values to verify no overflow

All existing tests continue to pass (333 library tests + 58 integration test files).
