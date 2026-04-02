## Summary

Refactored GPU queue submission functions to accept borrowed slices (`&[T]`) instead
of owned `Vec<T>`, moving the `to_vec()` copy inside the submission functions where
it is needed for cross-thread channel transfer. This eliminates unnecessary `to_vec()`
calls at the `GpuEvaluator` trait implementation boundary. Closes #977.

**Key changes:**
- `evaluate_relu_gpu`, `evaluate_activation_gpu`, and `evaluate_activations_batched_gpu`
  on `GpuWorkQueue` now accept `&[HelpfulSample]` / `&[(u32, f32, f32)]` instead of
  owned Vecs
- The `GpuEvaluator for GpuWorkQueue` trait impl no longer calls `to_vec()` — it
  passes slices directly to the submission functions
- The submission functions perform `to_vec()` internally only when constructing the
  `GpuWorkRequest` for cross-thread channel send (this copy is mandatory)

## Evidence

Benchmark results (`cargo bench --bench queue_submission_copies`) show the `to_vec()`
copy overhead is negligible compared to GPU execution time:

| Metric | 64 samples | 256 samples | 1024 samples | 4096 samples |
|---|---|---|---|---|
| GPU ReLU eval | 29.1ms | 28.5ms | 29.5ms | 29.5ms |
| `to_vec()` copy | 32ns | 88ns | 375ns | 1.9us |
| Copy as % of GPU | 0.0001% | 0.0003% | 0.001% | 0.006% |

The refactoring is an API improvement that properly encapsulates the copy as an
implementation detail of the submission functions. The actual copy cannot be eliminated
because data must be owned to cross the thread boundary via the crossbeam channel.

## Test Plan

- All 158 existing integration tests pass
- All existing unit tests in `execution.rs` pass (send_error_to_request, request_label)
- New benchmark `queue_submission_copies` added to measure copy overhead vs GPU time
- Full `./quality.sh` passes (clippy, fmt, tests, doc build, release build)
