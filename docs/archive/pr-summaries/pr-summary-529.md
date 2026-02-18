## Summary

Add GPU buffer transfer benchmarks to detect performance regressions in the critical path of moving data between CPU and GPU memory. Closes #529.

New benchmark suite `benches/gpu_buffer_transfers.rs` measures four transfer scenarios across batch sizes 64, 256, 1024, and 4096:

1. **CPU to GPU staging write** — buffer creation and upload cost
2. **GPU to CPU readback** — map_async + polling readback scaling
3. **End-to-end batch cycle** — full round-trip with multiple sub-batches
4. **Batch size scaling** — transfer overhead comparison across sizes

Skips gracefully on machines without GPU via `GpuAnalyzer::gpu_is_available()`.

## Evidence

This is a benchmark-only change (no runtime code modified). Benchmark results from Apple M4 Pro (Metal):

```
gpu_transfer_cpu_to_staging/samples/64     time: [26.431 ms 26.710 ms 26.986 ms]
gpu_transfer_cpu_to_staging/samples/256    time: [26.130 ms 26.467 ms 26.804 ms]
gpu_transfer_cpu_to_staging/samples/1024   time: [26.136 ms 26.438 ms 26.743 ms]
gpu_transfer_cpu_to_staging/samples/4096   time: [26.029 ms 26.383 ms 26.734 ms]

gpu_transfer_readback/samples/64           time: [26.221 ms 26.546 ms 26.863 ms]
gpu_transfer_readback/samples/256          time: [26.167 ms 26.499 ms 26.825 ms]
gpu_transfer_readback/samples/1024         time: [26.317 ms 26.650 ms 26.981 ms]
gpu_transfer_readback/samples/4096         time: [26.406 ms 26.751 ms 27.086 ms]

gpu_transfer_end_to_end/samples_x3_batches/64    time: [26.196 ms 26.528 ms 26.864 ms]
gpu_transfer_end_to_end/samples_x3_batches/256   time: [26.562 ms 26.884 ms 27.197 ms]
gpu_transfer_end_to_end/samples_x3_batches/1024  time: [26.375 ms 26.722 ms 27.070 ms]
gpu_transfer_end_to_end/samples_x3_batches/4096  time: [26.765 ms 27.100 ms 27.437 ms]

gpu_transfer_scaling/batch_size/64         time: [26.162 ms 26.465 ms 26.762 ms]
gpu_transfer_scaling/batch_size/256        time: [26.173 ms 26.495 ms 26.819 ms]
gpu_transfer_scaling/batch_size/1024       time: [26.483 ms 26.770 ms 27.057 ms]
gpu_transfer_scaling/batch_size/4096       time: [26.418 ms 26.731 ms 27.045 ms]
```

Results show flat scaling on unified memory (Apple Silicon), which is expected since CPU and GPU share the same physical memory. These benchmarks will be most valuable on discrete GPU systems where buffer transfers involve PCIe bus overhead.

## Test Plan

- `cargo bench --bench gpu_buffer_transfers` runs successfully with all 16 benchmark scenarios
- Benchmarks skip gracefully when no GPU is available
- `./quality.sh` passes cleanly (fmt, clippy, check, tests, release build)
