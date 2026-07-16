## Summary

Add three new Criterion benchmarks covering FFI JSON marshalling, cache eviction patterns, and parquet loading strategy comparison. These fill identified gaps in the benchmark suite where no benchmarks measured FFI serialisation overhead, cache behaviour under memory pressure, or comparative loading strategy performance. Closes #1040.

## Benchmarks Added

### `benches/ffi_marshalling.rs`
Measures JSON serialisation/deserialisation overhead at the FFI boundary for varying result set sizes (10, 100, 1000 candidates):
- **Serialisation**: `AnalyzeParallelOutput` to JSON string (11 us for 10 candidates, 923 us for 1000)
- **Deserialisation**: JSON string to `AnalyzeParallelInput` (4 us for 10 neurons, 369 us for 1000)
- **Round-trip**: Combined serialise + deserialise for full FFI call overhead

### `benches/cache_eviction.rs`
Exercises LRU and compressed cache throughput under memory pressure with different access patterns:
- **Sequential access**: Good locality — fewer evictions expected
- **Random access**: Poor locality — frequent evictions under pressure
- **Hot-spot (80/20)**: Skewed workload measuring LRU effectiveness
- **LRU vs Compressed**: Compares eviction overhead with and without LZ4 compression

### `benches/parquet_loading_comparison.rs`
Compares all loading strategies (pre-loaded, LRU, streaming, tiered auto) on the same datasets:
- **Small dataset** (50 neurons x 200 records): All strategies compared
- **Medium dataset** (200 neurons x 500 records): Reveals divergence under load
- **Scaling analysis**: Shows how strategies scale across dataset sizes (2.5K to 400K records)

## Evidence

All three benchmarks run successfully with `cargo bench --bench <name>`:
- `ffi_marshalling`: 9 benchmark cases, completes in ~20s
- `cache_eviction`: 13 benchmark cases, completes in ~50s
- `parquet_loading_comparison`: 14 benchmark cases, completes in ~60s
- `./quality.sh` passes cleanly with no regressions

## Test Plan

- All three benchmarks use Criterion framework consistent with existing 32 benchmarks
- Each benchmark runs within 60 seconds with default parameters
- `cargo bench` continues to pass with no regressions
- `./quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
