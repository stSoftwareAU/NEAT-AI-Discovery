## Summary

Implements tiered/streaming Parquet loading for large files (Issue #215). The library now automatically selects the optimal loading strategy based on file size and available system memory:

- **PreloadAll**: For small files (estimated expanded < available_memory ÷ 4) - loads entire file upfront for fastest access
- **LruCache**: For medium files - per-neuron caching with LRU eviction, bounded memory usage
- **Streaming**: For very large files - block-based loading for lowest memory footprint

### Key Features

1. **Automatic Strategy Selection**: `select_loading_strategy()` analyses file size vs available memory
2. **Per-Neuron LRU Cache**: `LruRecordCache` with configurable capacity, smart eviction, and thread-safe concurrent access
3. **Tiered Record Cache**: `TieredRecordCache` wraps all strategies with unified API
4. **New API**: `RecordCache::new_tiered()` for automatic optimal strategy selection

### Expected Impact

- Support for larger Parquet files (2-3× current limits through smarter loading)
- Reduced peak memory usage for medium-sized files
- Better memory utilisation (only loads what's needed)
- Graceful degradation instead of hard failures

## Evidence

### Benchmark Results

Performance comparison across loading strategies (100 neurons, 1000 records each):

| Strategy | Sequential Access Time | Notes |
|----------|------------------------|-------|
| PreloadAll | **4.4 µs** | Fastest - all data in memory |
| LruCache (large capacity) | **5.9 µs** | Near-preload performance |
| LruCache (with eviction) | **155 ms** | Eviction overhead when capacity exceeded |
| Tiered (auto-select) | **4.3 µs** | Auto-selects best strategy |

Access patterns comparison (50 neurons, 500 records each, LRU with 20-neuron capacity):

| Pattern | Time | Notes |
|---------|------|-------|
| Sequential | 65 ms | Good cache locality |
| Random | 65 ms | Similar due to sufficient capacity |
| Hotspot (80/20) | 22 ms | Hot neurons stay cached |

Concurrent access scaling:

| Threads | Time | Notes |
|---------|------|-------|
| 1 | 10 ms | Baseline |
| 2 | 12 ms | Minimal contention |
| 4 | 17 ms | Good scaling |
| 8 | 33 ms | Lock contention visible |

### Strategy Selection Thresholds (8GB System)

| File Size | Expanded Size | Strategy Selected |
|-----------|---------------|-------------------|
| 100 MB | 300 MB | PreloadAll |
| 500 MB | 1.5 GB | LruCache |
| 3 GB | 9 GB | Streaming |

## Test Plan

### Unit Tests Added (in `src/analysis/cache.rs`)

- `loading_strategy_preload_for_small_files` - Verifies PreloadAll for small files
- `loading_strategy_lru_for_medium_files` - Verifies LruCache for medium files
- `loading_strategy_streaming_for_large_files` - Verifies Streaming for large files
- `lru_capacity_scales_with_memory` - Verifies LRU capacity = half of available memory
- `estimate_records_size_basic` - Verifies memory estimation for records

### Integration Tests Added (`tests/issue_215_tiered_parquet_loader.rs`)

- `loading_strategy_selection_preload_all_for_small_files` - Strategy selection logic
- `loading_strategy_selection_lru_cache_for_medium_files` - Strategy selection logic
- `loading_strategy_selection_streaming_for_large_files` - Strategy selection logic
- `lru_cache_capacity_scales_with_memory` - LRU capacity based on available memory
- `lru_record_cache_evicts_least_recently_used` - LRU eviction correctness
- `lru_record_cache_returns_correct_data_after_eviction` - Data correctness after eviction
- `lru_record_cache_respects_capacity_limits` - Memory bounds enforcement
- `tiered_cache_bounds_memory_usage` - Overall memory bounding
- `all_strategies_return_identical_data` - Data consistency across strategies
- `new_tiered_selects_appropriate_strategy` - API integration test
- `lru_record_cache_concurrent_access` - Thread safety verification
- `lru_record_cache_statistics_are_accurate` - Statistics tracking

### Benchmark Added (`benches/tiered_loading.rs`)

- `benchmark_tiered_loading_access` - Compares access time across strategies
- `benchmark_lru_cache_patterns` - Tests sequential/random/hotspot access patterns
- `benchmark_concurrent_access` - Tests concurrent access with 1-8 threads

## Files Changed

- `src/analysis/cache.rs` - Added `LoadingStrategy`, `LruRecordCache`, `TieredRecordCache`, `select_loading_strategy()`
- `tests/issue_215_tiered_parquet_loader.rs` - New integration test file
- `benches/tiered_loading.rs` - New benchmark file
- `Cargo.toml` - Added benchmark configuration
- `README.md` - Added documentation for tiered loading strategy
