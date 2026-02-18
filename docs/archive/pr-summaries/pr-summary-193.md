# PR Summary: Implement Streaming Parquet Loading with Prefetch (#193)

## Summary

This PR implements block-based streaming loading for Parquet files to address memory constraints when analysing large datasets. Instead of loading the entire Parquet file into memory upfront (which can cause 3× file size memory spikes), records are now loaded on-demand in configurable blocks with LRU eviction and predictive prefetching.

### Key Changes

1. **New `StreamingRecordCache`** (`src/analysis/streaming.rs`):
   - Block-based loading from Parquet files
   - LRU (Least Recently Used) eviction when block limit reached
   - Background prefetch thread for adjacent blocks
   - Thread-safe with `parking_lot::RwLock` for concurrent access
   - Automatic handling of neurons spanning multiple blocks

2. **Configuration via Environment Variables**:
   - `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS`: Maximum blocks in memory (default: adaptive)
   - `NEAT_AI_DISCOVERY_PREFETCH_DEPTH`: Blocks ahead to prefetch (default: 2)
   - `NEAT_AI_DISCOVERY_PRELOAD_ALL`: Disable streaming, use full preload (default: 0)
   - `NEAT_AI_DISCOVERY_BLOCK_SIZE`: Records per block (default: 10000, min: 10)

3. **Adaptive Memory Tiers**:
   - Low memory (<8GB): 10 blocks max
   - Standard (8-16GB): 50 blocks max
   - High (>16GB): 100 blocks max

### Benefits

- **Lower peak memory**: Only loaded blocks remain in memory
- **Faster time-to-first-result**: Analysis begins as first block loads
- **Predictable memory usage**: Configurable block limits prevent spikes
- **Better I/O parallelism**: Loading and analysis overlap via prefetch

## Evidence

Unable to generate screenshot: This is a CLI-only library with no visual interface.

### Performance Considerations

This implementation provides memory efficiency for large datasets. The streaming approach trades some throughput for reduced memory usage:

- **Small datasets (<500MB)**: Full preload may be faster due to fewer I/O operations
- **Large datasets (>500MB)**: Streaming enables analysis that would otherwise fail due to memory constraints
- **Memory-constrained systems**: Streaming prevents OOM errors and swap thrashing

To compare streaming vs preload performance for your specific workload:
```bash
# Full preload (original behaviour)
export NEAT_AI_DISCOVERY_PRELOAD_ALL=1

# Streaming mode (new behaviour)
unset NEAT_AI_DISCOVERY_PRELOAD_ALL
```

## Test Plan

### New Tests Added (`tests/issue_193_streaming_parquet.rs`):

1. `streaming_cache_provides_correct_records` - Verifies records are loaded correctly
2. `streaming_cache_respects_max_blocks` - Confirms block limit is enforced
3. `streaming_cache_evicts_least_recently_used` - Tests LRU eviction
4. `streaming_cache_prefetches_adjacent_blocks` - Validates prefetch mechanism
5. `streaming_cache_respects_env_config` - Tests environment variable configuration
6. `preload_all_disables_streaming` - Confirms `PRELOAD_ALL=1` works
7. `streaming_cache_handles_concurrent_access` - Thread safety test
8. `streaming_cache_stats_are_accurate` - Statistics tracking
9. `streaming_cache_bounds_memory_usage` - Memory bound enforcement
10. `new_adaptive_uses_streaming_for_memory_constraints` - Adaptive mode selection
11. `streaming_mode_matches_preloaded_data` - Data consistency between modes

### Unit Tests Added (`src/analysis/streaming.rs`):

1. `config_default_values` - Default configuration values
2. `is_streaming_enabled_default` - Default streaming behaviour
3. `streaming_cache_stats_initial` - Initial statistics state
4. `block_size_respects_minimum` - Minimum block size enforcement

### All Existing Tests Pass

The implementation maintains backward compatibility - all 383+ existing tests continue to pass.

## Documentation

Updated `README.md` with new "Streaming Parquet Loading" section documenting:
- Configuration options
- When to use streaming vs full preload
- Memory tier behaviour
