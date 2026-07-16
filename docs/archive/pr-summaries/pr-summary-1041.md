## Summary

Add two new documentation guides for common workflows: streaming API usage and cache tier tuning. Closes #1041.

**`docs/STREAMING_GUIDE.md`** — Step-by-step guide for the streaming recording workflow (`start_discovery_session` → `append_discovery_records` → `finish_discovery_session`) with TypeScript/Deno code examples, error handling and recovery patterns, and best practices for batch sizing and session lifecycle.

**`docs/CACHE_TUNING.md`** — Overview of the three cache tiers (PreloadAll, LRU Cache, Streaming), how `max_analysis_memory_mb` interacts with tier selection, diagnostic steps for cache-related performance issues, and example configurations for common deployment scenarios.

Cross-references added from `README.md` (Additional Documentation table) and `docs/FFI_API.md` (streaming API section and memory budget section).

Both documents use Australian English spelling throughout.

## Evidence

- All 8 new tests in `tests/doc_cache_tuning_examples.rs` pass — these call the real `select_loading_strategy` function with the exact scenarios documented in the tier selection table, ensuring documentation stays in sync with the implementation.
- `./quality.sh` passes cleanly (fmt, clippy, all tests, doc build, release build).

## Test Plan

- Added `tests/doc_cache_tuning_examples.rs` with 8 tests validating documented cache tier selection examples against the real `select_loading_strategy` function:
  - `doc_example_small_file_preload` — 10 MB file, 8 GB RAM → PreloadAll
  - `doc_example_medium_file_preload` — 500 MB file, 8 GB RAM → PreloadAll
  - `doc_example_1gb_file_lru` — 1 GB file, 8 GB RAM → LRU Cache
  - `doc_example_large_file_streaming` — 4 GB file, 8 GB RAM → Streaming
  - `doc_example_constrained_ram_lru` — 100 MB file, 1 GB RAM → LRU Cache
  - `doc_example_constrained_ram_streaming` — 500 MB file, 1 GB RAM → Streaming
  - `decompression_ratio_boundary_preload` — boundary test at PreloadAll threshold
  - `decompression_ratio_boundary_lru` — boundary test at LRU threshold
