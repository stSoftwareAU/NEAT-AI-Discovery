## Summary

Enforce a hard entry-count limit on the LRU analysis cache with LRU eviction (Issue #1088). Closes #1088.

The `LruRecordCache` previously only evicted entries based on byte capacity. When `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` was set, it was only used by the streaming cache, not the LRU cache. This change adds a `max_entries` field to `LruRecordCache` that enforces the configured limit as a hard cap — no more than `max_entries` entries exist at any time. The least-recently-used entry is evicted before inserting a new one when either the byte limit or the entry-count limit is reached.

Changes to `src/analysis/cache/lru_cache.rs`:
- Added `max_entries` field read from `NEAT_AI_DISCOVERY_MAX_CACHED_BLOCKS` config
- Added `access_count` and `inserted_at` tracking on each cache entry
- Eviction now checks both byte capacity and entry-count limits
- Eviction logs at `info` level with the evicted entry's age and access count
- Cache hit/miss/eviction statistics logged at `debug` level (when verbose enabled)
- Added `with_loader()` constructor for testable cache instances
- Added `new_with_max_entries()` constructor for explicit entry-count limits
- Thread-safe under concurrent access via `parking_lot::RwLock`

## Evidence

This is a backend/library change with no UI. Verified via unit tests and the full quality gate (`./quality.sh`).

## Test Plan

Six unit tests added in `src/analysis/cache/lru_cache.rs`:
- `eviction_occurs_at_configured_limit` — verifies cache never exceeds `max_entries`
- `lru_ordering_evicts_oldest_accessed` — verifies LRU ordering (oldest-accessed entry evicted first)
- `cache_statistics_are_tracked` — verifies hit/miss/eviction counters
- `thread_safe_concurrent_access` — verifies correctness under concurrent access from 8 threads
- `max_entries_of_one_always_evicts` — edge case: single-entry cache always evicts
- `estimate_records_size_basic` — existing test (unchanged)
