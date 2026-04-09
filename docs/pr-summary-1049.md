## Summary

Handle missing parquet file gracefully in lazy cache reads. When a parquet file is deleted between cache construction and a cache-miss read (e.g., due to the host cleaning up temp directories while analysis is still active), the library now returns a clear "Parquet file removed" error instead of a generic I/O error. Bulk load methods (which use `.ok()` to skip errors) naturally return partial results when the file disappears mid-analysis. Closes #1049.

## Changes

- **`src/parquet_format/reader.rs`**: Added `open_parquet_file()` helper that distinguishes `NotFound` errors (file externally deleted) from other I/O errors (corruption, permissions). All three public read functions (`read_all_records_grouped_by_neuron_with_deadline`, `read_records_from_parquet`, `read_records_from_parquet_with_limit`) now use this helper.
- **`tests/cache_missing_parquet.rs`**: New integration test file with 5 tests covering cache behaviour when the parquet file is deleted after cache construction.

## Evidence

The fix flows through the existing call chain: `LruRecordCache::load_neuron_records()` and `CompressedLruRecordCache::load_neuron_records()` both call `read_records_from_parquet()`, which now uses `open_parquet_file()`. No changes needed in the cache modules themselves.

Partial results already work because `RecordCache::load_records_for_uuids()` and similar bulk methods use `filter_map(|uuid| self.get(uuid).ok())`, silently skipping failed loads.

## Test Plan

- `lru_cache_handles_missing_parquet_without_panic` — LRU cache returns clear error on cache miss after file deletion
- `compressed_cache_handles_missing_parquet_without_panic` — Compressed cache returns clear error on cache miss after file deletion
- `record_cache_lazy_handles_missing_parquet_without_panic` — RecordCache with lazy loader returns error after file deletion
- `read_records_from_parquet_returns_file_removed_error` — Low-level reader returns "file removed" error
- `lru_cache_returns_cached_data_after_file_deletion` — Already-cached data remains accessible after file deletion
