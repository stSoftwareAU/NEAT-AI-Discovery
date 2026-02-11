## Summary

Replaced 8 `unwrap()` calls in `src/analysis/cache.rs` with proper `Result`-based error handling during binary deserialisation of cached records. Previously, corrupted or truncated cache files would cause panics; now they return descriptive `Err` values that allow callers to fall back gracefully to a fresh parquet read.

### Changes

- **`deserialise_records()`** — Changed return type from `Vec<DiscoverRecord>` to `Result<Vec<DiscoverRecord>>`. Each of the 8 `unwrap()` calls replaced with bounds checks + `bail!()` / `.map_err()` that include the byte offset for diagnostics.
- **`CompressedCacheEntry::decompress()`** — Changed return type from `Vec<DiscoverRecord>` to `Result<Vec<DiscoverRecord>>`. LZ4 decompression failure now returns `Err` instead of panicking via `expect()`.
- **`CompressedLruRecordCache::get()`** — Updated to propagate the new `Result` from `decompress()` using `?`.

### What did NOT change

- No public API changes — both modified functions are private (`fn`, not `pub fn`).
- The serialisation format is unchanged.
- No performance regression — the added bounds checks are negligible compared to I/O.

## Evidence

This is a backend-only change with no UI components. Correctness is verified by the test suite below.

## Test Plan

Added 9 new unit tests in `src/analysis/cache.rs`:

| Test | Verifies |
|------|----------|
| `deserialise_records_round_trip` | Valid data serialises and deserialises correctly |
| `deserialise_records_empty_data` | Empty input returns `Ok(vec![])` |
| `deserialise_records_truncated_at_obs_index` | Truncation at obs_index returns `Err` with offset info |
| `deserialise_records_truncated_at_uuid` | Truncation at UUID field returns `Err` |
| `deserialise_records_truncated_at_value` | Truncation at value field returns `Err` |
| `deserialise_records_truncated_at_activation` | Truncation at activation field returns `Err` |
| `deserialise_records_truncated_at_errors` | Truncation at errors array returns `Err` |
| `decompress_returns_error_on_corrupted_lz4` | Corrupted LZ4 data returns `Err` |
| `decompress_valid_data_round_trip` | Valid compress/decompress round-trip works |

All 14 cache tests pass. Full `quality.sh` gate passes cleanly.
