## Summary

Split the monolithic `src/analysis/cache.rs` (~1,300 lines) into a `cache/` directory with six focused sub-modules, each under ~300 lines. Closes #565.

### New structure

| File | Lines | Responsibility |
|------|-------|----------------|
| `cache/mod.rs` | ~280 | `RecordCache` struct, public API re-exports |
| `cache/loading_strategy.rs` | ~120 | `LoadingStrategy` enum, `select_loading_strategy()` |
| `cache/lru_cache.rs` | ~210 | `LruRecordCache`, `LruCacheStats`, eviction logic |
| `cache/compressed_cache.rs` | ~140 | `CompressedLruRecordCache` (LZ4 compression) |
| `cache/tiered_cache.rs` | ~130 | `TieredRecordCache`, auto strategy selection |
| `cache/serialisation.rs` | ~280 | Binary serialisation, `CompressedCacheEntry`, defensive deserialisation |

### Key decisions

- **No public API changes** — all types remain accessible via `analysis::cache::*`
- **All existing unit tests preserved** — moved into their respective sub-modules
- **Australian English** maintained throughout (`serialisation`, `analyse`, `behaviour`, etc.)

## Evidence

This is a pure refactoring with no UI or performance changes. Evidence of correctness:
- All 503+ library and integration tests pass without modification
- `quality.sh` passes cleanly (fmt, clippy, check, test, release build)

## Test Plan

- Existing unit tests moved to sub-modules: `loading_strategy::tests`, `lru_cache::tests`, `serialisation::tests`
- All integration tests pass unchanged (issue_186, issue_193, issue_215, issue_420, issue_481, issue_493)
- No test modifications required — public API is identical
