## Summary

Replace the `once_cell` crate with standard library equivalents (`std::sync::OnceLock` and `std::sync::LazyLock`), removing it as a direct dependency. The project uses edition 2024 (Rust 1.85+) where these types are stabilised. Closes #671.

### Changes

- **`src/lib.rs`**: `once_cell::sync::OnceCell` replaced with `std::sync::OnceLock`
- **`src/debug.rs`**: `once_cell::sync::OnceCell` replaced with `std::sync::OnceLock`
- **`src/watchdog.rs`**: `once_cell::sync::Lazy` replaced with `std::sync::LazyLock`
- **`src/streaming.rs`**: `once_cell::sync::Lazy` replaced with `std::sync::LazyLock`
- **`src/analysis/cache/mod.rs`**: `once_cell::sync::OnceCell` replaced with `std::sync::OnceLock`. Since `OnceLock::get_or_try_init` is not yet stabilised, the cached value type was changed to `OnceLock<Result<..., String>>` with `get_or_init` to preserve the single-execution guarantee under contention.
- **`Cargo.toml`**: Removed `once_cell = "1.21"` direct dependency. `once_cell` remains only as a transitive dependency of other crates (e.g., `tracing`, `dashmap`, `arrow`).

## Evidence

This is a backend/internal change with no visual output. All 530+ unit tests and 97+ integration tests pass. The full quality gate (`./quality.sh`) passes cleanly including fmt, clippy, check, doc build, tests, and release build.

## Test Plan

- No new tests required — all existing tests exercise the migrated code paths
- The existing `record_cache_loads_once_per_neuron_under_contention` test validates that the `OnceLock`-based cache still guarantees single-execution under concurrent access
- All 530+ unit tests pass
- All integration tests pass
- `./quality.sh` passes cleanly
