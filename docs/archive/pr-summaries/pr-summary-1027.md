## Summary

Add `discovery_memory_usage_bytes()` FFI function that reports current Rust-side
heap allocation in bytes, enabling the Deno-side MemoryWatchdog to track total
process memory (V8 + Rust) and prevent OOM kills. Closes #1027.

The implementation uses the `cap` crate as a global tracking allocator wrapper
around `std::alloc::System`. It maintains a single atomic counter that is
incremented/decremented on every allocation/deallocation, making the query
function essentially free for periodic polling (every 5-30 seconds).

## Changes

- **`Cargo.toml`**: Added `cap` dependency (MIT OR Apache-2.0) with `stats` feature
- **`src/lib.rs`**: Set `cap::Cap<System>` as the `#[global_allocator]`
- **`src/ffi/utilities.rs`**: Added `discovery_memory_usage_bytes()` FFI export
- **`docs/FFI_API.md`**: Documented the new symbol with usage example
- **`AGENTS.md`**: Updated source layout comment

## Evidence

The new FFI symbol is verified exported from the release build:
```
nm -gU target/release/libneat_ai_discovery.dylib | grep discovery_memory_usage_bytes
00000000002513a4 T _discovery_memory_usage_bytes
```

All 3 new tests pass, and the full quality gate (`./quality.sh`) passes cleanly
with 171 tests (including the 3 new ones).

## Test Plan

- `tests/ffi/issue_1027_memory_usage_ffi.rs`:
  - `discovery_memory_usage_bytes_returns_nonzero` — verifies the function returns > 0
  - `discovery_memory_usage_bytes_increases_after_allocation` — verifies usage increases after a 1 MB allocation
  - `discovery_memory_usage_bytes_is_callable_many_times` — verifies safe repeated polling
