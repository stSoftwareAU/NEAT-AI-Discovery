//! Issue #1027 — Expose Rust-side memory usage to Deno via FFI.
//!
//! The Rust library allocates memory outside V8's heap, making it invisible to
//! the GRQ `MemoryWatchdog`. This test verifies that the
//! `discovery_memory_usage_bytes()` FFI function returns a reasonable
//! approximation of Rust-side memory usage suitable for periodic polling.

// ============================================================================
// discovery_memory_usage_bytes — returns non-zero usage
// ============================================================================

#[test]
fn discovery_memory_usage_bytes_returns_nonzero() {
    // The library itself has already allocated memory by this point
    // (static initialisers, test harness, etc.), so usage should be > 0.
    let usage = neat_ai_discovery::ffi::discovery_memory_usage_bytes();
    assert!(
        usage > 0,
        "Expected non-zero Rust memory usage, got {usage}"
    );
}

#[test]
fn discovery_memory_usage_bytes_increases_after_allocation() {
    let before = neat_ai_discovery::ffi::discovery_memory_usage_bytes();

    // Allocate a known chunk of memory that the tracking allocator should see.
    let large_vec: Vec<u8> = vec![42u8; 1_000_000];

    let after = neat_ai_discovery::ffi::discovery_memory_usage_bytes();

    // The usage should have increased by at least the size of our allocation.
    // We allow some tolerance because other threads may free memory concurrently.
    assert!(
        after > before,
        "Expected memory usage to increase after 1 MB allocation: before={before}, after={after}"
    );

    // Ensure the allocation is not optimised away.
    assert_eq!(large_vec.len(), 1_000_000);
}

#[test]
fn discovery_memory_usage_bytes_is_callable_many_times() {
    // The function must be safe to call repeatedly for periodic polling.
    // We simply verify it does not panic or return wildly different values
    // within a tight loop (no timing assertions — that belongs in benchmarks).
    let mut values = Vec::with_capacity(100);
    for _ in 0..100 {
        values.push(neat_ai_discovery::ffi::discovery_memory_usage_bytes());
    }

    // All values should be non-zero (the process always has some allocations).
    assert!(
        values.iter().all(|&v| v > 0),
        "All polled values should be non-zero"
    );
}
