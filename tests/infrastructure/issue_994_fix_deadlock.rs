//! Integration tests for issue #994: fix process hang after discovery completes.
//!
//! The root cause was that `init_debug_handlers()` spawned two background threads
//! (deadlock-detector and signal-handler) with no shutdown mechanism. After
//! discovery work finished, these threads kept running and could prevent the
//! host process from exiting cleanly.
//!
//! These tests verify:
//! - `shutdown_debug_handlers()` completes promptly (no hang)
//! - `shutdown_debug_handlers()` is idempotent (safe to call multiple times)
//! - `cleanup_discovery_lib()` FFI function works correctly
//! - Debug threads stop after shutdown is requested

use std::time::{Duration, Instant};

/// Verify that `shutdown_debug_handlers()` completes within a reasonable time
/// and does not hang the process (the core bug from issue #994).
#[test]
fn shutdown_debug_handlers_completes_promptly() {
    // Initialise first (idempotent, safe in multi-test runs).
    neat_ai_discovery::debug::init_debug_handlers();

    let start = Instant::now();
    neat_ai_discovery::debug::shutdown_debug_handlers();
    let elapsed = start.elapsed();

    // Shutdown should complete well within 15 seconds. The deadlock-detector
    // sleeps in 5-second intervals, and the signal-handler is unblocked by
    // Handle::close(). With the join timeout set to 10s, this is generous.
    assert!(
        elapsed < Duration::from_secs(15),
        "shutdown_debug_handlers() took {elapsed:?} — may be hanging"
    );
}

/// Calling shutdown multiple times must not panic or hang.
#[test]
fn shutdown_debug_handlers_is_idempotent() {
    neat_ai_discovery::debug::init_debug_handlers();

    for _ in 0..3 {
        let start = Instant::now();
        neat_ai_discovery::debug::shutdown_debug_handlers();
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(15),
            "repeated shutdown took {elapsed:?}"
        );
    }
}

/// The FFI cleanup function must work correctly and not panic.
#[test]
fn cleanup_discovery_lib_ffi_does_not_hang() {
    // Ensure the library is initialised (this also starts debug handlers).
    let _ = neat_ai_discovery::get_library_version_internal();

    let start = Instant::now();
    neat_ai_discovery::ffi::cleanup_discovery_lib();
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_secs(15),
        "cleanup_discovery_lib() took {elapsed:?} — may be hanging"
    );
}

/// Calling cleanup before init should not panic.
#[test]
fn cleanup_before_init_does_not_panic() {
    neat_ai_discovery::debug::shutdown_debug_handlers();
    // If we reach here, the test passes.
}
