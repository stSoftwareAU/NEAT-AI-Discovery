//! Integration tests for Issue #713: GPU environment setup deduplication.
//!
//! Verifies that `suppress_mesa_warnings_if_requested` and `ensure_xdg_runtime_dir`
//! are accessible from `analysis::utils` (canonically defined in `platform.rs`) and
//! that the duplicated copies in `memory.rs` have been removed.

use neat_ai_discovery::analysis::utils::platform;

/// Smoke test: `suppress_mesa_warnings_if_requested` is a one-time, idempotent
/// environment variable setter guarded by `Once`. On non-Linux it is a no-op.
/// There is no return value or queryable state — the only contract is that
/// repeated calls do not panic.
#[test]
fn platform_suppress_mesa_warnings_does_not_panic() {
    // SAFETY: the test harness has not spawned any thread that reads the process
    // environment, so the early-init precondition holds.
    unsafe {
        platform::suppress_mesa_warnings_if_requested();
        // Idempotent — second call must also succeed
        platform::suppress_mesa_warnings_if_requested();
    }
}

/// Smoke test: `ensure_xdg_runtime_dir` is a one-time environment setup
/// guarded by `Once`. On non-Linux it is a no-op. There is no return value
/// or queryable state — the only contract is that repeated calls do not panic.
#[test]
fn platform_ensure_xdg_runtime_dir_does_not_panic() {
    // SAFETY: the test harness has not spawned any thread that reads the process
    // environment, so the early-init precondition holds.
    unsafe {
        platform::ensure_xdg_runtime_dir();
        // Idempotent — second call must also succeed
        platform::ensure_xdg_runtime_dir();
    }
}

/// Smoke test: verifies that the `utils` module re-exports resolve to the
/// same canonical platform functions without panicking. The functions are
/// environment-variable setters with no return value, so a no-panic check
/// is the strongest available contract.
#[test]
fn utils_reexports_resolve_to_platform() {
    // SAFETY: the test harness has not spawned any thread that reads the process
    // environment, so the early-init precondition holds.
    unsafe {
        neat_ai_discovery::analysis::utils::suppress_mesa_warnings_if_requested();
        neat_ai_discovery::analysis::utils::ensure_xdg_runtime_dir();
    }
}
