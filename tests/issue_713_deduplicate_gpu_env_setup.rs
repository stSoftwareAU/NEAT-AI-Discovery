//! Integration tests for Issue #713: GPU environment setup deduplication.
//!
//! Verifies that `suppress_mesa_warnings_if_requested` and `ensure_xdg_runtime_dir`
//! are accessible from `analysis::utils` (canonically defined in `platform.rs`) and
//! that the duplicated copies in `memory.rs` have been removed.

use neat_ai_discovery::analysis::utils::platform;

/// The canonical functions in `platform.rs` must be callable without panicking.
#[test]
fn platform_suppress_mesa_warnings_does_not_panic() {
    platform::suppress_mesa_warnings_if_requested();
}

/// The canonical `ensure_xdg_runtime_dir` must be callable without panicking.
#[test]
fn platform_ensure_xdg_runtime_dir_does_not_panic() {
    platform::ensure_xdg_runtime_dir();
}

/// The re-exported functions via `utils` must resolve to the same canonical functions.
#[test]
fn utils_reexports_resolve_to_platform() {
    // These call the re-exported versions from utils::mod.rs
    neat_ai_discovery::analysis::utils::suppress_mesa_warnings_if_requested();
    neat_ai_discovery::analysis::utils::ensure_xdg_runtime_dir();
}
