//! Platform-specific setup utilities.
//!
//! This module provides platform-specific environment configuration required
//! for GPU initialisation on different operating systems.
//!
//! Extracted from `implementation.rs` as part of Issue #267.

// =============================================================================
// Linux-Specific Setup
// =============================================================================
//
// # Process-environment mutation safety
//
// The helpers below mutate the process environment via `std::env::set_var`,
// which is `unsafe` in Rust 2024. `set_var` is a data race on the global
// `environ` table whenever *any* other thread concurrently reads (`getenv`,
// `env::var`) or writes it — undefined behaviour, not a recoverable panic.
// Concurrent reads are common because this crate ships as a `cdylib` linked
// into a host process, and C GPU/graphics libraries (Mesa, libEGL) read the
// environment during initialisation.
//
// The `Once` guard on each public entry point guarantees the write runs
// *exactly once*, but it does **not** make the process single-threaded, so it
// alone cannot uphold the soundness invariant. The real precondition every
// caller must satisfy is: **no other thread may access the process environment
// concurrently.** These functions are therefore documented as early-init entry
// points — call them during GPU initialisation, before spawning any thread
// that touches the environment.

/// Set an environment variable only if it is currently unset.
///
/// Returns `true` when the variable was written, `false` when it was already
/// present (and left untouched).
///
/// # Safety precondition
///
/// The caller must guarantee that no other thread accesses the process
/// environment concurrently — see the module-level note. This helper does not
/// and cannot enforce that invariant; it centralises the single `unsafe`
/// write so its justification lives in one place.
#[cfg(target_os = "linux")]
fn set_env_if_unset(key: &str, value: &str) -> bool {
    use std::env;

    if env::var(key).is_ok() {
        return false;
    }
    // SAFETY: Upheld by the caller — no other thread may access the process
    // environment concurrently (see the module-level note). This is the real
    // precondition for a sound `set_var`; the `Once` guard on the public
    // entry points only ensures the write happens once, not exclusively.
    unsafe { env::set_var(key, value) };
    true
}

/// Suppress Mesa GPU warnings on Linux if requested.
///
/// Set `NEAT_AI_DISCOVERY_QUIET_GPU=1` to enable suppression.
///
/// Mutates the process environment. Must be called during early GPU
/// initialisation, before any thread that reads the environment is spawned —
/// see the module-level safety note.
#[cfg(target_os = "linux")]
pub fn suppress_mesa_warnings_if_requested() {
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if crate::config::quiet_gpu() {
            apply_mesa_suppression();
        }
    });
}

/// Apply the Mesa/libEGL suppression environment variables.
///
/// Each variable is only written when currently unset. Carries the same
/// safety precondition as [`set_env_if_unset`].
#[cfg(target_os = "linux")]
fn apply_mesa_suppression() {
    // Suppress EGL debug messages (these cause "failed to open /dev/dri/..." warnings)
    set_env_if_unset("EGL_LOG_LEVEL", "fatal");
    // Suppress Mesa GLSL shader cache warnings
    set_env_if_unset("MESA_GLSL_CACHE_DISABLE", "true");
    // Suppress general Mesa debug output
    set_env_if_unset("MESA_DEBUG", "silent");
}

/// No-op on non-Linux platforms.
#[cfg(not(target_os = "linux"))]
pub fn suppress_mesa_warnings_if_requested() {
    // No-op on non-Linux platforms
}

/// Ensure `XDG_RUNTIME_DIR` is set on Linux (required by wgpu on Wayland).
///
/// This function uses `Once` so it runs its work exactly once even when called
/// from multiple threads; the first call sets the variable and subsequent calls
/// are no-ops.
///
/// Mutates the process environment. Must be called during early GPU
/// initialisation, before any thread that reads the environment is spawned —
/// the `Once` guard ensures single execution but not exclusivity against other
/// threads (see the module-level safety note).
#[cfg(target_os = "linux")]
pub fn ensure_xdg_runtime_dir() {
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(apply_xdg_runtime_dir);
}

/// Create a fallback runtime directory and point `XDG_RUNTIME_DIR` at it when
/// the variable is unset.
///
/// Carries the same safety precondition as [`set_env_if_unset`].
#[cfg(target_os = "linux")]
fn apply_xdg_runtime_dir() {
    use std::env;

    if env::var("XDG_RUNTIME_DIR").is_ok() {
        return;
    }
    // Create a temporary runtime directory if XDG_RUNTIME_DIR is not set
    if let Ok(temp_dir) = std::env::temp_dir().canonicalize() {
        let runtime_dir = temp_dir.join("neat-ai-discovery-runtime");
        if let Err(e) = std::fs::create_dir_all(&runtime_dir) {
            tracing::warn!(?runtime_dir, %e, "failed to create XDG_RUNTIME_DIR");
        } else {
            set_env_if_unset("XDG_RUNTIME_DIR", runtime_dir.to_string_lossy().as_ref());
        }
    }
}

/// No-op on non-Linux platforms.
#[cfg(not(target_os = "linux"))]
pub fn ensure_xdg_runtime_dir() {
    // No-op on non-Linux platforms
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: `suppress_mesa_warnings_if_requested` is a one-time
    /// environment variable setter guarded by `Once`. On non-Linux it is a
    /// no-op. There is no return value or queryable state — the only
    /// contract is that repeated calls do not panic.
    #[test]
    fn test_suppress_mesa_warnings_does_not_panic() {
        suppress_mesa_warnings_if_requested();
        suppress_mesa_warnings_if_requested();
    }

    /// Smoke test: `ensure_xdg_runtime_dir` is a one-time environment setup
    /// guarded by `Once`. On non-Linux it is a no-op. There is no return
    /// value or queryable state — the only contract is that repeated calls
    /// do not panic.
    #[test]
    fn test_ensure_xdg_runtime_dir_does_not_panic() {
        ensure_xdg_runtime_dir();
        ensure_xdg_runtime_dir();
    }

    /// `set_env_if_unset` writes the variable when it is absent and reports
    /// that it did so. Serial because it mutates the process environment.
    #[cfg(target_os = "linux")]
    #[test]
    #[serial_test::serial]
    fn test_set_env_if_unset_sets_when_absent() {
        use std::env;

        let key = "NEAT_AI_DISCOVERY_TEST_PLATFORM_UNSET";
        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { env::remove_var(key) };

        let wrote = set_env_if_unset(key, "applied");

        assert!(wrote, "should report a write when the variable was unset");
        assert_eq!(env::var(key).as_deref(), Ok("applied"));

        // SAFETY: serialised via #[serial]; clean up the test-only variable.
        unsafe { env::remove_var(key) };
    }

    /// `set_env_if_unset` leaves an existing value untouched and reports that
    /// it made no change. Serial because it mutates the process environment.
    #[cfg(target_os = "linux")]
    #[test]
    #[serial_test::serial]
    fn test_set_env_if_unset_preserves_existing() {
        use std::env;

        let key = "NEAT_AI_DISCOVERY_TEST_PLATFORM_PRESET";
        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { env::set_var(key, "original") };

        let wrote = set_env_if_unset(key, "replacement");

        assert!(
            !wrote,
            "should not report a write when the variable was set"
        );
        assert_eq!(
            env::var(key).as_deref(),
            Ok("original"),
            "existing value must be preserved"
        );

        // SAFETY: serialised via #[serial]; clean up the test-only variable.
        unsafe { env::remove_var(key) };
    }

    /// `apply_xdg_runtime_dir` sets `XDG_RUNTIME_DIR` to a usable directory
    /// when it is unset. Serial because it mutates the process environment.
    #[cfg(target_os = "linux")]
    #[test]
    #[serial_test::serial]
    fn test_apply_xdg_runtime_dir_sets_when_unset() {
        use std::env;

        let saved = env::var("XDG_RUNTIME_DIR").ok();
        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { env::remove_var("XDG_RUNTIME_DIR") };

        apply_xdg_runtime_dir();

        let set = env::var("XDG_RUNTIME_DIR").expect("XDG_RUNTIME_DIR should be set");
        assert!(
            std::path::Path::new(&set).is_dir(),
            "XDG_RUNTIME_DIR should point at an existing directory"
        );

        // SAFETY: serialised via #[serial]; restore any pre-existing value.
        unsafe {
            match saved {
                Some(v) => env::set_var("XDG_RUNTIME_DIR", v),
                None => env::remove_var("XDG_RUNTIME_DIR"),
            }
        }
    }
}
