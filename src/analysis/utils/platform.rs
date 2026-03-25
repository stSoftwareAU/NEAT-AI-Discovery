//! Platform-specific setup utilities.
//!
//! This module provides platform-specific environment configuration required
//! for GPU initialisation on different operating systems.
//!
//! Extracted from `implementation.rs` as part of Issue #267.

// =============================================================================
// Linux-Specific Setup
// =============================================================================

/// Suppress Mesa GPU warnings on Linux if requested.
///
/// Set `NEAT_AI_DISCOVERY_QUIET_GPU=1` to enable suppression.
#[cfg(target_os = "linux")]
pub fn suppress_mesa_warnings_if_requested() {
    use std::env;
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if crate::config::quiet_gpu() {
            // Suppress EGL debug messages (these cause "failed to open /dev/dri/..." warnings)
            if env::var("EGL_LOG_LEVEL").is_err() {
                // SAFETY: single-threaded at this point (Once guard) and before GPU init
                unsafe { env::set_var("EGL_LOG_LEVEL", "fatal") };
            }

            // Suppress Mesa GLSL shader cache warnings
            if env::var("MESA_GLSL_CACHE_DISABLE").is_err() {
                // SAFETY: single-threaded at this point (Once guard) and before GPU init
                unsafe { env::set_var("MESA_GLSL_CACHE_DISABLE", "true") };
            }

            // Suppress general Mesa debug output
            if env::var("MESA_DEBUG").is_err() {
                // SAFETY: single-threaded at this point (Once guard) and before GPU init
                unsafe { env::set_var("MESA_DEBUG", "silent") };
            }
        }
    });
}

/// No-op on non-Linux platforms.
#[cfg(not(target_os = "linux"))]
pub fn suppress_mesa_warnings_if_requested() {
    // No-op on non-Linux platforms
}

/// Ensure `XDG_RUNTIME_DIR` is set on Linux (required by wgpu on Wayland).
///
/// This function uses `Once` for thread-safe one-time initialisation. It's safe to
/// call from multiple threads concurrently - only the first call will set the
/// environment variable, and subsequent calls are no-ops.
///
/// Must be called before any GPU initialisation (wgpu Instance creation).
#[cfg(target_os = "linux")]
pub fn ensure_xdg_runtime_dir() {
    use std::env;
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if env::var("XDG_RUNTIME_DIR").is_err() {
            // Create a temporary runtime directory if XDG_RUNTIME_DIR is not set
            if let Ok(temp_dir) = std::env::temp_dir().canonicalize() {
                let runtime_dir = temp_dir.join("neat-ai-discovery-runtime");
                if let Err(e) = std::fs::create_dir_all(&runtime_dir) {
                    tracing::warn!(?runtime_dir, %e, "failed to create XDG_RUNTIME_DIR");
                } else {
                    // SAFETY: Inside Once::call_once, so guaranteed single-threaded execution.
                    // Called before any GPU init.
                    unsafe {
                        env::set_var("XDG_RUNTIME_DIR", runtime_dir.to_string_lossy().as_ref());
                    }
                }
            }
        }
    });
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
}
