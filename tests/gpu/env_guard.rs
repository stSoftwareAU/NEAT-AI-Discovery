//! Save/restore guard for process-environment variables in the GPU tests.
//!
//! Tests that call the `unsafe` GPU environment setters (or depend on the
//! variables they write) must be `#[serial_test::serial]` and hold one of
//! these guards, so a variable the process-wide `Once` writes does not leak
//! into the tests that run after them (Issue #2538).

use std::ffi::OsString;

/// Records a variable's current value on creation and restores it on drop.
pub struct EnvGuard {
    key: &'static str,
    prior: Option<OsString>,
}

impl EnvGuard {
    /// Captures `key` without modifying it.
    pub fn capture(key: &'static str) -> Self {
        Self {
            key,
            prior: std::env::var_os(key),
        }
    }

    /// Captures every variable the GPU environment setup can write.
    pub fn capture_gpu_setup_vars() -> Vec<Self> {
        [
            "XDG_RUNTIME_DIR",
            "EGL_LOG_LEVEL",
            "MESA_GLSL_CACHE_DISABLE",
            "MESA_DEBUG",
        ]
        .into_iter()
        .map(Self::capture)
        .collect()
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: only used by `#[serial_test::serial]` tests, which are
        // serialised with every env-mutating or env-dependent test in the binary.
        unsafe {
            match &self.prior {
                Some(value) => std::env::set_var(self.key, value),
                None => std::env::remove_var(self.key),
            }
        }
    }
}
