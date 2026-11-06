//! Save/restore guard for process-environment variables in the GPU tests.
//!
//! Tests that call the `unsafe` GPU environment setters must use this guard to
//! restore their writes. Tests that read these variables during GPU
//! initialisation share their serial lock so they cannot overlap those writes
//! (Issue #2538).

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

#[cfg(test)]
mod tests {
    use super::EnvGuard;

    const GPU_SETUP_VARS: [&str; 4] = [
        "XDG_RUNTIME_DIR",
        "EGL_LOG_LEVEL",
        "MESA_GLSL_CACHE_DISABLE",
        "MESA_DEBUG",
    ];

    #[test]
    #[serial_test::serial]
    fn capture_gpu_setup_vars_restores_set_and_unset_values() {
        assert!(serial_test::is_locked_serially(None));
        let original = EnvGuard::capture_gpu_setup_vars();

        for key in GPU_SETUP_VARS {
            // SAFETY: This test holds the shared serial lock used by all GPU
            // integration tests that access these environment variables.
            unsafe { std::env::set_var(key, format!("original-{key}")) };
        }
        let existing = EnvGuard::capture_gpu_setup_vars();
        for key in GPU_SETUP_VARS {
            // SAFETY: This test holds the shared serial lock used by all GPU
            // integration tests that access these environment variables.
            unsafe { std::env::set_var(key, format!("changed-{key}")) };
        }
        drop(existing);
        for key in GPU_SETUP_VARS {
            assert_eq!(
                std::env::var_os(key).as_deref(),
                Some(std::ffi::OsStr::new(&format!("original-{key}"))),
                "{key} must be restored to its existing value"
            );
        }

        for key in GPU_SETUP_VARS {
            // SAFETY: This test holds the shared serial lock used by all GPU
            // integration tests that access these environment variables.
            unsafe { std::env::remove_var(key) };
        }
        let unset = EnvGuard::capture_gpu_setup_vars();
        for key in GPU_SETUP_VARS {
            // SAFETY: This test holds the shared serial lock used by all GPU
            // integration tests that access these environment variables.
            unsafe { std::env::set_var(key, format!("changed-{key}")) };
        }
        drop(unset);
        for key in GPU_SETUP_VARS {
            assert_eq!(
                std::env::var_os(key),
                None,
                "{key} must be restored to unset"
            );
        }

        drop(original);
    }
}
