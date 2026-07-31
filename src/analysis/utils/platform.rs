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
//
// Safe callers must not assert that precondition — they cannot enforce it. GPU
// init is lazy and reachable from a multi-threaded host at an arbitrary time
// (Issue #1873). [`setup_gpu_environment`] is the safe entry point: it *checks*
// the invariant by reading the live OS thread count from `/proc/self/task` and
// only writes when the process is observably single-threaded. When other
// threads already exist it skips the writes and warns loudly, degrading GPU
// diagnostics rather than risking a `getenv` data race.

/// Set an environment variable only if it is currently unset.
///
/// Returns `true` when the variable was written, `false` when it was already
/// present (and left untouched).
///
/// # Safety
///
/// The caller must guarantee that no other thread accesses the process
/// environment concurrently — see the module-level note. This helper does not
/// and cannot enforce that invariant; it centralises the single `unsafe`
/// write so its justification lives in one place. The obligation is propagated
/// to the type level so no safe caller can reach the `set_var` without an
/// `unsafe` block acknowledging the precondition.
#[cfg(target_os = "linux")]
unsafe fn set_env_if_unset(key: &str, value: &str) -> bool {
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
///
/// # Safety
///
/// No other thread may access the process environment concurrently — call
/// during early GPU initialisation, before spawning any thread that touches
/// the environment (see the module-level note). The `Once` guard ensures the
/// write happens once, not exclusively, so it cannot uphold this invariant on
/// the caller's behalf.
#[cfg(target_os = "linux")]
pub unsafe fn suppress_mesa_warnings_if_requested() {
    use std::sync::Once;

    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if crate::config::quiet_gpu() {
            // SAFETY: The caller of this `unsafe fn` guarantees no other thread
            // accesses the process environment concurrently.
            unsafe { apply_mesa_suppression() };
        }
    });
}

/// Apply the Mesa/libEGL suppression environment variables.
///
/// Each variable is only written when currently unset.
///
/// # Safety
///
/// Carries the same precondition as [`set_env_if_unset`]: no other thread may
/// access the process environment concurrently.
#[cfg(target_os = "linux")]
unsafe fn apply_mesa_suppression() {
    // SAFETY: The caller of this `unsafe fn` upholds the no-concurrent-access
    // precondition of `set_env_if_unset`.
    unsafe {
        // Suppress EGL debug messages (these cause "failed to open /dev/dri/..." warnings)
        set_env_if_unset("EGL_LOG_LEVEL", "fatal");
        // Suppress Mesa GLSL shader cache warnings
        set_env_if_unset("MESA_GLSL_CACHE_DISABLE", "true");
        // Suppress general Mesa debug output
        set_env_if_unset("MESA_DEBUG", "silent");
    }
}

/// No-op on non-Linux platforms.
///
/// # Safety
///
/// This stub does nothing, but is marked `unsafe` so call sites are uniform
/// across platforms with the Linux implementation (see the module-level note).
#[cfg(not(target_os = "linux"))]
pub unsafe fn suppress_mesa_warnings_if_requested() {
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
///
/// # Safety
///
/// No other thread may access the process environment concurrently — call
/// during early GPU initialisation, before spawning any thread that touches
/// the environment (see the module-level note). The `Once` guard ensures the
/// write happens once, not exclusively, so it cannot uphold this invariant on
/// the caller's behalf.
#[cfg(target_os = "linux")]
pub unsafe fn ensure_xdg_runtime_dir() {
    use std::sync::Once;

    static INIT: Once = Once::new();
    // SAFETY: The caller of this `unsafe fn` guarantees no other thread accesses
    // the process environment concurrently.
    INIT.call_once(|| unsafe { apply_xdg_runtime_dir() });
}

/// Create a fallback runtime directory and point `XDG_RUNTIME_DIR` at it when
/// the variable is unset.
///
/// # Safety
///
/// Carries the same precondition as [`set_env_if_unset`]: no other thread may
/// access the process environment concurrently.
#[cfg(target_os = "linux")]
unsafe fn apply_xdg_runtime_dir() {
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
            // SAFETY: The caller of this `unsafe fn` upholds the
            // no-concurrent-access precondition of `set_env_if_unset`.
            unsafe {
                set_env_if_unset("XDG_RUNTIME_DIR", runtime_dir.to_string_lossy().as_ref());
            }
        }
    }
}

/// No-op on non-Linux platforms.
///
/// # Safety
///
/// This stub does nothing, but is marked `unsafe` so call sites are uniform
/// across platforms with the Linux implementation (see the module-level note).
#[cfg(not(target_os = "linux"))]
pub unsafe fn ensure_xdg_runtime_dir() {
    // No-op on non-Linux platforms
}

// =============================================================================
// Guarded (safe) GPU environment setup — Issue #1873
// =============================================================================

/// Verdict returned by [`setup_gpu_environment`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuEnvSetup {
    /// The process was observably single-threaded, so the environment writes
    /// were applied.
    Applied,
    /// Nothing needed writing — either the platform requires no setup
    /// (non-Linux) or every variable is already present in the environment.
    NotRequired,
    /// Other threads are already live (or the thread count could not be
    /// determined), so the writes were skipped to avoid racing a concurrent
    /// `getenv`. GPU init continues with the environment as the host left it.
    Skipped,
}

/// Whether the process environment may be mutated, given the observed number of
/// live OS threads (`None` when the count could not be determined).
///
/// Only a genuinely single-threaded process is safe: `set_var` races any
/// concurrent `getenv`. A new thread can only be created by an existing thread,
/// so when this thread is the only one, and it spawns none before the write,
/// no concurrent reader can exist. An undeterminable count is treated as unsafe.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn may_mutate_environment(thread_count: Option<usize>) -> bool {
    thread_count == Some(1)
}

/// Count the live OS threads in this process, or `None` if it cannot be read.
#[cfg(target_os = "linux")]
fn live_thread_count() -> Option<usize> {
    let mut count = 0usize;
    for entry in std::fs::read_dir("/proc/self/task").ok()? {
        // A read error mid-iteration means the count is untrustworthy.
        entry.ok()?;
        count += 1;
    }
    (count > 0).then_some(count)
}

/// Whether any GPU environment variable still needs writing.
#[cfg(target_os = "linux")]
fn env_setup_pending() -> bool {
    use std::env;

    if env::var("XDG_RUNTIME_DIR").is_err() {
        return true;
    }
    crate::config::quiet_gpu()
        && ["EGL_LOG_LEVEL", "MESA_GLSL_CACHE_DISABLE", "MESA_DEBUG"]
            .iter()
            .any(|key| env::var(key).is_err())
}

/// Warn once that the GPU environment setup was skipped, and why.
#[cfg(target_os = "linux")]
fn warn_setup_skipped(thread_count: Option<usize>) {
    use std::sync::Once;

    static WARNED: Once = Once::new();
    WARNED.call_once(|| {
        tracing::warn!(
            ?thread_count,
            "GPU environment setup skipped: this process is already \
             multi-threaded, so writing XDG_RUNTIME_DIR / Mesa variables would \
             race a concurrent getenv (undefined behaviour). Set \
             XDG_RUNTIME_DIR — and the Mesa variables when \
             NEAT_AI_DISCOVERY_QUIET_GPU=1 — in the host environment before \
             starting the process. GPU initialisation continues and may emit \
             Wayland/Mesa warnings."
        );
    });
}

/// Apply the Linux GPU environment setup, but only while it is provably safe.
///
/// This is the safe entry point every GPU initialisation path should use. It
/// enforces — rather than assumes — the soundness precondition of the `unsafe`
/// [`suppress_mesa_warnings_if_requested`] / [`ensure_xdg_runtime_dir`] entry
/// points by reading the live thread count from `/proc/self/task` first
/// (Issue #1873). GPU init is lazy and reachable from a multi-threaded host, so
/// no caller can promise the process is single-threaded; when it is not, the
/// writes are skipped and logged instead of risking a `getenv` data race.
#[cfg(target_os = "linux")]
pub fn setup_gpu_environment() -> GpuEnvSetup {
    if !env_setup_pending() {
        return GpuEnvSetup::NotRequired;
    }

    let thread_count = live_thread_count();
    if !may_mutate_environment(thread_count) {
        warn_setup_skipped(thread_count);
        return GpuEnvSetup::Skipped;
    }

    // SAFETY: `/proc/self/task` reported exactly one live thread — this one.
    // Only an existing thread can create a new one, and this thread creates
    // none between the check and the writes, so no other thread can read the
    // process environment concurrently.
    unsafe {
        suppress_mesa_warnings_if_requested();
        ensure_xdg_runtime_dir();
    }
    GpuEnvSetup::Applied
}

/// No-op on non-Linux platforms: nothing in the environment needs setting up.
#[cfg(not(target_os = "linux"))]
pub fn setup_gpu_environment() -> GpuEnvSetup {
    GpuEnvSetup::NotRequired
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
        // SAFETY: the test harness has not spawned any thread that reads the
        // process environment, so the early-init precondition holds.
        unsafe {
            suppress_mesa_warnings_if_requested();
            suppress_mesa_warnings_if_requested();
        }
    }

    /// Smoke test: `ensure_xdg_runtime_dir` is a one-time environment setup
    /// guarded by `Once`. On non-Linux it is a no-op. There is no return
    /// value or queryable state — the only contract is that repeated calls
    /// do not panic.
    #[test]
    fn test_ensure_xdg_runtime_dir_does_not_panic() {
        // SAFETY: the test harness has not spawned any thread that reads the
        // process environment, so the early-init precondition holds.
        unsafe {
            ensure_xdg_runtime_dir();
            ensure_xdg_runtime_dir();
        }
    }

    /// Only a single live thread permits an environment write; anything else —
    /// including an undeterminable count — must be refused (Issue #1873).
    #[test]
    fn test_may_mutate_environment_requires_single_thread() {
        assert!(
            may_mutate_environment(Some(1)),
            "a single-threaded process has no concurrent getenv reader"
        );
        assert!(
            !may_mutate_environment(Some(2)),
            "a second live thread may read the environment concurrently"
        );
        assert!(
            !may_mutate_environment(Some(64)),
            "a rayon/host thread pool must block the write"
        );
        assert!(
            !may_mutate_environment(None),
            "an unknown thread count must be treated as unsafe"
        );
    }

    /// The live thread count reflects threads that actually exist: it grows
    /// while an extra thread is alive.
    #[cfg(target_os = "linux")]
    #[test]
    fn test_live_thread_count_sees_extra_threads() {
        use std::sync::mpsc;

        let baseline = live_thread_count().expect("/proc/self/task should be readable");
        assert!(baseline >= 1, "the calling thread must be counted");

        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            ready_tx.send(()).expect("ready signal should send");
            release_rx.recv().ok();
        });
        ready_rx.recv().expect("worker thread should start");

        let with_worker = live_thread_count().expect("/proc/self/task should be readable");
        assert!(
            with_worker >= 2,
            "an extra live thread must be visible in the count, saw {with_worker}"
        );
        assert!(
            !may_mutate_environment(Some(with_worker)),
            "the environment must not be written while another thread lives"
        );

        release_tx.send(()).expect("release signal should send");
        handle.join().expect("worker thread should join");
    }

    /// `setup_gpu_environment` is a *safe* function: no caller has to promise
    /// an invariant it cannot enforce, and repeated calls agree (Issue #1873).
    #[test]
    fn test_setup_gpu_environment_is_safe_and_consistent() {
        let first = setup_gpu_environment();
        let second = setup_gpu_environment();
        assert_eq!(first, second, "repeated setup must return the same verdict");
    }

    /// On non-Linux platforms there is nothing to set up.
    #[cfg(not(target_os = "linux"))]
    #[test]
    fn test_setup_gpu_environment_is_a_noop_off_linux() {
        assert_eq!(setup_gpu_environment(), GpuEnvSetup::NotRequired);
    }

    /// With another thread alive, the setup refuses to write and leaves an
    /// unset `XDG_RUNTIME_DIR` unset — the regression guard for the #1873
    /// `getenv` race. Serial because it mutates the process environment.
    #[cfg(target_os = "linux")]
    #[test]
    #[serial_test::serial]
    fn test_setup_gpu_environment_skips_when_other_threads_live() {
        use std::env;
        use std::sync::mpsc;

        let saved = env::var("XDG_RUNTIME_DIR").ok();
        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { env::remove_var("XDG_RUNTIME_DIR") };

        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let handle = std::thread::spawn(move || {
            ready_tx.send(()).expect("ready signal should send");
            release_rx.recv().ok();
        });
        ready_rx.recv().expect("worker thread should start");

        let verdict = setup_gpu_environment();

        assert_eq!(
            verdict,
            GpuEnvSetup::Skipped,
            "a live second thread must block the environment write"
        );
        assert!(
            env::var("XDG_RUNTIME_DIR").is_err(),
            "XDG_RUNTIME_DIR must stay unset when the write is refused"
        );

        release_tx.send(()).expect("release signal should send");
        handle.join().expect("worker thread should join");

        // SAFETY: serialised via #[serial]; restore any pre-existing value.
        unsafe {
            match saved {
                Some(v) => env::set_var("XDG_RUNTIME_DIR", v),
                None => env::remove_var("XDG_RUNTIME_DIR"),
            }
        }
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

        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        let wrote = unsafe { set_env_if_unset(key, "applied") };

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

        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        let wrote = unsafe { set_env_if_unset(key, "replacement") };

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

        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { apply_xdg_runtime_dir() };

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
