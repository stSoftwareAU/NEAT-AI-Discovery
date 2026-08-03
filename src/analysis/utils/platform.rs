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
#[cfg(any(target_os = "linux", all(unix, test)))]
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

/// Permissions the fallback runtime directory must carry: owner-only access, as
/// the XDG Base Directory specification requires (Issue #1904).
#[cfg(any(target_os = "linux", all(unix, test)))]
const RUNTIME_DIR_MODE: u32 = 0o700;

/// Create — or validate an already-present — owner-only runtime directory under
/// `base`, returning its path only when it is safe to hand to wgpu.
///
/// The leaf name is per-uid so two users on one host never contend for the same
/// path. The directory is created non-recursively so creation fails loudly with
/// `AlreadyExists` rather than silently adopting whatever is already there, and
/// its mode is set explicitly afterwards because `mkdir` masks the requested
/// mode with the process umask.
///
/// An existing entry is adopted only when it is a real directory (not a
/// symlink), owned by this effective uid, and carries no group or other
/// permission bits. Anything else is refused with a warning: `XDG_RUNTIME_DIR`
/// is then left unset, which wgpu tolerates, rather than pointing it at a
/// directory another local user controls.
#[cfg(any(target_os = "linux", all(unix, test)))]
fn prepare_runtime_dir(base: &std::path::Path) -> Option<std::path::PathBuf> {
    use std::fs::{DirBuilder, Permissions, set_permissions};
    use std::io::ErrorKind;
    use std::os::unix::fs::{DirBuilderExt, PermissionsExt};

    // SAFETY: `geteuid` takes no arguments, touches no caller-owned memory and
    // cannot fail; it is only `unsafe` because it is an FFI call.
    let uid = unsafe { libc::geteuid() };
    let runtime_dir = base.join(format!("neat-ai-discovery-runtime-{uid}"));

    match DirBuilder::new()
        .mode(RUNTIME_DIR_MODE)
        .recursive(false)
        .create(&runtime_dir)
    {
        Ok(()) => {
            // We created it exclusively, so it is ours to chmod; this pins the
            // mode to 0700 whatever the process umask cleared.
            if let Err(e) = set_permissions(&runtime_dir, Permissions::from_mode(RUNTIME_DIR_MODE))
            {
                tracing::warn!(
                    ?runtime_dir,
                    %e,
                    "failed to restrict runtime directory to mode 0700; leaving XDG_RUNTIME_DIR unset"
                );
                return None;
            }
            Some(runtime_dir)
        }
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {
            if runtime_dir_is_trustworthy(&runtime_dir, uid) {
                Some(runtime_dir)
            } else {
                None
            }
        }
        Err(e) => {
            tracing::warn!(?runtime_dir, %e, "failed to create XDG_RUNTIME_DIR");
            None
        }
    }
}

/// Whether a pre-existing runtime directory may be adopted: a real directory,
/// owned by `uid`, with no group or other permission bits. Every rejection is
/// warned about, naming the reason.
#[cfg(any(target_os = "linux", all(unix, test)))]
fn runtime_dir_is_trustworthy(runtime_dir: &std::path::Path, uid: u32) -> bool {
    use std::os::unix::fs::MetadataExt;

    // `symlink_metadata` does not follow a final symlink, so a symlinked path
    // is reported as one instead of as its (possibly innocuous) target.
    let metadata = match std::fs::symlink_metadata(runtime_dir) {
        Ok(metadata) => metadata,
        Err(e) => {
            tracing::warn!(
                ?runtime_dir,
                %e,
                "refusing pre-existing XDG_RUNTIME_DIR: cannot stat it; leaving XDG_RUNTIME_DIR unset"
            );
            return false;
        }
    };

    if !metadata.file_type().is_dir() {
        tracing::warn!(
            ?runtime_dir,
            file_type = ?metadata.file_type(),
            "refusing pre-existing XDG_RUNTIME_DIR: not a directory (a symlink or file may be an \
             attempt to redirect the runtime directory); leaving XDG_RUNTIME_DIR unset"
        );
        return false;
    }

    if metadata.uid() != uid {
        tracing::warn!(
            ?runtime_dir,
            owner_uid = metadata.uid(),
            expected_uid = uid,
            "refusing pre-existing XDG_RUNTIME_DIR: owned by another user; leaving \
             XDG_RUNTIME_DIR unset"
        );
        return false;
    }

    let mode = metadata.mode() & 0o777;
    if mode & 0o077 != 0 {
        tracing::warn!(
            ?runtime_dir,
            mode = format!("{mode:04o}"),
            "refusing pre-existing XDG_RUNTIME_DIR: group or other access is permitted, it must \
             be mode 0700; leaving XDG_RUNTIME_DIR unset"
        );
        return false;
    }

    true
}

/// Create a fallback runtime directory and point `XDG_RUNTIME_DIR` at it when
/// the variable is unset.
///
/// The directory is owner-only and must be owned by this process's user — see
/// [`prepare_runtime_dir`]. When no trustworthy directory can be obtained the
/// variable is left unset rather than pointed at an untrusted path.
///
/// # Safety
///
/// Carries the same precondition as [`set_env_if_unset`]: no other thread may
/// access the process environment concurrently.
#[cfg(any(target_os = "linux", all(unix, test)))]
unsafe fn apply_xdg_runtime_dir() {
    use std::env;

    if env::var("XDG_RUNTIME_DIR").is_ok() {
        return;
    }
    // Create a temporary runtime directory if XDG_RUNTIME_DIR is not set
    let Ok(temp_dir) = std::env::temp_dir().canonicalize() else {
        return;
    };
    let Some(runtime_dir) = prepare_runtime_dir(&temp_dir) else {
        return;
    };
    // SAFETY: The caller of this `unsafe fn` upholds the no-concurrent-access
    // precondition of `set_env_if_unset`.
    unsafe {
        set_env_if_unset("XDG_RUNTIME_DIR", runtime_dir.to_string_lossy().as_ref());
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

    /// Collects formatted tracing output so a test can assert that a rejection
    /// was reported rather than swallowed.
    #[cfg(unix)]
    #[derive(Clone, Default)]
    struct LogCapture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

    #[cfg(unix)]
    impl LogCapture {
        fn contents(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().expect("log buffer should not be poisoned"))
                .into_owned()
        }
    }

    #[cfg(unix)]
    impl std::io::Write for LogCapture {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("log buffer should not be poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[cfg(unix)]
    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogCapture {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    /// Run `body` with tracing captured, returning the emitted log text.
    #[cfg(unix)]
    fn capture_logs<T>(body: impl FnOnce() -> T) -> (T, String) {
        let capture = LogCapture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture.clone())
            .with_max_level(tracing::Level::WARN)
            .finish();
        let result = tracing::subscriber::with_default(subscriber, body);
        (result, capture.contents())
    }

    /// The per-uid runtime directory this build expects under `base`.
    #[cfg(unix)]
    fn expected_runtime_dir(base: &std::path::Path) -> std::path::PathBuf {
        // SAFETY: `geteuid` takes no arguments and cannot fail.
        let uid = unsafe { libc::geteuid() };
        base.join(format!("neat-ai-discovery-runtime-{uid}"))
    }

    /// The mode bits of `path`, following no symlink.
    #[cfg(unix)]
    fn mode_of(path: &std::path::Path) -> u32 {
        use std::os::unix::fs::MetadataExt;

        std::fs::symlink_metadata(path)
            .expect("path should exist")
            .mode()
            & 0o777
    }

    /// A freshly created runtime directory is owner-only, even when the process
    /// umask would otherwise leave it world-readable (Issue #1904). Serial
    /// because the umask is process-global.
    #[cfg(unix)]
    #[test]
    #[serial_test::serial]
    fn xdg_runtime_dir_created_with_mode_0700() {
        let base = tempfile::tempdir().expect("temp dir should be created");

        // SAFETY: serialised via #[serial]; the permissive umask is restored below.
        let previous_umask = unsafe { libc::umask(0) };
        let created = prepare_runtime_dir(base.path());
        // SAFETY: serialised via #[serial]; restore the process umask.
        unsafe { libc::umask(previous_umask) };

        let created = created.expect("a fresh runtime directory should be usable");
        assert_eq!(
            created,
            expected_runtime_dir(base.path()),
            "the runtime directory leaf name must be per-uid"
        );
        assert_eq!(
            mode_of(&created),
            0o700,
            "the runtime directory must be owner-only regardless of the umask"
        );
    }

    /// A pre-existing world-writable directory is refused, with a warning, and
    /// `XDG_RUNTIME_DIR` is left unset rather than adopted (Issue #1904).
    /// Serial because it mutates the process environment.
    #[cfg(unix)]
    #[test]
    #[serial_test::serial]
    fn xdg_runtime_dir_rejects_preexisting_world_writable() {
        use std::env;
        use std::os::unix::fs::PermissionsExt;

        let base = tempfile::tempdir().expect("temp dir should be created");
        let base_path = base
            .path()
            .canonicalize()
            .expect("temp dir should canonicalise");
        let hostile = expected_runtime_dir(&base_path);
        std::fs::create_dir(&hostile).expect("pre-existing directory should be created");
        std::fs::set_permissions(&hostile, std::fs::Permissions::from_mode(0o777))
            .expect("pre-existing directory should be made world-writable");

        let (prepared, logs) = capture_logs(|| prepare_runtime_dir(&base_path));
        assert!(
            prepared.is_none(),
            "a world-writable directory must never be adopted"
        );
        assert!(
            logs.contains("refusing pre-existing XDG_RUNTIME_DIR"),
            "the rejection must be warned about, got: {logs}"
        );

        let saved_runtime = env::var("XDG_RUNTIME_DIR").ok();
        let saved_tmpdir = env::var("TMPDIR").ok();
        // SAFETY: serialised via #[serial]; both variables are restored below.
        unsafe {
            env::remove_var("XDG_RUNTIME_DIR");
            env::set_var("TMPDIR", &base_path);
        }

        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { apply_xdg_runtime_dir() };

        let after = env::var("XDG_RUNTIME_DIR").ok();

        // SAFETY: serialised via #[serial]; restore the pre-existing values.
        unsafe {
            match saved_runtime {
                Some(v) => env::set_var("XDG_RUNTIME_DIR", v),
                None => env::remove_var("XDG_RUNTIME_DIR"),
            }
            match saved_tmpdir {
                Some(v) => env::set_var("TMPDIR", v),
                None => env::remove_var("TMPDIR"),
            }
        }

        assert_eq!(
            after, None,
            "XDG_RUNTIME_DIR must stay unset when only an untrusted directory is available"
        );
    }

    /// A symlink at the runtime-directory path is refused: it could redirect
    /// wgpu's runtime files anywhere (Issue #1904).
    #[cfg(unix)]
    #[test]
    fn xdg_runtime_dir_rejects_symlink() {
        let base = tempfile::tempdir().expect("temp dir should be created");
        let target = base.path().join("attacker-controlled");
        std::fs::create_dir(&target).expect("symlink target should be created");
        std::os::unix::fs::symlink(&target, expected_runtime_dir(base.path()))
            .expect("symlink should be created");

        let (prepared, logs) = capture_logs(|| prepare_runtime_dir(base.path()));

        assert!(prepared.is_none(), "a symlinked path must never be adopted");
        assert!(
            logs.contains("refusing pre-existing XDG_RUNTIME_DIR"),
            "the rejection must be warned about, got: {logs}"
        );
    }

    /// The common case still works: an owner-only directory left by an earlier
    /// run is reused, so preparation is idempotent (Issue #1904).
    #[cfg(unix)]
    #[test]
    fn xdg_runtime_dir_adopts_own_owner_only_directory() {
        let base = tempfile::tempdir().expect("temp dir should be created");

        let first = prepare_runtime_dir(base.path()).expect("first preparation should succeed");
        let second = prepare_runtime_dir(base.path())
            .expect("an owner-only directory from an earlier run should be reused");

        assert_eq!(first, second, "preparation must be idempotent");
        assert_eq!(mode_of(&second), 0o700, "the mode must remain owner-only");
    }

    /// An already-set `XDG_RUNTIME_DIR` is respected untouched — the fallback
    /// never overrides the host's choice (Issue #1904). Serial because it
    /// mutates the process environment.
    #[cfg(unix)]
    #[test]
    #[serial_test::serial]
    fn xdg_runtime_dir_preserves_host_value() {
        use std::env;

        let saved = env::var("XDG_RUNTIME_DIR").ok();
        // SAFETY: serialised via #[serial]; the original value is restored below.
        unsafe { env::set_var("XDG_RUNTIME_DIR", "/run/user/host-supplied") };

        // SAFETY: serialised via #[serial]; no other thread touches the env here.
        unsafe { apply_xdg_runtime_dir() };

        let after = env::var("XDG_RUNTIME_DIR").ok();

        // SAFETY: serialised via #[serial]; restore any pre-existing value.
        unsafe {
            match saved {
                Some(v) => env::set_var("XDG_RUNTIME_DIR", v),
                None => env::remove_var("XDG_RUNTIME_DIR"),
            }
        }

        assert_eq!(
            after.as_deref(),
            Some("/run/user/host-supplied"),
            "a host-supplied XDG_RUNTIME_DIR must be left untouched"
        );
    }

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
