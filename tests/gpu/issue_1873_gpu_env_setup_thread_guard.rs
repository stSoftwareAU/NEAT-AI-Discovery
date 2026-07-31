//! Integration tests for Issue #1873: the GPU environment setup must enforce
//! its own soundness precondition instead of asserting an unenforceable one.
//!
//! `setup_gpu_environment` is reachable from safe, lazily-initialised GPU paths
//! in a multi-threaded host, so it may not assume the process is
//! single-threaded — it must check, and skip the `set_var` when other threads
//! could be calling `getenv`.

use neat_ai_discovery::analysis::utils::platform::{GpuEnvSetup, setup_gpu_environment};
use std::sync::mpsc;

/// Park a thread until the returned sender is dropped or signalled, so the
/// process is guaranteed multi-threaded for the duration of the test.
fn spawn_live_thread() -> (mpsc::Sender<()>, std::thread::JoinHandle<()>) {
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    let handle = std::thread::spawn(move || {
        ready_tx.send(()).expect("ready signal should send");
        release_rx.recv().ok();
    });
    ready_rx.recv().expect("worker thread should start");
    (release_tx, handle)
}

/// The GPU environment setup is callable from ordinary safe code — no caller
/// has to discharge an invariant it cannot uphold — and repeated calls agree.
#[test]
fn setup_gpu_environment_is_safe_and_consistent() {
    let first = setup_gpu_environment();
    let second = setup_gpu_environment();

    assert_eq!(first, second, "repeated setup must return the same verdict");
}

/// With another thread alive the setup must never report `Applied`: writing the
/// environment then would race a concurrent `getenv`.
#[test]
fn setup_gpu_environment_never_writes_while_threads_live() {
    let (release_tx, handle) = spawn_live_thread();

    let verdict = setup_gpu_environment();

    release_tx.send(()).expect("release signal should send");
    handle.join().expect("worker thread should join");

    assert_ne!(
        verdict,
        GpuEnvSetup::Applied,
        "the environment must not be written while another thread is live"
    );
}

/// On Linux the verdict with a live second thread is specifically `Skipped`
/// (when setup is pending) or `NotRequired` (when the host already set every
/// variable) — never a write.
#[cfg(target_os = "linux")]
#[test]
fn setup_gpu_environment_skips_or_is_unnecessary_on_linux() {
    let (release_tx, handle) = spawn_live_thread();

    let verdict = setup_gpu_environment();

    release_tx.send(()).expect("release signal should send");
    handle.join().expect("worker thread should join");

    assert!(
        matches!(verdict, GpuEnvSetup::Skipped | GpuEnvSetup::NotRequired),
        "expected a skipped or unnecessary verdict, got {verdict:?}"
    );
}

/// Non-Linux platforms need no environment setup at all.
#[cfg(not(target_os = "linux"))]
#[test]
fn setup_gpu_environment_is_not_required_off_linux() {
    assert_eq!(setup_gpu_environment(), GpuEnvSetup::NotRequired);
}
