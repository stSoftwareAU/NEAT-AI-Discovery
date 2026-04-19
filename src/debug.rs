//! Debug utilities for diagnosing deadlocks and thread issues.
//!
//! This module provides:
//! - **Deadlock detection**: Using `parking_lot`'s deadlock detection feature
//! - **Thread dump on signal**: SIGUSR1 (kill -USR1) dumps all thread backtraces
//!
//! # Usage
//!
//! Call `init_debug_handlers()` early in your program to enable both features.
//! On deadlock, the program will abort after printing backtrace information.
//! On SIGUSR1 (kill -USR1), thread backtraces are printed to stderr without exiting.
//!
//! Call `shutdown_debug_handlers()` before process exit to cleanly stop the
//! background threads and unregister signal handlers (Issue #994).
//!
//! # Example
//!
//! ```rust,ignore
//! // At startup
//! neat_ai_discovery::debug::init_debug_handlers();
//!
//! // Now you can send kill -USR1 <pid> to dump threads
//! // And deadlocks will be detected and abort the process
//!
//! // Before shutdown
//! neat_ai_discovery::debug::shutdown_debug_handlers();
//! ```

use parking_lot::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Static flag to ensure handlers are only initialised once.
static DEBUG_HANDLERS_INITIALISED: OnceLock<()> = OnceLock::new();

/// Flag to track if we're in verbose mode (shows more detail).
static VERBOSE_MODE: AtomicBool = AtomicBool::new(false);

/// Shutdown flag — when set, background threads exit their loops.
static SHUTDOWN_REQUESTED: AtomicBool = AtomicBool::new(false);

/// Interval between deadlock checks (in seconds).
///
/// Kept short (5 s) so that deadlocks are detected quickly without adding
/// meaningful overhead — the check itself is very cheap.
const DEADLOCK_CHECK_INTERVAL_SECS: u64 = 5;

/// Maximum time to wait for each debug thread to exit during shutdown (seconds).
const SHUTDOWN_JOIN_TIMEOUT_SECS: u64 = 10;

/// Holds the join handles and shutdown primitives for background debug threads.
///
/// Stored globally so `shutdown_debug_handlers()` can stop them cleanly.
struct DebugThreadState {
    deadlock_handle: Option<thread::JoinHandle<()>>,
    #[cfg(unix)]
    signal_handle: Option<thread::JoinHandle<()>>,
    #[cfg(unix)]
    signal_closer: Option<signal_hook::iterator::backend::Handle>,
}

static DEBUG_THREADS: Mutex<Option<DebugThreadState>> = Mutex::new(None);

/// Initialise debug handlers for deadlock detection and signal-based thread dumps.
///
/// This function is idempotent - calling it multiple times has no effect.
///
/// # Features
///
/// 1. **Deadlock Detection**: A background thread checks for deadlocks every 5 seconds.
///    If a deadlock is detected, the program prints diagnostics and aborts.
///
/// 2. **SIGUSR1 Handler (Unix only)**: Sending `kill -USR1 <pid>` prints all
///    thread backtraces to stderr without terminating the process.
///
/// # Platform Support
///
/// - **macOS/Linux**: Full support for both features
/// - **Windows**: Only deadlock detection is available (no SIGUSR1 equivalent)
pub fn init_debug_handlers() {
    DEBUG_HANDLERS_INITIALISED.get_or_init(|| {
        // Reset shutdown flag in case a previous shutdown was called (e.g. in tests).
        SHUTDOWN_REQUESTED.store(false, Ordering::Relaxed);

        // Check if verbose mode is enabled
        if crate::config::verbose() {
            VERBOSE_MODE.store(true, Ordering::Relaxed);
        }

        let mut state = DebugThreadState {
            deadlock_handle: None,
            #[cfg(unix)]
            signal_handle: None,
            #[cfg(unix)]
            signal_closer: None,
        };

        // Start deadlock detection thread
        state.deadlock_handle = start_deadlock_detector();

        // Install signal handler (Unix only)
        #[cfg(unix)]
        {
            let (handle, closer) = install_signal_handler();
            state.signal_handle = handle;
            state.signal_closer = closer;
        }

        *DEBUG_THREADS.lock() = Some(state);

        tracing::debug!("debug handlers initialised (deadlock detection + kill -USR1 thread dump)");
    });
}

/// Shut down background debug threads cleanly (Issue #994).
///
/// This function signals all debug background threads to exit and waits
/// (with a timeout) for them to finish. It should be called before
/// process exit to prevent the deadlock-detector and signal-handler
/// threads from keeping the process alive.
///
/// This function is idempotent — calling it multiple times is safe.
pub fn shutdown_debug_handlers() {
    // Signal all debug threads to stop.
    SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);

    let mut guard = DEBUG_THREADS.lock();
    let Some(mut state) = guard.take() else {
        return;
    };

    // Close the signal handler first (unblocks `signals.forever()`).
    #[cfg(unix)]
    if let Some(closer) = state.signal_closer.take() {
        closer.close();
    }

    // Join deadlock-detector thread with timeout.
    if let Some(handle) = state.deadlock_handle.take() {
        join_with_timeout(handle, "deadlock-detector");
    }

    // Join signal-handler thread with timeout.
    #[cfg(unix)]
    if let Some(handle) = state.signal_handle.take() {
        join_with_timeout(handle, "signal-handler");
    }

    tracing::debug!("debug handlers shut down");
}

/// Join a thread with a bounded timeout to avoid blocking shutdown forever.
fn join_with_timeout(handle: thread::JoinHandle<()>, name: &str) {
    // We cannot set a timeout on `JoinHandle::join()` directly, so we spawn
    // a helper thread that performs the blocking join and notify via channel.
    let (tx, rx) = std::sync::mpsc::channel();
    let thread_name = name.to_string();
    let _ = thread::Builder::new()
        .name(format!("{thread_name}-joiner"))
        .spawn(move || {
            let _ = handle.join();
            let _ = tx.send(());
        });

    match rx.recv_timeout(Duration::from_secs(SHUTDOWN_JOIN_TIMEOUT_SECS)) {
        Ok(()) => {}
        Err(_) => {
            tracing::trace!(
                thread = name,
                timeout_secs = SHUTDOWN_JOIN_TIMEOUT_SECS,
                "debug thread did not exit within timeout — abandoning"
            );
        }
    }
}

/// Start background thread for deadlock detection.
///
/// Uses `parking_lot::deadlock::check_deadlock()` to detect deadlocks.
/// When a deadlock is found, prints full information to stderr, flushes it,
/// and calls `std::process::abort()` to terminate the process immediately.
///
/// We use `abort()` rather than `panic!()` because most FFI entrypoints wrap
/// calls in `catch_unwind()`, which would swallow a panic and keep the worker
/// alive with permanently deadlocked threads.
fn start_deadlock_detector() -> Option<thread::JoinHandle<()>> {
    match thread::Builder::new()
        .name("deadlock-detector".to_string())
        .spawn(move || {
            loop {
                // Check shutdown flag before sleeping so we exit promptly.
                if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
                    return;
                }

                thread::sleep(Duration::from_secs(DEADLOCK_CHECK_INTERVAL_SECS));

                // Re-check after sleep in case shutdown was requested while sleeping.
                if SHUTDOWN_REQUESTED.load(Ordering::Relaxed) {
                    return;
                }

                let deadlocks = parking_lot::deadlock::check_deadlock();
                if deadlocks.is_empty() {
                    continue;
                }

                // Deadlock detected! Print details and abort.
                eprintln!("\n{}", "=".repeat(80));
                eprintln!("DEADLOCK DETECTED - {} deadlock(s) found", deadlocks.len());
                eprintln!("{}\n", "=".repeat(80));

                for (i, threads) in deadlocks.iter().enumerate() {
                    eprintln!(
                        "--- Deadlock #{} ({} threads involved) ---",
                        i + 1,
                        threads.len()
                    );
                    for t in threads {
                        eprintln!("\nThread ID: {:?}", t.thread_id());
                        eprintln!("Backtrace:\n{:#?}", t.backtrace());
                    }
                    eprintln!();
                }

                eprintln!("{}", "=".repeat(80));
                eprintln!(
                    "ABORTING due to deadlock — {} deadlock(s) involving {} total threads. \
                     See above for thread backtraces.",
                    deadlocks.len(),
                    deadlocks.iter().map(std::vec::Vec::len).sum::<usize>()
                );
                eprintln!("{}\n", "=".repeat(80));

                // Flush stderr so the diagnostic output is not lost before we abort.
                let _ = std::io::Write::flush(&mut std::io::stderr());

                // We intentionally use abort() rather than panic!() because most FFI
                // entrypoints use catch_unwind(), which would swallow a panic and keep
                // the worker alive with permanently deadlocked threads.
                std::process::abort();
            }
        }) {
        Ok(handle) => Some(handle),
        Err(e) => {
            tracing::warn!("failed to spawn deadlock detector thread: {e}");
            None
        }
    }
}

/// Install signal handler for SIGUSR1 (kill -USR1) to dump thread backtraces.
///
/// Returns the thread `JoinHandle` and a `signal_hook` `Handle` that can be used to
/// close the signal iterator and unblock the thread (Issue #994).
///
/// SIGUSR1 is a user-defined signal with no default action, making it safe
/// to use for diagnostics without risk of terminating the process.
#[cfg(unix)]
fn install_signal_handler() -> (
    Option<thread::JoinHandle<()>>,
    Option<signal_hook::iterator::backend::Handle>,
) {
    use signal_hook::consts::SIGUSR1;
    use signal_hook::iterator::Signals;

    let mut signals = match Signals::new([SIGUSR1]) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("failed to install SIGUSR1 handler: {e}");
            return (None, None);
        }
    };

    // Get a handle BEFORE moving signals into the thread so we can close
    // the iterator from shutdown_debug_handlers() (Issue #994).
    let closer = signals.handle();

    let handle = match thread::Builder::new()
        .name("signal-handler".to_string())
        .spawn(move || {
            for _sig in signals.forever() {
                dump_all_threads();
            }
        }) {
        Ok(h) => Some(h),
        Err(e) => {
            tracing::warn!("failed to spawn signal handler thread: {e}");
            return (None, None);
        }
    };

    (handle, Some(closer))
}

/// Dump backtraces of all threads to stderr.
///
/// This is called when SIGUSR1 (kill -USR1) is received.
/// On macOS, automatically runs `sample` to get full thread backtraces.
/// On other platforms, provides instructions for manual debugging.
fn dump_all_threads() {
    let timestamp = chrono_lite_timestamp();
    let pid = std::process::id();

    eprintln!("\n{}", "=".repeat(80));
    eprintln!("THREAD DUMP - {timestamp} (kill -USR1 received)");
    eprintln!("Process ID: {pid}");
    eprintln!("{}\n", "=".repeat(80));

    // Check for any deadlocked threads first (this is fast)
    let deadlocks = parking_lot::deadlock::check_deadlock();
    if !deadlocks.is_empty() {
        eprintln!("--- DEADLOCKED THREADS DETECTED ---");
        for (i, threads) in deadlocks.iter().enumerate() {
            eprintln!("\nDeadlock #{} ({} threads):", i + 1, threads.len());
            for t in threads {
                eprintln!("  Thread ID: {:?}", t.thread_id());
                if VERBOSE_MODE.load(Ordering::Relaxed) {
                    eprintln!("  Backtrace:\n{:#?}", t.backtrace());
                }
            }
        }
        eprintln!();
    } else {
        eprintln!("--- No mutex deadlocks detected ---\n");
    }

    // On macOS, run `sample` to get full thread backtraces
    #[cfg(target_os = "macos")]
    {
        eprintln!("--- Running 'sample' for full thread analysis (1 second) ---\n");
        run_sample_command(pid);
    }

    // On non-macOS, show manual instructions
    #[cfg(not(target_os = "macos"))]
    {
        use std::backtrace::Backtrace;

        // Print current thread backtrace as fallback
        let current = thread::current();
        eprintln!("--- Signal Handler Thread ---");
        eprintln!("Name: {:?}", current.name().unwrap_or("<unnamed>"));
        eprintln!("ID: {:?}", current.id());
        let bt = Backtrace::force_capture();
        eprintln!("Backtrace:\n{bt}");
        eprintln!();

        eprintln!("Note: For full thread backtraces on Linux, use:");
        eprintln!("  gdb -p {pid} -ex 'thread apply all bt' -ex 'quit'");
    }

    eprintln!("{}", "=".repeat(80));
    eprintln!("END THREAD DUMP");
    eprintln!("{}\n", "=".repeat(80));
}

/// Run macOS `sample` command to capture all thread backtraces.
#[cfg(target_os = "macos")]
fn run_sample_command(pid: u32) {
    use std::time::Duration;

    // IMPORTANT: This must NEVER hang. It's used for debugging stuck processes.
    //
    // On some macOS installs, `sample` can itself hang (eg if the kernel is under
    // extreme pressure or a driver is wedged). If we block here, we make the
    // original hang harder to diagnose on unattended machines.
    const SAMPLE_TIMEOUT_SECS: u64 = 5;
    const SAMPLE_KILL_GRACE_MS: u64 = 500;

    // Run sample for 1 second to get a snapshot (not a profile).
    // Note: We use `-mayDie` to avoid requiring elevated permissions.
    //
    // IMPORTANT: Do NOT pipe stdout here. `sample` can emit a lot of output; if we
    // pipe it and don't continuously drain the pipe, the child can block forever
    // once the buffer fills. That manifests exactly as "sample did not exit".
    let out_path = std::env::temp_dir().join(format!(
        "neat_ai_discovery.sample.{pid}.{}.txt",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis())
    ));
    let out_path_str = out_path.to_string_lossy().to_string();

    let sample_program = crate::config::sample_program();
    let sample_args = vec![
        pid.to_string(),
        "1".to_string(),
        "-mayDie".to_string(),
        "-file".to_string(),
        out_path_str.clone(),
    ];

    let run = match run_external_command_with_timeout(
        &sample_program,
        &sample_args,
        Duration::from_secs(SAMPLE_TIMEOUT_SECS),
        Duration::from_millis(SAMPLE_KILL_GRACE_MS),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("Failed to run 'sample': {e}");
            eprintln!("Try manually: sample {pid} 1 -mayDie -file /tmp/sample.txt");
            return;
        }
    };

    if run.timed_out {
        eprintln!(
            "[NEAT-AI-Discovery][debug] WARNING: 'sample' did not exit within {SAMPLE_TIMEOUT_SECS}s. \
             Attempting to print any partial output captured so far."
        );
        match std::fs::read_to_string(&out_path) {
            Ok(contents) => {
                eprintln!(
                    "[NEAT-AI-Discovery][debug] Partial 'sample' output saved to: {out_path_str}\n"
                );
                print_filtered_sample_output(&contents);
            }
            Err(e) => {
                eprintln!(
                    "[NEAT-AI-Discovery][debug] WARNING: Could not read partial 'sample' output file: {e}"
                );
                eprintln!("Expected output at: {out_path_str}");
                eprintln!("Try manually: sample {pid} 1 -mayDie -file /tmp/sample.txt");
            }
        }
        return;
    }

    let Some(status) = run.status else {
        // We only return `None` status when the child was killed but didn't report status
        // within the grace window.
        eprintln!(
            "[NEAT-AI-Discovery][debug] WARNING: 'sample' did not report an exit status. \
             See partial output at: {out_path_str}"
        );
        return;
    };

    if status.success() {
        match std::fs::read_to_string(&out_path) {
            Ok(contents) => {
                eprintln!(
                    "[NEAT-AI-Discovery][debug] Full 'sample' output saved to: {out_path_str}\n"
                );
                print_filtered_sample_output(&contents);
            }
            Err(e) => {
                eprintln!(
                    "[NEAT-AI-Discovery][debug] 'sample' exited successfully but output file could not be read: {e}"
                );
                eprintln!("Expected output at: {out_path_str}");
                eprintln!("Try manually: sample {pid} 1 -mayDie -file /tmp/sample.txt");
            }
        }
    } else {
        eprintln!("'sample' command failed (exit code: {status}).");
        eprintln!("Try manually: sample {pid} 1 -mayDie -file /tmp/sample.txt");
    }
}

// This helper is only required on macOS (to run `sample`) and in tests (to prevent regressions).
// On other platforms we intentionally do not compile it to avoid `-D dead-code` failures under
// the `./quality.sh` gate.
#[cfg(any(target_os = "macos", test))]
#[derive(Debug, Clone, Copy)]
struct ExternalCommandRun {
    status: Option<std::process::ExitStatus>,
    timed_out: bool,
}

/// Run an external command with a hard timeout, avoiding pipe backpressure.
///
/// IMPORTANT: This helper must never hang. It intentionally discards stdout/stderr
/// to avoid deadlocks when a child writes more than a pipe buffer and the parent
/// isn't continuously draining it.
#[cfg(any(target_os = "macos", test))]
fn run_external_command_with_timeout(
    program: &str,
    args: &[String],
    timeout: Duration,
    kill_grace: Duration,
) -> std::io::Result<ExternalCommandRun> {
    use std::process::{Command, Stdio};
    use std::time::Instant;

    let mut child = Command::new(program)
        .args(args)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                return Ok(ExternalCommandRun {
                    status: Some(status),
                    timed_out: false,
                });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();

                    // Never block indefinitely waiting for the child to die.
                    let kill_start = Instant::now();
                    while kill_start.elapsed() < kill_grace {
                        match child.try_wait() {
                            Ok(Some(status)) => {
                                return Ok(ExternalCommandRun {
                                    status: Some(status),
                                    timed_out: true,
                                });
                            }
                            Ok(None) => thread::sleep(Duration::from_millis(25)),
                            Err(_) => break,
                        }
                    }

                    return Ok(ExternalCommandRun {
                        status: None,
                        timed_out: true,
                    });
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(e) => return Err(e),
        }
    }
}

/// Filter sample output to show the most relevant thread information.
#[cfg(target_os = "macos")]
fn print_filtered_sample_output(output: &str) {
    let mut in_call_graph = false;
    let mut thread_count = 0;

    for line in output.lines() {
        // Start of call graph section
        if line.starts_with("Call graph:") {
            in_call_graph = true;
            eprintln!("{line}");
            continue;
        }

        // End markers
        if line.starts_with("Total number in stack") || line.starts_with("Binary Images:") {
            if in_call_graph {
                eprintln!("\n--- End of call graph ({thread_count} threads) ---\n");
            }
            in_call_graph = false;
            continue;
        }

        if in_call_graph {
            // Thread headers
            if line.contains("Thread_") {
                thread_count += 1;
                eprintln!("\n{line}");
            }
            // Show lines containing our library or interesting keywords
            else if line.contains("neat_ai_discovery")
                || line.contains("wgpu")
                || line.contains("metal")
                || line.contains("Metal")
                || line.contains("crossbeam")
                || line.contains("rayon")
                || line.contains("recv")
                || line.contains("poll")
                || line.contains("wait")
                || line.contains("park")
                || line.contains("sleep")
                || line.contains("pthread_cond")
                || line.contains("kevent")
            {
                eprintln!("{line}");
            }
        }
    }
}

/// Simple timestamp without heavy dependencies.
fn chrono_lite_timestamp() -> String {
    use std::time::SystemTime;

    match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
        Ok(d) => {
            let secs = d.as_secs();
            // Simple UTC timestamp calculation
            let days = secs / 86400;
            let remaining = secs % 86400;
            let hours = remaining / 3600;
            let mins = (remaining % 3600) / 60;
            let secs = remaining % 60;

            // Days since 1970-01-01 to approximate date
            // This is a rough approximation, not accounting for leap years properly
            let years = 1970 + days / 365;
            let day_of_year = days % 365;

            format!("{years}-day{day_of_year:03} {hours:02}:{mins:02}:{secs:02} UTC")
        }
        Err(_) => "unknown time".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_init_debug_handlers_is_idempotent() {
        // Calling multiple times should not panic
        init_debug_handlers();
        init_debug_handlers();
        init_debug_handlers();
        // If we get here without panic, the test passes
    }

    #[test]
    fn test_chrono_lite_timestamp_returns_string() {
        let ts = chrono_lite_timestamp();
        assert!(!ts.is_empty());
        assert!(ts.contains("UTC") || ts.contains("unknown"));
    }

    #[test]
    fn test_deadlock_check_interval_is_short() {
        // The deadlock check interval should be short enough for fast detection
        // but not so short as to waste CPU cycles.
        let interval = DEADLOCK_CHECK_INTERVAL_SECS;
        assert!(
            interval <= 5,
            "Deadlock check interval should be at most 5 seconds for fast detection, got {interval}"
        );
        assert!(
            interval >= 1,
            "Deadlock check interval should be at least 1 second to avoid wasting CPU, got {interval}"
        );
    }

    #[test]
    fn test_no_deadlock_when_clean() {
        // Should not detect deadlocks in a clean state
        let deadlocks = parking_lot::deadlock::check_deadlock();
        assert!(
            deadlocks.is_empty(),
            "No deadlocks should exist in clean test"
        );
    }

    #[test]
    fn test_shutdown_is_idempotent() {
        // Calling shutdown multiple times should not panic, even without init.
        shutdown_debug_handlers();
        shutdown_debug_handlers();
    }

    #[test]
    fn test_shutdown_stops_deadlock_detector() {
        // Verify the shutdown flag is respected by the deadlock detector loop.
        SHUTDOWN_REQUESTED.store(false, Ordering::Relaxed);
        assert!(!SHUTDOWN_REQUESTED.load(Ordering::Relaxed));

        SHUTDOWN_REQUESTED.store(true, Ordering::Relaxed);
        assert!(SHUTDOWN_REQUESTED.load(Ordering::Relaxed));

        // Reset for other tests.
        SHUTDOWN_REQUESTED.store(false, Ordering::Relaxed);
    }

    #[test]
    #[cfg(unix)]
    fn run_external_command_with_timeout_does_not_hang_and_preserves_partial_output() {
        use std::os::unix::fs::PermissionsExt;

        // Create a small shell script that writes to a file immediately, then hangs.
        // This simulates the "child doesn't exit promptly" scenario while ensuring we
        // can still read partial output from the file after we kill it.
        let tmp = std::env::temp_dir();
        let script_path = tmp.join(format!(
            "neat_ai_discovery_test_hang_{}_{}.sh",
            std::process::id(),
            chrono_lite_timestamp().replace(' ', "_")
        ));
        let out_path = tmp.join(format!(
            "neat_ai_discovery_test_output_{}_{}.txt",
            std::process::id(),
            chrono_lite_timestamp().replace(' ', "_")
        ));
        // Pre-create the output file so the test can't fail with "not found" if the
        // child is killed before it gets scheduled.
        std::fs::write(&out_path, "").expect("precreate output file");

        // The script writes a sentinel line, then hangs on `sleep`.
        // Using a single `echo` keeps I/O minimal so even slow runners flush before
        // the timeout fires.
        let script = "#!/bin/sh\n\
             echo \"Call graph:\" > \"$1\"\n\
             echo \"Thread_0\" >> \"$1\"\n\
             # Signal that output is ready via a separate sentinel file.\n\
             touch \"$1.ready\"\n\
             sleep 60\n"
            .to_string();
        std::fs::write(&script_path, script).expect("write test script");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("chmod");

        // Spawn the child ourselves first so we can wait for it to finish writing
        // before we exercise the timeout/kill logic in `run_external_command_with_timeout`.
        let args = vec![out_path.to_string_lossy().to_string()];
        {
            use std::process::{Command, Stdio};
            let mut child = Command::new(script_path.to_string_lossy().as_ref())
                .args(&args)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .expect("spawn pre-run child");

            // Wait for the sentinel file that proves the script flushed its output.
            let sentinel = format!("{}.ready", out_path.display());
            let poll_start = Instant::now();
            while poll_start.elapsed() < Duration::from_secs(10) {
                if std::path::Path::new(&sentinel).exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(25));
            }
            let _ = child.kill();
            let _ = child.wait();
            let _ = std::fs::remove_file(&sentinel);
        }

        // Verify that the pre-run child wrote partial output.
        let contents = std::fs::read_to_string(&out_path).expect("read partial output");
        assert!(
            contents.contains("Call graph:") && contents.contains("Thread_0"),
            "expected partial output, got: {contents:?}"
        );

        // Now exercise the function under test – a fresh invocation that will time out.
        // Reset output file so the second child writes fresh.
        std::fs::write(&out_path, "").expect("reset output file");
        let start = Instant::now();
        let run = run_external_command_with_timeout(
            script_path.to_string_lossy().as_ref(),
            &args,
            Duration::from_secs(3),
            Duration::from_millis(500),
        )
        .expect("run");

        assert!(run.timed_out, "expected timeout");
        // Touch `status` so it doesn't get optimised into "dead code" on non-macOS test builds.
        // (It is used by the macOS `sample` path.)
        let _ = run.status;
        assert!(
            start.elapsed() < Duration::from_secs(8),
            "expected quick return, got {:#?}",
            start.elapsed()
        );

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_file(&out_path);
    }
}
