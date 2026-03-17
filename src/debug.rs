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
//! # Example
//!
//! ```rust,ignore
//! // At startup
//! neat_ai_discovery::debug::init_debug_handlers();
//!
//! // Now you can send kill -USR1 <pid> to dump threads
//! // And deadlocks will be detected and abort the process
//! ```

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Static flag to ensure handlers are only initialised once.
static DEBUG_HANDLERS_INITIALISED: OnceLock<()> = OnceLock::new();

/// Flag to track if we're in verbose mode (shows more detail).
static VERBOSE_MODE: AtomicBool = AtomicBool::new(false);

/// Interval between deadlock checks (in seconds).
///
/// Kept short (5 s) so that deadlocks are detected quickly without adding
/// meaningful overhead — the check itself is very cheap.
const DEADLOCK_CHECK_INTERVAL_SECS: u64 = 5;

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
        // Check if verbose mode is enabled
        if crate::config::verbose() {
            VERBOSE_MODE.store(true, Ordering::Relaxed);
        }

        // Start deadlock detection thread
        start_deadlock_detector();

        // Install signal handler (Unix only)
        #[cfg(unix)]
        install_signal_handler();

        tracing::debug!("debug handlers initialised (deadlock detection + kill -USR1 thread dump)");
    });
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
fn start_deadlock_detector() {
    thread::Builder::new()
        .name("deadlock-detector".to_string())
        .spawn(move || {
            loop {
                thread::sleep(Duration::from_secs(DEADLOCK_CHECK_INTERVAL_SECS));

                let deadlocks = parking_lot::deadlock::check_deadlock();
                if deadlocks.is_empty() {
                    continue;
                }

                // Deadlock detected! Print details and panic.
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
        })
        .expect("Failed to spawn deadlock detector thread");
}

/// Install signal handler for SIGUSR1 (kill -USR1) to dump thread backtraces.
///
/// SIGUSR1 is a user-defined signal with no default action, making it safe
/// to use for diagnostics without risk of terminating the process.
#[cfg(unix)]
fn install_signal_handler() {
    use signal_hook::consts::SIGUSR1;
    use signal_hook::iterator::Signals;

    thread::Builder::new()
        .name("signal-handler".to_string())
        .spawn(move || {
            let mut signals = match Signals::new([SIGUSR1]) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("failed to install SIGUSR1 handler: {e}");
                    return;
                }
            };

            for _sig in signals.forever() {
                dump_all_threads();
            }
        })
        .expect("Failed to spawn signal handler thread");
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
            .map(|d| d.as_millis())
            .unwrap_or(0)
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

        let script = "#!/bin/sh\n\
             OUT=\"$1\"\n\
             # Write output immediately (and plenty of it) so the test can reliably\n\
             # observe partial output even if we kill the process shortly after.\n\
             echo \"Call graph:\" > \"$OUT\"\n\
             i=0\n\
             while [ $i -lt 2000 ]; do\n\
               echo \"Thread_0\" >> \"$OUT\"\n\
               i=$((i+1))\n\
             done\n\
             # hang long enough that the test timeout must kill us\n\
             sleep 60\n"
            .to_string();
        std::fs::write(&script_path, script).expect("write test script");
        let mut perms = std::fs::metadata(&script_path)
            .expect("metadata")
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&script_path, perms).expect("chmod");

        let start = Instant::now();
        let args = vec![out_path.to_string_lossy().to_string()];
        let run = run_external_command_with_timeout(
            script_path.to_string_lossy().as_ref(),
            &args,
            Duration::from_millis(750),
            Duration::from_millis(250),
        )
        .expect("run");

        assert!(run.timed_out, "expected timeout");
        // Touch `status` so it doesn't get optimised into "dead code" on non-macOS test builds.
        // (It is used by the macOS `sample` path.)
        let _ = run.status;
        assert!(
            start.elapsed() < Duration::from_secs(2),
            "expected quick return, got {:#?}",
            start.elapsed()
        );

        // Allow a short window for the child to be scheduled and write its initial output.
        // (On heavily loaded CI runners, immediate scheduling isn't guaranteed.)
        let mut contents = String::new();
        let poll_start = Instant::now();
        while poll_start.elapsed() < Duration::from_secs(1) {
            contents = std::fs::read_to_string(&out_path).expect("read partial output");
            if !contents.is_empty() {
                break;
            }
            thread::sleep(Duration::from_millis(25));
        }
        assert!(
            contents.contains("Call graph:") && contents.contains("Thread_0"),
            "expected partial output, got: {contents:?}"
        );

        let _ = std::fs::remove_file(&script_path);
        let _ = std::fs::remove_file(&out_path);
    }
}
