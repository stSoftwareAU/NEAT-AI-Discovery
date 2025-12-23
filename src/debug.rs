//! Debug utilities for diagnosing deadlocks and thread issues.
//!
//! This module provides:
//! - **Deadlock detection**: Using `parking_lot`'s deadlock detection feature
//! - **Thread dump on signal**: SIGUSR1 (kill -USR1) dumps all thread backtraces
//!
//! # Usage
//!
//! Call `init_debug_handlers()` early in your program to enable both features.
//! On deadlock, the program will panic with backtrace information.
//! On SIGUSR1 (kill -USR1), thread backtraces are printed to stderr without exiting.
//!
//! # Example
//!
//! ```rust,ignore
//! // At startup
//! neat_ai_discovery::debug::init_debug_handlers();
//!
//! // Now you can send kill -USR1 <pid> to dump threads
//! // And deadlocks will be detected and panic
//! ```

use once_cell::sync::OnceCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Static flag to ensure handlers are only initialised once.
static DEBUG_HANDLERS_INITIALISED: OnceCell<()> = OnceCell::new();

/// Flag to track if we're in verbose mode (shows more detail).
static VERBOSE_MODE: AtomicBool = AtomicBool::new(false);

/// Interval between deadlock checks (in seconds).
const DEADLOCK_CHECK_INTERVAL_SECS: u64 = 10;

/// Initialise debug handlers for deadlock detection and signal-based thread dumps.
///
/// This function is idempotent - calling it multiple times has no effect.
///
/// # Features
///
/// 1. **Deadlock Detection**: A background thread checks for deadlocks every 10 seconds.
///    If a deadlock is detected, the program panics with full backtrace information.
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
        if std::env::var("NEAT_AI_DISCOVERY_VERBOSE").is_ok() {
            VERBOSE_MODE.store(true, Ordering::Relaxed);
        }

        // Start deadlock detection thread
        start_deadlock_detector();

        // Install signal handler (Unix only)
        #[cfg(unix)]
        install_signal_handler();

        eprintln!(
            "[NEAT-AI-Discovery][debug] Debug handlers initialised (deadlock detection + kill -USR1 thread dump)"
        );
    });
}

/// Start background thread for deadlock detection.
///
/// Uses `parking_lot::deadlock::check_deadlock()` to detect deadlocks.
/// When a deadlock is found, prints full information and panics.
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
                eprintln!("PANICKING due to deadlock. See above for thread backtraces.");
                eprintln!("{}\n", "=".repeat(80));

                // Panic to abort the program - this allows the error to propagate
                panic!(
                    "Deadlock detected! {} deadlock(s) involving {} total threads. \
                     See stderr for full backtrace information.",
                    deadlocks.len(),
                    deadlocks.iter().map(|d| d.len()).sum::<usize>()
                );
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
                    eprintln!("[NEAT-AI-Discovery][debug] Failed to install SIGUSR1 handler: {e}");
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
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    // IMPORTANT: This must NEVER hang. It's used for debugging stuck processes.
    //
    // On some macOS installs, `sample` can itself hang (eg if the kernel is under
    // extreme pressure or a driver is wedged). If we block here, we make the
    // original hang harder to diagnose on unattended machines.
    const SAMPLE_TIMEOUT_SECS: u64 = 5;

    // Run sample for 1 second to get a snapshot (not a profile).
    // Note: We use `-mayDie` to avoid requiring elevated permissions.
    let mut child = match Command::new("sample")
        .args([&pid.to_string(), "1", "-mayDie"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(e) => {
            eprintln!("Failed to run 'sample': {e}");
            eprintln!("Try manually: sample {pid} 1 -file /tmp/sample.txt");
            return;
        }
    };

    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = Vec::new();
                let mut stderr = Vec::new();

                if let Some(mut out) = child.stdout.take() {
                    let _ = out.read_to_end(&mut stdout);
                }
                if let Some(mut err) = child.stderr.take() {
                    let _ = err.read_to_end(&mut stderr);
                }

                if status.success() {
                    let stdout = String::from_utf8_lossy(&stdout);
                    // Filter to show the most relevant parts.
                    print_filtered_sample_output(&stdout);
                } else {
                    let stderr = String::from_utf8_lossy(&stderr);
                    eprintln!("'sample' command failed: {stderr}");
                    eprintln!("Try manually: sample {pid} 1 -file /tmp/sample.txt");
                }
                return;
            }
            Ok(None) => {
                if start.elapsed() > Duration::from_secs(SAMPLE_TIMEOUT_SECS) {
                    eprintln!(
                        "[NEAT-AI-Discovery][debug] WARNING: 'sample' did not exit within {SAMPLE_TIMEOUT_SECS}s. \
                         Skipping thread dump capture to avoid hanging the process."
                    );
                    let _ = child.kill();
                    let _ = child.wait();
                    return;
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => {
                eprintln!("Failed to run 'sample': {e}");
                eprintln!("Try manually: sample {pid} 1 -file /tmp/sample.txt");
                return;
            }
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
    fn test_no_deadlock_when_clean() {
        // Should not detect deadlocks in a clean state
        let deadlocks = parking_lot::deadlock::check_deadlock();
        assert!(
            deadlocks.is_empty(),
            "No deadlocks should exist in clean test"
        );
    }
}
