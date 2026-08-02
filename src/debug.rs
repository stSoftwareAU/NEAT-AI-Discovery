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
//! # Degrading, never disappearing (Issue #1934)
//!
//! A thread dump is only useful if it appears when the process is stuck — and
//! that is exactly when the external sampler is least likely to answer. The dump
//! is therefore built in two parts, and the banner names which of them survived:
//!
//! ```text
//!  ┌──────────────────────────┐
//!  │ external sampler (macOS) │──▶ full output    ──▶ "full dump"
//!  │ bounded, killable        │──▶ partial output ──▶ "partial dump"
//!  └──────────────────────────┘──▶ nothing        ──▶ "no backtraces captured"
//!  ┌──────────────────────────┐
//!  │ in-process state block   │──▶ always printed: PID, elapsed run time,
//!  │ (no external tools)      │    breaker state, abandoned threads, last
//!  └──────────────────────────┘    heartbeat, outstanding GPU requests
//! ```
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

mod process_state;
mod sample_capture;

use parking_lot::Mutex;
use std::fmt::Write as _;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

/// Hard cap on how long the external sampler may run before it is killed.
///
/// Re-exported so callers (and the regression tests) can bound how long a
/// SIGUSR1 dump may take.
pub const SAMPLE_TIMEOUT_SECS: u64 = sample_capture::SAMPLE_TIMEOUT_SECS;

/// Grace period allowed for a killed sampler to actually exit.
pub const SAMPLE_KILL_GRACE_MS: u64 = sample_capture::SAMPLE_KILL_GRACE_MS;

/// How much of a thread dump was actually captured (Issue #1934).
///
/// An empty dump used to be indistinguishable from a clean one. The banner now
/// carries this classification so a field report can be grepped for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DumpCompleteness {
    /// The sampler ran to completion and its output was read.
    Full,
    /// The sampler was killed, but partial output was recovered.
    Partial,
    /// No backtraces at all — only the in-process state block.
    NoBacktraces,
}

impl DumpCompleteness {
    /// The wording used in the END THREAD DUMP banner.
    #[must_use]
    pub const fn banner(self) -> &'static str {
        match self {
            Self::Full => "full dump",
            Self::Partial => "partial dump",
            Self::NoBacktraces => "no backtraces captured",
        }
    }
}

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

        // Anchor the elapsed-run-time line in thread dumps (Issue #1934).
        process_state::mark_start();

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

/// Dump thread state to stderr.
///
/// This is called when SIGUSR1 (kill -USR1) is received.
fn dump_all_threads() {
    eprint!("{}", render_thread_dump());
    let _ = std::io::Write::flush(&mut std::io::stderr());
}

/// Build the full thread dump as text (Issue #1934).
///
/// This is what SIGUSR1 writes to stderr. It is exposed so the fallback paths
/// can be exercised directly by tests, and so a host can capture a dump without
/// signalling itself.
///
/// # Guarantees
///
/// - It never blocks for longer than [`SAMPLE_TIMEOUT_SECS`] plus
///   [`SAMPLE_KILL_GRACE_MS`], whatever the external sampler does.
/// - It always contains the in-process state block, so an unhelpful dump is
///   never an empty one.
/// - The END banner always names the [`DumpCompleteness`], so an empty capture
///   cannot be mistaken for a clean one.
#[must_use]
pub fn render_thread_dump() -> String {
    let timestamp = chrono_lite_timestamp();
    let pid = std::process::id();
    let mut out = String::new();
    let rule = "=".repeat(80);

    let _ = writeln!(out, "\n{rule}");
    let _ = writeln!(out, "THREAD DUMP - {timestamp} (kill -USR1 received)");
    let _ = writeln!(out, "Process ID: {pid}");
    let _ = writeln!(out, "{rule}\n");

    render_deadlocks(&mut out);

    // Best-effort full backtraces. Bounded, killable, and allowed to fail.
    let completeness = sample_capture::capture(pid, &mut out);

    // Always available, never dependent on a wedged driver or external tool.
    let _ = writeln!(out);
    process_state::render(&mut out, pid);

    let _ = writeln!(out, "{rule}");
    let _ = writeln!(out, "END THREAD DUMP - {}", completeness.banner());
    let _ = writeln!(out, "{rule}\n");

    out
}

/// Report `parking_lot` deadlock cycles — fast, and purely in-process.
fn render_deadlocks(out: &mut String) {
    let deadlocks = parking_lot::deadlock::check_deadlock();
    if deadlocks.is_empty() {
        let _ = writeln!(out, "--- No mutex deadlocks detected ---\n");
        return;
    }

    let _ = writeln!(out, "--- DEADLOCKED THREADS DETECTED ---");
    for (i, threads) in deadlocks.iter().enumerate() {
        let _ = writeln!(out, "\nDeadlock #{} ({} threads):", i + 1, threads.len());
        for t in threads {
            let _ = writeln!(out, "  Thread ID: {:?}", t.thread_id());
            if VERBOSE_MODE.load(Ordering::Relaxed) {
                let _ = writeln!(out, "  Backtrace:\n{:#?}", t.backtrace());
            }
        }
    }
    let _ = writeln!(out);
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

    // =========================================================================
    // Issue #1934 — the dump degrades instead of disappearing
    // =========================================================================

    /// Each completeness level has its own banner wording, so a field report can
    /// be grepped for the one that matters.
    #[test]
    fn banner_wording_distinguishes_the_three_outcomes() {
        assert_eq!(DumpCompleteness::Full.banner(), "full dump");
        assert_eq!(DumpCompleteness::Partial.banner(), "partial dump");
        assert_eq!(
            DumpCompleteness::NoBacktraces.banner(),
            "no backtraces captured"
        );
    }

    /// The dump always carries the in-process state block and a classified
    /// banner — never a bare END THREAD DUMP with nothing above it.
    #[test]
    fn a_dump_always_contains_state_and_a_classified_banner() {
        let dump = render_thread_dump();

        assert!(dump.contains("THREAD DUMP"), "{dump}");
        assert!(dump.contains("In-process state"), "{dump}");
        assert!(dump.contains("GPU circuit breaker:"), "{dump}");
        assert!(dump.contains("Outstanding GPU requests"), "{dump}");
        assert!(
            dump.contains("END THREAD DUMP - full dump")
                || dump.contains("END THREAD DUMP - partial dump")
                || dump.contains("END THREAD DUMP - no backtraces captured"),
            "the END banner must classify the dump: {dump}"
        );
    }

    /// The handler must stay bounded even when the sampler misbehaves.
    #[test]
    fn a_dump_returns_within_the_sampler_bound() {
        let started = Instant::now();
        let _ = render_thread_dump();
        let bound = Duration::from_secs(SAMPLE_TIMEOUT_SECS)
            + Duration::from_millis(SAMPLE_KILL_GRACE_MS)
            + Duration::from_secs(5);
        assert!(
            started.elapsed() < bound,
            "a dump must never block the process, took {:?}",
            started.elapsed()
        );
    }
}
