//! Process hang watchdog.
//!
//! This is designed for unattended discovery workers where "hang forever" is worse than
//! a crash. When enabled, the watchdog monitors a heartbeat and will:
//! - trigger a SIGUSR1 thread dump (if available), then
//! - abort the process so orchestration can capture logs and restart.
//!
//! Configuration is via environment variables so callers can tune per machine:
//! - `NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS` (u64): If set > 0, enables the watchdog and
//!   triggers if no heartbeat occurs for this many seconds.
//! - `NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS` (u64): Delay between requesting a thread
//!   dump and aborting (default: 2 seconds).
//!
//! Notes:
//! - We use `abort()` rather than `panic!()` because the library is commonly used through
//!   Deno FFI entrypoints that wrap calls in `catch_unwind()`. A panic would be caught and
//!   returned as JSON, but would not terminate the worker process.
//! - All comments are written in Australian English.

use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

/// Global watchdog heartbeat state (if watchdog is enabled for the current operation).
static ACTIVE: Lazy<Mutex<Option<Arc<BeatState>>>> = Lazy::new(|| Mutex::new(None));

/// Watchdog configuration loaded from environment variables.
#[derive(Debug, Clone)]
pub(crate) struct WatchdogConfig {
    pub(crate) stall_timeout: Duration,
    pub(crate) abort_delay: Duration,
}

impl WatchdogConfig {
    /// Load watchdog configuration from environment variables.
    ///
    /// Returns `None` when watchdog is disabled.
    pub(crate) fn from_env() -> Option<Self> {
        let stall_secs = std::env::var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        if stall_secs == 0 {
            return None;
        }

        let abort_delay_secs = std::env::var("NEAT_AI_DISCOVERY_WATCHDOG_ABORT_DELAY_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(2);

        Some(Self {
            stall_timeout: Duration::from_secs(stall_secs),
            abort_delay: Duration::from_secs(abort_delay_secs),
        })
    }
}

/// Update the watchdog heartbeat if a watchdog is currently active.
///
/// This is intentionally a no-op when watchdog is disabled, so call sites can keep
/// instrumentation cheap and simple.
pub(crate) fn beat(stage: impl Into<String>) {
    let guard = ACTIVE.lock();
    let Some(state) = guard.as_ref() else {
        return;
    };
    state.beat(stage.into());
}

/// Start a watchdog if enabled via environment variables.
///
/// Returns a handle that must be kept alive for the duration of the operation.
pub(crate) fn start_from_env(initial_stage: &str) -> Option<Watchdog> {
    let config = WatchdogConfig::from_env()?;
    let wd = Watchdog::start(config);
    beat(initial_stage.to_string());
    Some(wd)
}

/// Test-only global lock to prevent parallel tests from racing on the global ACTIVE state.
#[cfg(test)]
static TEST_SERIAL: Lazy<parking_lot::Mutex<()>> = Lazy::new(|| parking_lot::Mutex::new(()));

/// Acquire the test-only serialisation lock (used by tests that touch global watchdog state).
#[cfg(test)]
pub(crate) fn lock_for_test_serialisation() -> parking_lot::MutexGuard<'static, ()> {
    TEST_SERIAL.lock()
}

/// Test-only helper to read the current watchdog stage (if any).
///
/// This is intentionally `pub(crate)` and only compiled for tests so we don't expose
/// watchdog internals as part of the public API.
#[cfg(test)]
pub(crate) fn active_stage_for_test() -> Option<String> {
    let guard = ACTIVE.lock();
    guard.as_ref().map(|state| state.stage.lock().clone())
}

struct BeatState {
    last_beat_ms: AtomicU64,
    stage: Mutex<String>,
}

impl BeatState {
    fn new(initial_stage: &str) -> Self {
        Self {
            last_beat_ms: AtomicU64::new(monotonic_ms()),
            stage: Mutex::new(initial_stage.to_string()),
        }
    }

    fn beat(&self, stage: String) {
        *self.stage.lock() = stage;
        self.last_beat_ms.store(monotonic_ms(), Ordering::Relaxed);
    }
}

/// A handle used to refresh the watchdog heartbeat and provide context.
///
/// If this handle is dropped, the watchdog thread is requested to stop (best effort).
pub(crate) struct Watchdog {
    stop: Arc<AtomicBool>,
    state: Arc<BeatState>,
    _thread: Option<thread::JoinHandle<()>>,
}

impl Watchdog {
    pub(crate) fn start(config: WatchdogConfig) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let state = Arc::new(BeatState::new("initialising"));

        // Make this watchdog globally available for lightweight instrumentation.
        *ACTIVE.lock() = Some(Arc::clone(&state));

        let thread_stop = Arc::clone(&stop);
        let thread_state = Arc::clone(&state);

        let handle = thread::Builder::new()
            .name("hang-watchdog".to_string())
            .spawn(move || watchdog_loop(config, thread_stop, thread_state))
            .ok();

        Self {
            stop,
            state,
            _thread: handle,
        }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        // Clear the active global watchdog (best effort; only if it's the same instance).
        //
        // IMPORTANT: Do not hold the global ACTIVE lock while joining the watchdog thread.
        // The watchdog thread can sleep for up to `poll_every` (currently up to 5 seconds),
        // and other threads calling `beat()` should not be blocked for that duration.
        {
            let mut active = ACTIVE.lock();
            if let Some(current) = active.as_ref()
                && Arc::ptr_eq(current, &self.state)
            {
                *active = None;
            }
        }
        if let Some(handle) = self._thread.take() {
            // Best effort join.
            //
            // Note: This may block until the watchdog thread wakes up and observes `stop`.
            // However, we intentionally avoid holding any global locks while waiting.
            let _ = handle.join();
        }
    }
}

fn watchdog_loop(config: WatchdogConfig, stop: Arc<AtomicBool>, state: Arc<BeatState>) {
    // Poll periodically; we don't need high resolution.
    let poll_every = Duration::from_secs(5).min(config.stall_timeout.max(Duration::from_secs(1)));

    loop {
        if stop.load(Ordering::Relaxed) {
            return;
        }

        let now_ms = monotonic_ms();
        let last_ms = state.last_beat_ms.load(Ordering::Relaxed);

        let elapsed_ms = now_ms.saturating_sub(last_ms);
        if elapsed_ms >= config.stall_timeout.as_millis() as u64 {
            let current_stage = state.stage.lock().clone();

            tracing::error!(
                elapsed_minutes = format_args!("{:.1}", elapsed_ms as f64 / 1000.0 / 60.0),
                stall_timeout_minutes = format_args!("{:.1}", config.stall_timeout.as_secs_f64() / 60.0),
                last_stage = %current_stage,
                abort_delay_secs = format_args!("{:.1}", config.abort_delay.as_secs_f64()),
                "WATCHDOG STALL DETECTED — triggering thread dump then aborting"
            );

            // Trigger a thread dump if supported (Unix). This is best-effort.
            #[cfg(unix)]
            {
                use signal_hook::consts::SIGUSR1;
                let _ = signal_hook::low_level::raise(SIGUSR1);
            }

            thread::sleep(config.abort_delay);

            // Abort the process so orchestration captures logs and restarts.
            //
            // We intentionally do NOT panic here because most FFI entrypoints use
            // `catch_unwind()`, which would swallow a panic and keep the worker alive.
            std::process::abort();
        }

        thread::sleep(poll_every);
    }
}

fn monotonic_ms() -> u64 {
    // `Instant` is monotonic but not directly representable as a duration since epoch.
    // We emulate a monotonic counter using a static start.
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_millis().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic smoke test: config parsing should disable when env var missing.
    #[test]
    fn watchdog_config_disabled_by_default() {
        let _lock = lock_for_test_serialisation();
        // SAFETY: Tests run single-threaded (--test-threads=1), no concurrent env access.
        unsafe {
            std::env::remove_var("NEAT_AI_DISCOVERY_WATCHDOG_STALL_SECS");
        }
        assert!(WatchdogConfig::from_env().is_none());
    }

    /// Test the heartbeat machinery without aborting the process by running the loop logic
    /// with an injected "abort" path.
    ///
    /// We keep this as a unit test to avoid exposing watchdog internals as public API.
    #[test]
    fn watchdog_heartbeat_updates_stage_and_timestamp() {
        let _lock = lock_for_test_serialisation();
        let cfg = WatchdogConfig {
            stall_timeout: Duration::from_secs(60),
            abort_delay: Duration::from_secs(1),
        };

        let wd = Watchdog::start(cfg);
        beat("stage A");
        let a = wd.state.last_beat_ms.load(Ordering::Relaxed);
        assert!(!wd.state.stage.lock().is_empty());

        thread::sleep(Duration::from_millis(10));
        beat("stage B");
        let b = wd.state.last_beat_ms.load(Ordering::Relaxed);
        assert!(b >= a);
        assert_eq!(&*wd.state.stage.lock(), "stage B");
    }

    /// Regression guard: `monotonic_ms` should be non-decreasing.
    #[test]
    fn monotonic_ms_is_monotonic() {
        let _lock = lock_for_test_serialisation();
        let mut last = monotonic_ms();
        for _ in 0..10 {
            let now = monotonic_ms();
            assert!(now >= last);
            last = now;
        }
    }

    /// Regression test: dropping a watchdog must not hold the global ACTIVE lock while it
    /// waits for the watchdog thread to exit.
    ///
    /// Without this, a concurrent `beat()` from another thread can block for up to the
    /// watchdog thread's sleep interval (currently up to 5 seconds), which contradicts the
    /// intent of lightweight instrumentation.
    #[test]
    fn drop_does_not_block_beat_on_active_lock() {
        let _lock = lock_for_test_serialisation();

        let cfg = WatchdogConfig {
            stall_timeout: Duration::from_secs(60),
            abort_delay: Duration::from_secs(1),
        };

        let wd = Watchdog::start(cfg);

        // Give the watchdog thread a chance to start and enter its sleep loop so `join()`
        // is likely to block for a noticeable period.
        thread::sleep(Duration::from_millis(100));

        // Drop the watchdog on another thread so we can call `beat()` concurrently.
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        thread::spawn(move || {
            drop(wd);
            let _ = tx.send(());
        });

        // Small delay to increase the chance that the drop thread has reached `join()`.
        thread::sleep(Duration::from_millis(20));

        let start = std::time::Instant::now();
        beat("concurrent beat during drop");
        let elapsed = start.elapsed();

        // If the ACTIVE lock is incorrectly held across `join()`, this can take ~5 seconds.
        assert!(
            elapsed < Duration::from_millis(500),
            "beat() was blocked for {elapsed:?}; drop() may be holding ACTIVE lock while joining"
        );

        // Ensure the drop thread completed (avoid a detached thread lingering in the test).
        let _ = rx.recv_timeout(Duration::from_secs(10));
    }
}
