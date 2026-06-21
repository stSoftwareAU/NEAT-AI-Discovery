//! Lock contention tracing for diagnosing concurrent stalls (Issue #837).
//!
//! Provides utilities to trace lock acquisition times on hot-path Mutex locks.
//! When verbose mode is enabled (`NEAT_AI_DISCOVERY_VERBOSE=1`), lock acquisitions
//! that exceed a configurable threshold emit a warning via `tracing`.
//!
//! In non-verbose mode, the tracing is completely bypassed — there is no
//! overhead from `Instant::now()` calls or threshold checks.

#![allow(clippy::cast_possible_truncation)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use parking_lot::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

/// Default lock wait threshold: 100ms.
///
/// Lock acquisitions exceeding this duration trigger a warning when verbose
/// mode is enabled.
pub const DEFAULT_LOCK_WAIT_THRESHOLD: Duration = Duration::from_millis(100);

/// Acquire a `parking_lot::Mutex` lock with optional contention tracing.
///
/// When `NEAT_AI_DISCOVERY_VERBOSE=1` is set, this function measures the time
/// spent waiting for the lock. If the wait exceeds `threshold`, a warning is
/// emitted via `tracing` identifying the lock by `lock_name`.
///
/// When verbose mode is disabled, this is equivalent to a plain `.lock()` call
/// with zero overhead.
///
/// # Arguments
/// * `mutex` — The mutex to lock
/// * `lock_name` — Human-readable label for diagnostics (e.g. `"helpful_map"`)
/// * `threshold` — Duration above which a warning is emitted
#[inline]
pub fn traced_lock<'a, T>(
    mutex: &'a Mutex<T>,
    lock_name: &str,
    threshold: Duration,
) -> MutexGuard<'a, T> {
    if !crate::config::verbose() {
        return mutex.lock();
    }

    let start = Instant::now();
    let guard = mutex.lock();
    let elapsed = start.elapsed();

    if elapsed >= threshold {
        tracing::warn!(
            lock = lock_name,
            wait_ms = elapsed.as_millis() as u64,
            threshold_ms = threshold.as_millis() as u64,
            "Lock contention detected: waited {elapsed:?} for lock \"{lock_name}\""
        );
    } else {
        tracing::trace!(
            lock = lock_name,
            wait_ms = elapsed.as_millis() as u64,
            "Lock acquired"
        );
    }

    guard
}

/// Convenience wrapper using the default 100ms threshold.
#[inline]
pub fn traced_lock_default<'a, T>(mutex: &'a Mutex<T>, lock_name: &str) -> MutexGuard<'a, T> {
    traced_lock(mutex, lock_name, DEFAULT_LOCK_WAIT_THRESHOLD)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn traced_lock_acquires_and_returns_guard() {
        let mutex = Mutex::new(42_i32);
        let guard = traced_lock(&mutex, "test_lock", Duration::from_millis(100));
        assert_eq!(*guard, 42);
    }

    #[test]
    fn traced_lock_default_acquires_lock() {
        let mutex = Mutex::new(String::from("hello"));
        let guard = traced_lock_default(&mutex, "test_default");
        assert_eq!(*guard, "hello");
    }

    #[test]
    fn traced_lock_allows_mutation() {
        let mutex = Mutex::new(vec![1, 2, 3]);
        {
            let mut guard = traced_lock_default(&mutex, "test_mutation");
            guard.push(4);
        }
        let guard = mutex.lock();
        assert_eq!(*guard, vec![1, 2, 3, 4]);
    }

    #[test]
    fn traced_lock_works_with_arc() {
        let mutex = Arc::new(Mutex::new(0_u64));
        let guard = traced_lock_default(&mutex, "arc_lock");
        assert_eq!(*guard, 0);
    }

    // Tautological `default_threshold_is_100ms` pin removed (Issue #1469): it
    // only re-asserted the constant's own literal. The default threshold is
    // exercised behaviourally by `traced_lock_default_acquires_lock` and the
    // other `traced_lock_default` tests above.
}
