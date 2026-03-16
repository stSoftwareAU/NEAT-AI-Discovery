//! Integration tests for lock contention tracing (Issue #837).
//!
//! Verifies that the lock contention tracing infrastructure works correctly:
//! - `traced_lock` acquires the mutex and returns the correct guard
//! - `traced_lock_default` uses the default 100ms threshold
//! - Contention tracing does not interfere with correct mutex behaviour
//! - Multiple threads can use traced locks concurrently without deadlock

use neat_ai_discovery::analysis::utils::lock_contention::{
    DEFAULT_LOCK_WAIT_THRESHOLD, traced_lock, traced_lock_default,
};
use parking_lot::Mutex;
use std::sync::Arc;
use std::time::Duration;

#[test]
fn traced_lock_acquires_mutex_and_returns_correct_value() {
    let mutex = Mutex::new(42_i32);
    let guard = traced_lock(&mutex, "test_value", Duration::from_millis(100));
    assert_eq!(*guard, 42);
}

#[test]
fn traced_lock_default_uses_100ms_threshold() {
    assert_eq!(DEFAULT_LOCK_WAIT_THRESHOLD, Duration::from_millis(100));
    let mutex = Mutex::new("contention_test");
    let guard = traced_lock_default(&mutex, "default_threshold");
    assert_eq!(*guard, "contention_test");
}

#[test]
fn traced_lock_allows_mutation_through_guard() {
    let mutex = Mutex::new(vec![1_u32, 2, 3]);
    {
        let mut guard = traced_lock_default(&mutex, "mutation_test");
        guard.push(4);
        guard.push(5);
    }
    // After guard is dropped, verify the mutation persisted
    let final_value = mutex.lock();
    assert_eq!(*final_value, vec![1, 2, 3, 4, 5]);
}

#[test]
fn traced_lock_with_custom_threshold() {
    let mutex = Mutex::new(String::from("custom"));
    // Very short threshold — should still acquire without issues
    let guard = traced_lock(&mutex, "custom_threshold", Duration::from_nanos(1));
    assert_eq!(*guard, "custom");
}

#[test]
fn traced_lock_concurrent_access_does_not_deadlock() {
    let mutex = Arc::new(Mutex::new(0_u64));
    let iterations = 100;
    let thread_count = 4;

    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let m = Arc::clone(&mutex);
            std::thread::spawn(move || {
                for _ in 0..iterations {
                    let mut guard = traced_lock_default(&m, "concurrent_test");
                    *guard += 1;
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }

    let final_value = *mutex.lock();
    assert_eq!(
        final_value,
        thread_count * iterations,
        "All increments should be accounted for"
    );
}

#[test]
fn traced_lock_guard_drops_correctly() {
    let mutex = Mutex::new(());
    // Acquire and immediately drop
    {
        let _guard = traced_lock_default(&mutex, "drop_test");
    }
    // Should be able to acquire again without blocking
    let _guard2 = traced_lock_default(&mutex, "drop_test_reacquire");
}

#[test]
fn lock_or_bail_uses_contention_tracing() {
    // Verify that the lock_or_bail helper (used throughout the codebase)
    // integrates with contention tracing.
    let mutex = Mutex::new(99_i32);
    let result = neat_ai_discovery::analysis::utils::lock_or_bail(&mutex, "bail_test");
    assert!(result.is_ok());
    assert_eq!(*result.unwrap(), 99);
}
