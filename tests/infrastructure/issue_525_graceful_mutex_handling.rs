//! Issue #525 / #833: Verify that mutex helpers work correctly with parking_lot::Mutex.
//!
//! parking_lot::Mutex does not poison on thread panic, so lock_or_bail and
//! into_inner_or_bail always succeed. These tests confirm that behaviour and
//! verify the helpers remain usable after a thread panics while holding the lock.

use parking_lot::Mutex;
use std::sync::Arc;

/// Verify that `lock_or_bail` succeeds for a healthy mutex.
#[test]
fn test_healthy_mutex_returns_guard() {
    let mutex = Mutex::new(99_i32);
    let result = neat_ai_discovery::analysis::utils::lock_or_bail(&mutex, "healthy mutex");
    assert!(
        result.is_ok(),
        "lock_or_bail should succeed for a healthy mutex"
    );
    assert_eq!(*result.unwrap(), 99);
}

/// Verify that `into_inner_or_bail` works normally for a healthy mutex.
#[test]
fn test_healthy_mutex_into_inner_returns_value() {
    let mutex = Mutex::new(vec![10, 20, 30]);
    let result = neat_ai_discovery::analysis::utils::into_inner_or_bail(mutex, "healthy vec mutex");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![10, 20, 30]);
}

/// Verify that parking_lot::Mutex remains usable after a thread panics while
/// holding the lock (no poisoning). This is the key behavioural difference
/// from std::sync::Mutex that issue #833 migrates to.
#[test]
fn test_mutex_usable_after_thread_panic() {
    let mutex = Arc::new(Mutex::new(42_i32));

    // Panic inside a thread while holding the lock
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::thread::spawn(move || {
        let _guard = mutex_clone.lock();
        panic!("intentional panic to test non-poisoning behaviour");
    })
    .join();

    // parking_lot::Mutex does not poison — the lock should succeed
    let result = neat_ai_discovery::analysis::utils::lock_or_bail(&mutex, "post-panic mutex");
    assert!(
        result.is_ok(),
        "parking_lot::Mutex should remain usable after a thread panic (no poisoning)"
    );
    assert_eq!(*result.unwrap(), 42);
}

/// Verify that into_inner_or_bail succeeds after a thread panic.
#[test]
fn test_into_inner_after_thread_panic() {
    let mutex = Arc::new(Mutex::new(vec![1, 2, 3]));

    // Panic inside a thread while holding the lock
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::thread::spawn(move || {
        let _guard = mutex_clone.lock();
        panic!("intentional panic to test non-poisoning behaviour");
    })
    .join();

    // into_inner should succeed since parking_lot does not poison
    let owned = Arc::try_unwrap(mutex).unwrap();
    let result = neat_ai_discovery::analysis::utils::into_inner_or_bail(owned, "post-panic vec");
    assert!(
        result.is_ok(),
        "into_inner_or_bail should succeed after a thread panic with parking_lot"
    );
    assert_eq!(result.unwrap(), vec![1, 2, 3]);
}
