//! Issue #525: Replace production panic! calls with proper error handling
//!
//! Tests that mutex lock failures are handled gracefully (returning errors)
//! rather than panicking, which would crash the FFI caller process.

use std::sync::{Arc, Mutex};

/// Verify that `lock_or_bail` returns an error instead of panicking
/// when a mutex is poisoned.
#[test]
fn test_poisoned_mutex_returns_error_instead_of_panic() {
    let mutex = Arc::new(Mutex::new(42_i32));

    // Poison the mutex by panicking inside a lock scope
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::thread::spawn(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("intentional panic to poison the mutex");
    })
    .join();

    // The mutex should now be poisoned
    assert!(
        mutex.lock().is_err(),
        "Mutex should be poisoned after thread panic"
    );

    // Using lock_or_bail should return an Err, not panic
    let result = neat_ai_discovery::analysis::utils::lock_or_bail(&mutex, "test mutex");
    assert!(
        result.is_err(),
        "lock_or_bail should return Err for a poisoned mutex, not panic"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("poisoned"),
        "Error message should mention poisoning: {err_msg}"
    );
}

/// Verify that `lock_or_bail` works normally for a healthy mutex.
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

/// Verify that `into_inner_or_bail` returns an error for a poisoned mutex.
#[test]
fn test_poisoned_mutex_into_inner_returns_error() {
    let mutex = Arc::new(Mutex::new(vec![1, 2, 3]));

    // Poison the mutex
    let mutex_clone = Arc::clone(&mutex);
    let _ = std::thread::spawn(move || {
        let _guard = mutex_clone.lock().unwrap();
        panic!("intentional panic to poison the mutex");
    })
    .join();

    // into_inner_or_bail should return Err, not panic
    let owned = Arc::try_unwrap(mutex).unwrap();
    let result = neat_ai_discovery::analysis::utils::into_inner_or_bail(owned, "test vec mutex");
    assert!(
        result.is_err(),
        "into_inner_or_bail should return Err for a poisoned mutex"
    );
}

/// Verify that `into_inner_or_bail` works normally for a healthy mutex.
#[test]
fn test_healthy_mutex_into_inner_returns_value() {
    let mutex = Mutex::new(vec![10, 20, 30]);
    let result = neat_ai_discovery::analysis::utils::into_inner_or_bail(mutex, "healthy vec mutex");
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), vec![10, 20, 30]);
}
