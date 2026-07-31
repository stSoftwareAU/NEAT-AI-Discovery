//! Poisoned-lock recovery at the drought-reset sites (Issue #1875).
//!
//! The orchestrator's drought-reset call sites used to guard the global
//! target-failure tracker with `if let Ok(mut guard) = ...lock()` and no else
//! arm, so a poisoned mutex silently skipped the operator escape hatch with
//! zero log output — precisely the "fail silently" outcome the drought-reset
//! design (Issue #1794) forbids. The lock now recovers poison via
//! `PoisonError::into_inner`, matching the convention already documented in
//! `target_pass_outcomes.rs`.
//!
//! These tests poison a local tracker mutex and assert the reset, the re-arm,
//! and the diagnostic snapshot all still do their work.

use std::sync::Mutex;

use neat_ai_discovery::analysis::drought_reset::{
    maybe_perform_drought_reset_locked, rearm_drought_reset_locked,
};
use neat_ai_discovery::analysis::target_failure_tracker::{TargetFailureTracker, snapshot_tracker};

/// Two targets in cooldown at epoch 1, one below threshold.
fn populated_tracker() -> TargetFailureTracker {
    let mut tracker = TargetFailureTracker::with_thresholds(2, 100);
    tracker.record_failure("A", 0);
    tracker.record_failure("A", 1);
    tracker.record_failure("B", 0);
    tracker.record_failure("B", 1);
    tracker.record_failure("C", 0);
    tracker
}

/// Poison `mutex` by panicking inside a thread that holds its guard.
fn poison(mutex: &'static Mutex<TargetFailureTracker>) {
    let result = std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let _guard = mutex.lock().expect("first lock must succeed");
                panic!("deliberate poison for Issue #1875");
            })
            .join()
    });
    assert!(result.is_err(), "the poisoning thread must have panicked");
    assert!(mutex.is_poisoned(), "the mutex must now be poisoned");
}

/// Leak a tracker mutex so the poisoning helper can borrow it for `'static`.
fn leaked(tracker: TargetFailureTracker) -> &'static Mutex<TargetFailureTracker> {
    Box::leak(Box::new(Mutex::new(tracker)))
}

#[test]
fn drought_reset_clears_cooldowns_through_a_poisoned_lock() {
    let tracker = leaked(populated_tracker());
    poison(tracker);

    let outcome = maybe_perform_drought_reset_locked(tracker, 10, 5)
        .expect("a poisoned lock must not skip the escape hatch");

    assert_eq!(
        outcome.target_cooldown_cleared, 2,
        "both cooldown entries must be cleared despite the poisoned lock"
    );
    let guard = snapshot_tracker(tracker);
    assert!(
        guard.state("A").is_none(),
        "target A must have been dropped from cooldown"
    );
    assert!(
        guard.state("C").is_some(),
        "below-threshold tracking must be preserved"
    );
}

#[test]
fn drought_reset_still_honours_thresholds_through_a_poisoned_lock() {
    let tracker = leaked(populated_tracker());
    poison(tracker);

    assert!(
        maybe_perform_drought_reset_locked(tracker, 2, 5).is_none(),
        "poison recovery must not bypass the streak threshold"
    );
    assert!(
        snapshot_tracker(tracker).state("A").is_some(),
        "no reset means the cooldown entry survives"
    );
}

#[test]
fn rearm_clears_the_tombstone_through_a_poisoned_lock() {
    let tracker = leaked(populated_tracker());

    // Fire once so a tombstone is stamped, then poison the lock.
    maybe_perform_drought_reset_locked(tracker, 10, 5).expect("first reset must fire");
    assert!(
        snapshot_tracker(tracker)
            .drought_reset_tombstone()
            .is_some(),
        "an effective reset stamps the one-shot tombstone"
    );
    poison(tracker);

    rearm_drought_reset_locked(tracker);

    assert!(
        snapshot_tracker(tracker)
            .drought_reset_tombstone()
            .is_none(),
        "a poisoned lock must not skip the re-arm"
    );
}

#[test]
fn snapshot_returns_tracker_state_through_a_poisoned_lock() {
    let tracker = leaked(populated_tracker());
    poison(tracker);

    let snapshot = snapshot_tracker(tracker);

    assert!(
        snapshot.state("A").is_some(),
        "the diagnostic snapshot must carry real tracker state, not an empty fallback"
    );
    assert_eq!(
        snapshot.current_epoch(),
        0,
        "the snapshot preserves the tracker's epoch counter"
    );
}
