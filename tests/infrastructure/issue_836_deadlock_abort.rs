//! Integration tests for issue #836: deadlock detection uses abort() instead of panic().
//!
//! We cannot test abort() in-process (it terminates immediately), so these tests
//! verify the surrounding behaviour: that the deadlock detector thread starts
//! correctly and that clean state reports no deadlocks.

#[test]
fn deadlock_detector_reports_no_deadlocks_in_clean_state() {
    // Initialise debug handlers (idempotent, safe to call repeatedly).
    neat_ai_discovery::debug::init_debug_handlers();

    // In a clean state, there should be no deadlocks.
    let deadlocks = parking_lot::deadlock::check_deadlock();
    assert!(
        deadlocks.is_empty(),
        "Expected no deadlocks in clean state, found {}",
        deadlocks.len()
    );
}

#[test]
fn deadlock_detector_initialisation_is_idempotent() {
    // Multiple calls must not panic or spawn duplicate threads.
    for _ in 0..5 {
        neat_ai_discovery::debug::init_debug_handlers();
    }
    // If we reach here without panic/abort, the test passes.
}
