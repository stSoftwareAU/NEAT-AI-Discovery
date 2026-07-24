//! Integration tests for the within-batch target-failure short-circuit
//! (Issue #1164).
//!
//! These tests cover the orchestration-layer tracker, not the GPU evaluation
//! pipeline (which requires a GPU adapter and a Parquet record fixture).
//! They verify the three acceptance behaviours called out in #1164:
//!   1. First failure for target T → subsequent same-target candidates are
//!      skipped within the same batch.
//!   2. A success for target T does NOT add T to the failure set.
//!   3. Threshold > 1 → only short-circuit after N within-batch failures.

use neat_ai_discovery::analysis::constants::WITHIN_BATCH_TARGET_FAILURE_LIMIT;
use neat_ai_discovery::analysis::within_batch_failures::WithinBatchFailureTracker;

/// Issue #1164 acceptance: first failure for a target → subsequent same-target
/// candidates are short-circuited.
#[test]
fn first_failure_skips_subsequent_same_target_candidates() {
    let tracker = WithinBatchFailureTracker::with_threshold(1);

    // Simulate the first candidate for T failing post-evaluation.
    tracker.record_failure("neuron-1063112866");

    // The next same-target candidate sees the short-circuit.
    assert!(tracker.should_skip("neuron-1063112866"));
    tracker.record_skip();

    // A third candidate for the same target also short-circuits.
    assert!(tracker.should_skip("neuron-1063112866"));
    tracker.record_skip();

    assert_eq!(tracker.skip_count(), 2);
    assert_eq!(tracker.failure_count("neuron-1063112866"), 1);
}

/// Issue #1164 acceptance: a success for target T does not add T to the
/// failure set; subsequent same-target candidates proceed.
#[test]
fn success_does_not_short_circuit_subsequent_candidates() {
    let tracker = WithinBatchFailureTracker::with_threshold(1);

    // No failure recorded for T — a successful candidate does not call
    // record_failure, so the tracker stays empty for T.
    assert!(!tracker.should_skip("T"));
    assert_eq!(tracker.skip_count(), 0);
    assert_eq!(tracker.failed_target_count(), 0);
}

/// Issue #1164 acceptance: threshold > 1 → only short-circuit after N
/// within-batch failures.
#[test]
fn threshold_greater_than_one_only_skips_after_n_failures() {
    let tracker = WithinBatchFailureTracker::with_threshold(3);

    tracker.record_failure("T");
    assert!(!tracker.should_skip("T"));
    tracker.record_failure("T");
    assert!(!tracker.should_skip("T"));

    tracker.record_failure("T");
    assert!(tracker.should_skip("T"));
}

/// Issue #1164: separate targets are tracked independently.
#[test]
fn unrelated_target_not_affected_by_failures_on_other_target() {
    let tracker = WithinBatchFailureTracker::with_threshold(1);

    tracker.record_failure("neuron-1063112866");

    assert!(tracker.should_skip("neuron-1063112866"));
    assert!(
        !tracker.should_skip("neuron-healthy"),
        "an unrelated target must not be short-circuited"
    );
}

/// Issue #1164: compiled default short-circuits as soon as the first failure
/// is recorded for a target.
#[test]
fn compiled_default_short_circuits_after_first_failure() {
    assert_eq!(
        WITHIN_BATCH_TARGET_FAILURE_LIMIT, 1,
        "documented default is 1 failure"
    );

    let tracker = WithinBatchFailureTracker::new();
    assert_eq!(tracker.failure_limit(), 1);

    tracker.record_failure("T");
    assert!(tracker.should_skip("T"));
}

/// Issue #1164: simulate the production failure-cache workload — three
/// same-target add-neuron candidates land in one batch. After the first
/// failure, the remaining two are short-circuited.
#[test]
fn production_batch_short_circuits_remaining_same_target_candidates() {
    let tracker = WithinBatchFailureTracker::with_threshold(1);
    let hot_target = "neuron-1063112866";

    // Three add-neuron candidates targeting the same neuron arrive in one
    // batch. The first one is evaluated and fails; the next two should be
    // short-circuited.
    let mut evaluations = 0u32;
    for candidate_idx in 0..3 {
        if tracker.should_skip(hot_target) {
            tracker.record_skip();
            continue;
        }
        // Simulate evaluation: the first candidate runs and fails.
        evaluations += 1;
        let _ = candidate_idx;
        tracker.record_failure(hot_target);
    }

    assert_eq!(
        evaluations, 1,
        "only the first candidate should reach evaluation; the other two are short-circuited"
    );
    assert_eq!(
        tracker.skip_count(),
        2,
        "two candidates should be short-circuited after the first failure"
    );
}

/// Issue #1164: the diagnostic skip counter accumulates across many recorded
/// skips (used by the orchestration log line so the operator sees the budget
/// being saved).
#[test]
fn skip_count_accumulates_across_record_skip_calls() {
    let tracker = WithinBatchFailureTracker::with_threshold(1);
    tracker.record_failure("A");
    tracker.record_failure("B");

    for _ in 0..5 {
        tracker.record_skip();
    }
    assert_eq!(tracker.skip_count(), 5);
    assert_eq!(tracker.failed_target_count(), 2);
}
