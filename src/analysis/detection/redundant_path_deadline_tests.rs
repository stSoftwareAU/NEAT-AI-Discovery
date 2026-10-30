//! Issue #2374: the redundant-path pair loop must check the analysis deadline
//! before every pair, so a scan that started before the deadline is stopped
//! mid-flight rather than visiting every remaining pair.
//!
//! These in-crate tests use the `deadline_override` test seam (`#[cfg(test)]`
//! only, so they cannot live in the `tests/` integration crate) to script
//! `deadline_passed` to return `false` for the first pair and `true` after,
//! proving the pair-loop check is the one that stops the scan.
//!
//! `#[serial]` keeps these tests from interleaving with any other test that
//! installs a deadline override or flips the global cancellation flag
//! `deadline_passed` also consults.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use super::*;
use crate::analysis::utils::deadline_override::DeadlineOverrideGuard;
use serial_test::serial;
use std::time::{Duration, SystemTime};

/// `n` paths with identical activation samples (so every pair is a redundancy
/// candidate) and distinct, non-trivial weights.
fn identical_paths(n: usize) -> Vec<ExistingPathContribution> {
    let samples: Vec<HelpfulSample> = (0..50)
        .map(|i| HelpfulSample {
            activation: i as f32 / 50.0,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        })
        .collect();

    (0..n)
        .map(|k| ExistingPathContribution {
            source_uuid: format!("source-{k}"),
            existing_weight: 0.5 - 0.001 * k as f32,
            samples: samples.clone(),
        })
        .collect()
}

/// A deadline value is required by the signature, but every call these tests
/// drive is intercepted by the override queue — the real value is only a
/// fallback that must never be consulted while the queue still has entries.
fn unused_live_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

/// The pair-loop deadline check must stop a scan that started before the
/// deadline after exactly one pair has been evaluated.
#[test]
#[serial]
fn pair_loop_deadline_check_stops_a_scan_that_started_before_the_deadline() {
    let paths = identical_paths(4);

    // [false /*pair (0,1) proceeds*/, true /*pair (0,2) — deadline now reported passed*/]
    let _guard = DeadlineOverrideGuard::with_sequence(vec![false, true]);

    let mut pair_count = 0;
    let scan =
        detect_redundant_paths_observed("target-0", &paths, 1.0, &unused_live_deadline(), || {
            pair_count += 1;
        });

    assert_eq!(
        pair_count, 1,
        "the pair-loop deadline check must stop the scan after exactly one pair"
    );
    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "the pair-loop deadline check must report DeadlinePassed"
    );
    assert_eq!(
        scan.candidates.len(),
        1,
        "only pair (0,1), evaluated before the deadline, may contribute a candidate"
    );
}

/// With the override reporting "not passed" for every check, the full
/// 4-path scan (6 pairs) completes untouched.
#[test]
#[serial]
fn no_deadline_override_leaves_full_scan_untouched() {
    let paths = identical_paths(4);

    let _guard = DeadlineOverrideGuard::with_sequence(vec![false; 6]);

    let mut pair_count = 0;
    let scan =
        detect_redundant_paths_observed("target-0", &paths, 1.0, &unused_live_deadline(), || {
            pair_count += 1;
        });

    assert_eq!(pair_count, 6, "all 6 pairs of 4 paths must be evaluated");
    assert_eq!(
        scan.truncation, None,
        "a complete scan must not be truncated"
    );
    assert_eq!(
        scan.candidates.len(),
        6,
        "every identical-sample pair must yield a candidate"
    );
}
