//! Issue #2374 (CWE-834): integration tests for the redundant-path scan bounds.
//!
//! Covers the candidate ceiling, the fan-in cap (pair evaluations, skipped
//! count, and retention of the strongest weights), the no-cap path, and the
//! deadline behaviour of `detect_redundant_paths_with_deadline`.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use neat_ai_discovery::analysis::detection::redundant_path::{
    ExistingPathContribution, MAX_REDUNDANT_PATH_CANDIDATES, MAX_REDUNDANT_PATH_FAN_IN,
    detect_redundant_paths, detect_redundant_paths_observed, detect_redundant_paths_with_deadline,
};
use neat_ai_discovery::analysis::recommendation::epistatic::ScanTruncation;
use neat_ai_discovery::analysis::samples::HelpfulSample;

fn sample(activation: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        avg_error: 0.1,
        target_value: None,
        target_activation: None,
    }
}

/// Ramp samples shared by every "identical" path, so any two of them are
/// perfectly correlated.
fn ramp_samples() -> Vec<HelpfulSample> {
    (0..64).map(|i| sample(i as f32 / 64.0)).collect()
}

/// Deterministic pseudo-random samples keyed by `seed` (64-bit LCG), so paths
/// with distinct seeds are uncorrelated (|r| far below 0.85).
fn noise_samples(seed: u64) -> Vec<HelpfulSample> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
    (0..64)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let activation = (state >> 40) as f32 / (1u64 << 24) as f32;
            sample(activation)
        })
        .collect()
}

/// `n` paths with identical ramp samples and distinct, non-trivial weights.
fn identical_paths(n: usize) -> Vec<ExistingPathContribution> {
    (0..n)
        .map(|k| ExistingPathContribution {
            source_uuid: format!("source-{k}"),
            existing_weight: 0.5 - 0.001 * k as f32,
            samples: ramp_samples(),
        })
        .collect()
}

/// `n` mutually uncorrelated paths with the given weight.
fn uncorrelated_paths(n: usize, weight: f32) -> Vec<ExistingPathContribution> {
    (0..n)
        .map(|k| ExistingPathContribution {
            source_uuid: format!("noise-{k}"),
            existing_weight: weight,
            samples: noise_samples(k as u64),
        })
        .collect()
}

fn far_future_deadline() -> Option<SystemTime> {
    Some(SystemTime::now() + Duration::from_secs(86_400))
}

#[test]
fn candidate_ceiling_stops_the_scan() {
    // 64 identical paths: all 2,016 pairs are redundant, so the ceiling binds.
    let paths = identical_paths(64);

    let mut pairs_observed = 0usize;
    let scan = detect_redundant_paths_observed("target-0", &paths, 1.0, &None, || {
        pairs_observed += 1;
    });

    assert_eq!(
        scan.candidates.len(),
        MAX_REDUNDANT_PATH_CANDIDATES,
        "candidates must stop at the ceiling"
    );
    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::CandidateCeiling),
        "hitting the candidate ceiling must be reported"
    );
    assert_eq!(
        scan.paths_skipped, 0,
        "64 paths is within the fan-in cap, so nothing is skipped"
    );
    assert_eq!(
        pairs_observed, MAX_REDUNDANT_PATH_CANDIDATES,
        "the scan must stop evaluating pairs once the ceiling is reached"
    );
}

#[test]
fn fan_in_cap_bounds_pair_evaluations() {
    // 200 mutually uncorrelated paths: no pair is a candidate, so the only
    // thing bounding the work is the fan-in cap.
    let paths = uncorrelated_paths(200, 0.5);

    let mut pairs_observed = 0usize;
    let scan = detect_redundant_paths_observed("target-0", &paths, 1.0, &None, || {
        pairs_observed += 1;
    });

    assert_eq!(
        pairs_observed,
        MAX_REDUNDANT_PATH_FAN_IN * (MAX_REDUNDANT_PATH_FAN_IN - 1) / 2,
        "pair evaluations must be bounded by the fan-in cap of 64 (2,016 pairs)"
    );
    assert_eq!(
        scan.paths_skipped, 136,
        "200 valid paths minus the 64 kept must be reported as skipped"
    );
    assert_eq!(scan.truncation, None, "no deadline or ceiling was hit");
    assert!(
        scan.candidates.is_empty(),
        "uncorrelated paths must not produce candidates"
    );
}

#[test]
fn fan_in_cap_keeps_the_strongest_weights() {
    // 200 uncorrelated low-weight paths, then two identical-sample paths with
    // the strongest weights placed at the END. A cap that simply took the first
    // 64 would drop them and find no candidate.
    let mut paths = uncorrelated_paths(200, 0.02);
    paths.push(ExistingPathContribution {
        source_uuid: "strong-keep".to_string(),
        existing_weight: 0.9,
        samples: ramp_samples(),
    });
    paths.push(ExistingPathContribution {
        source_uuid: "strong-prune".to_string(),
        existing_weight: 0.8,
        samples: ramp_samples(),
    });

    let scan = detect_redundant_paths_with_deadline("target-0", &paths, 1.0, &None);

    assert_eq!(
        scan.paths_skipped, 138,
        "202 valid paths minus the 64 kept must be reported as skipped"
    );
    assert!(
        scan.candidates
            .iter()
            .any(|c| c.keep_source_uuid == "strong-keep" && c.prune_source_uuid == "strong-prune"),
        "the strongest-weight pair must survive the fan-in cap (Issue #2374); got {:?}",
        scan.candidates
            .iter()
            .map(|c| (&c.keep_source_uuid, &c.prune_source_uuid))
            .collect::<Vec<_>>()
    );
}

#[test]
fn fan_in_cap_keeps_the_strongest_weights_positive_control() {
    // Same two strong paths with only 10 uncorrelated paths (no cap): the
    // candidate must be found, proving the cap is what hides it above.
    let mut paths = uncorrelated_paths(10, 0.02);
    paths.push(ExistingPathContribution {
        source_uuid: "strong-keep".to_string(),
        existing_weight: 0.9,
        samples: ramp_samples(),
    });
    paths.push(ExistingPathContribution {
        source_uuid: "strong-prune".to_string(),
        existing_weight: 0.8,
        samples: ramp_samples(),
    });

    let scan = detect_redundant_paths_with_deadline("target-0", &paths, 1.0, &None);

    assert_eq!(scan.paths_skipped, 0, "12 valid paths is under the cap");
    assert!(
        scan.candidates
            .iter()
            .any(|c| c.keep_source_uuid == "strong-keep" && c.prune_source_uuid == "strong-prune"),
        "positive control: the strong pair must be found when nothing is capped"
    );
}

#[test]
fn under_the_cap_nothing_is_skipped() {
    let paths = uncorrelated_paths(10, 0.5);

    let scan = detect_redundant_paths_with_deadline("target-0", &paths, 1.0, &None);

    assert_eq!(scan.paths_skipped, 0);
    assert_eq!(scan.truncation, None);
}

#[test]
fn far_future_deadline_matches_no_deadline() {
    let paths = identical_paths(12);

    let baseline = detect_redundant_paths("target-0", &paths, 1.0);
    let scan =
        detect_redundant_paths_with_deadline("target-0", &paths, 1.0, &far_future_deadline());

    assert_eq!(
        scan.candidates.len(),
        baseline.len(),
        "a deadline that has not elapsed must not change the candidates"
    );
    assert_eq!(scan.truncation, None);
}

#[test]
fn elapsed_deadline_returns_deadline_passed() {
    let paths = identical_paths(12);

    let scan = detect_redundant_paths_with_deadline("target-0", &paths, 1.0, &Some(UNIX_EPOCH));

    assert_eq!(
        scan.truncation,
        Some(ScanTruncation::DeadlinePassed),
        "an elapsed deadline must be reported as DeadlinePassed"
    );
    assert!(
        scan.candidates.is_empty(),
        "an elapsed deadline stops the scan before the first pair"
    );
}
