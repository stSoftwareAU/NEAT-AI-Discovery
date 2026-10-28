//! Issue #2374 (CWE-834): the redundant-path pair scan must not grow
//! quadratically with fan-in, and must not run an uncancellable O(P²·S) loop.
//!
//! This test deliberately uses only the pre-existing public API
//! (`ExistingPathContribution` and `detect_redundant_paths`). It therefore
//! compiles against the unfixed base and must fail at runtime there: 16 paths
//! yield 120 candidates, while 64 paths yield 2,016 (P(P-1)/2) with no ceiling.
//! With the fan-in cap and candidate ceiling in place, 64 paths yield at most
//! 256 candidates, so the growth ratio stays within the 4x bound.

#![allow(clippy::cast_precision_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use neat_ai_discovery::analysis::detection::redundant_path::{
    ExistingPathContribution, detect_redundant_paths,
};
use neat_ai_discovery::analysis::samples::HelpfulSample;

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

#[test]
fn pair_scan_grows_linearly_with_fan_in() {
    let small = detect_redundant_paths("target-0", &identical_paths(16), 1.0);
    let big = detect_redundant_paths("target-0", &identical_paths(64), 1.0);

    // Positive control: 16 identical paths must yield candidates (120 pairs).
    assert!(
        !small.is_empty(),
        "positive control: 16 identical-sample paths must yield redundant candidates"
    );

    // Quadrupling fan-in must not multiply the candidate count by more than 4.
    assert!(
        big.len() <= 4 * small.len(),
        "redundant-path candidates grew super-linearly with fan-in (Issue #2374): \
         16 paths gave {} candidates, 64 paths gave {} (limit {})",
        small.len(),
        big.len(),
        4 * small.len()
    );
}
