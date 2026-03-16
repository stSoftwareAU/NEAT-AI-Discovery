//! Integration tests for consolidated Pearson correlation variants (Issue #767).

use std::collections::HashMap;

use neat_ai_discovery::analysis::detection::stats::{
    pearson_correlation, pearson_correlation_hashmaps, pearson_correlation_samples,
};
use neat_ai_discovery::analysis::samples::HelpfulSample;

/// Helper: build a `HelpfulSample` with just an activation value.
fn sample(activation: f32) -> HelpfulSample {
    HelpfulSample {
        activation,
        ..Default::default()
    }
}

// ── pearson_correlation (existing, f32 slices) ──────────────────────────

#[test]
fn pearson_correlation_perfect_positive() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
    let r = pearson_correlation(&x, &y);
    assert!((r - 1.0).abs() < 1e-5, "expected ~1.0, got {r}");
}

#[test]
fn pearson_correlation_perfect_negative() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
    let r = pearson_correlation(&x, &y);
    assert!((r + 1.0).abs() < 1e-5, "expected ~-1.0, got {r}");
}

#[test]
fn pearson_correlation_zero_variance_returns_zero() {
    let x = vec![3.0, 3.0, 3.0, 3.0];
    let y = vec![1.0, 2.0, 3.0, 4.0];
    assert_eq!(pearson_correlation(&x, &y), 0.0);
}

#[test]
fn pearson_correlation_too_few_returns_zero() {
    assert_eq!(pearson_correlation(&[1.0], &[2.0]), 0.0);
    assert_eq!(pearson_correlation(&[], &[]), 0.0);
}

// ── pearson_correlation_samples (HelpfulSample, f64) ─────────────────────

#[test]
fn samples_perfect_positive() {
    let a: Vec<HelpfulSample> = (1..=5).map(|i| sample(i as f32)).collect();
    let b: Vec<HelpfulSample> = (1..=5).map(|i| sample(i as f32 * 2.0)).collect();
    let r = pearson_correlation_samples(&a, &b, 5);
    assert!((r - 1.0).abs() < 1e-10, "expected ~1.0, got {r}");
}

#[test]
fn samples_perfect_negative() {
    let a: Vec<HelpfulSample> = (1..=5).map(|i| sample(i as f32)).collect();
    let b: Vec<HelpfulSample> = (1..=5).map(|i| sample(6.0 - i as f32)).collect();
    let r = pearson_correlation_samples(&a, &b, 5);
    assert!((r + 1.0).abs() < 1e-10, "expected ~-1.0, got {r}");
}

#[test]
fn samples_too_few_returns_zero() {
    let a = vec![sample(1.0), sample(2.0)];
    let b = vec![sample(3.0), sample(4.0)];
    assert_eq!(pearson_correlation_samples(&a, &b, 2), 0.0);
}

#[test]
fn samples_respects_n_samples_limit() {
    // First 3 samples are perfectly correlated; 4th would break it
    let a = vec![sample(1.0), sample(2.0), sample(3.0), sample(100.0)];
    let b = vec![sample(2.0), sample(4.0), sample(6.0), sample(-50.0)];
    let r = pearson_correlation_samples(&a, &b, 3);
    assert!((r - 1.0).abs() < 1e-10, "expected ~1.0, got {r}");
}

// ── pearson_correlation_hashmaps (HashMap<u32, f32>) ─────────────────────

#[test]
fn hashmaps_perfect_positive() {
    let a: HashMap<u32, f32> = (0..5).map(|i| (i, i as f32 + 1.0)).collect();
    let b: HashMap<u32, f32> = (0..5).map(|i| (i, (i as f32 + 1.0) * 2.0)).collect();
    let r = pearson_correlation_hashmaps(&a, &b, 2);
    assert!((r - 1.0).abs() < 1e-5, "expected ~1.0, got {r}");
}

#[test]
fn hashmaps_only_uses_shared_keys() {
    let a: HashMap<u32, f32> = [(0, 1.0), (1, 2.0), (2, 3.0), (99, 999.0)]
        .into_iter()
        .collect();
    let b: HashMap<u32, f32> = [(0, 2.0), (1, 4.0), (2, 6.0), (50, -100.0)]
        .into_iter()
        .collect();
    let r = pearson_correlation_hashmaps(&a, &b, 2);
    assert!((r - 1.0).abs() < 1e-5, "expected ~1.0, got {r}");
}

#[test]
fn hashmaps_below_min_samples_returns_zero() {
    let a: HashMap<u32, f32> = [(0, 1.0)].into_iter().collect();
    let b: HashMap<u32, f32> = [(0, 2.0)].into_iter().collect();
    assert_eq!(pearson_correlation_hashmaps(&a, &b, 5), 0.0);
}

#[test]
fn hashmaps_no_shared_keys_returns_zero() {
    let a: HashMap<u32, f32> = [(0, 1.0), (1, 2.0)].into_iter().collect();
    let b: HashMap<u32, f32> = [(10, 3.0), (11, 4.0)].into_iter().collect();
    assert_eq!(pearson_correlation_hashmaps(&a, &b, 2), 0.0);
}
