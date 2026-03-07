//! Shared statistical utility functions for detection and recommendation modules.
//!
//! Consolidates Pearson correlation, mean, and variance computations
//! that were previously duplicated across multiple modules (Issue #767).

use std::collections::HashMap;

use crate::analysis::samples::HelpfulSample;

/// Compute the arithmetic mean of a slice of values.
///
/// Returns `0.0` for empty input.
pub fn compute_mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

/// Compute the population variance of a slice of values.
///
/// Returns `0.0` for slices with fewer than 2 elements.
pub fn compute_variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n
}

/// Compute the Pearson correlation coefficient between two slices.
///
/// Uses the minimum of the two slice lengths when they differ.
/// Returns `0.0` for inputs with fewer than 2 elements or zero variance.
/// Result is always clamped to `[-1.0, 1.0]` to guard against
/// floating-point overshoot.
pub fn pearson_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let n_f = n as f32;
    let mean_x: f32 = x[..n].iter().sum::<f32>() / n_f;
    let mean_y: f32 = y[..n].iter().sum::<f32>() / n_f;

    let mut cov = 0.0_f32;
    let mut var_x = 0.0_f32;
    let mut var_y = 0.0_f32;

    for i in 0..n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < f32::EPSILON {
        return 0.0;
    }

    (cov / denom).clamp(-1.0, 1.0)
}

/// Compute Pearson correlation between two `HelpfulSample` slices using f64 precision.
///
/// Correlates the `activation` field of each sample. Uses the first `n_samples`
/// entries from each slice. Returns `0.0` when `n_samples < 3` or variance is
/// near zero.
pub fn pearson_correlation_samples(
    samples_a: &[HelpfulSample],
    samples_b: &[HelpfulSample],
    n_samples: usize,
) -> f64 {
    if n_samples < 3 {
        return 0.0;
    }

    let mut sum_a = 0.0_f64;
    let mut sum_b = 0.0_f64;
    for i in 0..n_samples {
        sum_a += samples_a[i].activation as f64;
        sum_b += samples_b[i].activation as f64;
    }
    let mean_a = sum_a / n_samples as f64;
    let mean_b = sum_b / n_samples as f64;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for i in 0..n_samples {
        let da = samples_a[i].activation as f64 - mean_a;
        let db = samples_b[i].activation as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denominator = (var_a * var_b).sqrt();
    if denominator < 1e-10 {
        return 0.0;
    }

    cov / denominator
}

/// Compute Pearson correlation between two `HashMap<u32, f32>` maps.
///
/// Pairs values by shared keys (typically `obs_index`). Returns `0.0` when
/// the number of shared entries is below `min_samples` or variance is near zero.
pub fn pearson_correlation_hashmaps(
    map_a: &HashMap<u32, f32>,
    map_b: &HashMap<u32, f32>,
    min_samples: usize,
) -> f32 {
    let vals: Vec<(f32, f32)> = map_a
        .iter()
        .filter_map(|(k, &va)| map_b.get(k).map(|&vb| (va, vb)))
        .collect();

    let n = vals.len();
    if n < min_samples {
        return 0.0;
    }

    let n_f = n as f32;
    let mean_a: f32 = vals.iter().map(|(a, _)| a).sum::<f32>() / n_f;
    let mean_b: f32 = vals.iter().map(|(_, b)| b).sum::<f32>() / n_f;

    let mut cov = 0.0_f32;
    let mut var_a = 0.0_f32;
    let mut var_b = 0.0_f32;
    for &(a, b) in &vals {
        let da = a - mean_a;
        let db = b - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    let denom = (var_a * var_b).sqrt();
    if denom < 1e-10 {
        return 0.0;
    }

    cov / denom
}

/// Compute fractional ranks for a slice of f32 values.
///
/// Ties receive the average of their ranks (1-based).
/// Returns an empty vector for empty input.
pub fn compute_ranks(values: &[f32]) -> Vec<f32> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut ranks = vec![0.0f32; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        // Find all ties
        while j < n && indexed[j].1.total_cmp(&indexed[i].1) == std::cmp::Ordering::Equal {
            j += 1;
        }
        // Average rank for tied values (1-based)
        let avg_rank = (i + j) as f32 / 2.0 + 0.5;
        for item in indexed.iter().take(j).skip(i) {
            ranks[item.0] = avg_rank;
        }
        i = j;
    }

    ranks
}

/// Compute Spearman's rank correlation coefficient between two slices.
///
/// This is equivalent to computing the Pearson correlation on the ranks of
/// the input values.
///
/// Returns a value in `[-1.0, 1.0]` where:
/// - `+1.0` = perfectly monotonically increasing
/// - `-1.0` = perfectly monotonically decreasing
/// - `0.0` = no monotonic relationship
///
/// Returns `0.0` for inputs with fewer than 2 elements or zero variance.
pub fn spearman_rank_correlation(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let rank_x = compute_ranks(x);
    let rank_y = compute_ranks(y);

    // Pearson correlation on ranks
    let n_f = n as f32;
    let mean_rx: f32 = rank_x.iter().sum::<f32>() / n_f;
    let mean_ry: f32 = rank_y.iter().sum::<f32>() / n_f;

    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;

    for i in 0..n {
        let dx = rank_x[i] - mean_rx;
        let dy = rank_y[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 {
        return 0.0;
    }

    cov / denom
}
