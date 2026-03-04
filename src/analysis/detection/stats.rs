//! Shared statistical utility functions for detection modules.
//!
//! Consolidates Pearson correlation, mean, and variance computations
//! that were previously duplicated across multiple detection modules.

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
