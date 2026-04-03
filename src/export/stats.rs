//! Statistics computation and numeric safety utilities for export.
//!
//! Provides aggregation functions (`compute_stats`) and JSON-safe float
//! conversion (`json_safe_f32`) used when building visualisation snapshots.

/// Apply a squash function by name.
///
/// # Performance (Issue #211)
/// Delegates to `crate::activations::apply_scalar_squash` which uses `normalise_squash_name`
/// with a fast path for already-uppercase strings (zero allocation in that case).
pub(crate) fn apply_squash(squash: &str, value: f32) -> f32 {
    // Use the centralised activation function implementation.
    // Returns None for aggregate squashes (MINIMUM/MAXIMUM/IF/etc.) which cannot be
    // represented as f(value). For those, default to identity (passthrough).
    crate::activations::apply_scalar_squash(squash, value).unwrap_or(value)
}

/// Compute stats from a slice of f32 values
pub(crate) fn compute_stats(values: &[f32]) -> (f32, f32, f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0);
    }

    // We intentionally ignore non-finite values (NaN/±Infinity) so they cannot
    // corrupt summary statistics or JSON serialisation.
    //
    // Important: Even with finite inputs, naive f32 accumulation can overflow
    // internally (eg variance for `[0.0, f32::MAX]`). We compute in f64 and then
    // clamp to JSON-safe f32 outputs.
    //
    // When there are no finite values, return zeros rather than ±Infinity.
    // This matches the empty-slice behaviour and keeps the output JSON valid.
    let mut count: u64 = 0;
    let mut mean: f64 = 0.0;
    let mut m2: f64 = 0.0;
    let mut min: f64 = 0.0;
    let mut max: f64 = 0.0;

    for &x in values {
        if !x.is_finite() {
            continue;
        }

        let xf = x as f64;

        if count == 0 {
            count = 1;
            mean = xf;
            m2 = 0.0;
            min = xf;
            max = xf;
            continue;
        }

        if xf < min {
            min = xf;
        }
        if xf > max {
            max = xf;
        }

        count += 1;
        let delta = xf - mean;
        mean += delta / count as f64;
        let delta2 = xf - mean;
        m2 += delta * delta2;
    }

    if count == 0 {
        return (0.0, 0.0, 0.0, 0.0);
    }

    let variance = m2 / count as f64;

    (
        json_safe_f32(mean as f32),
        json_safe_f32(variance as f32),
        json_safe_f32(min as f32),
        json_safe_f32(max as f32),
    )
}

/// Convert a float to a JSON-safe finite value.
///
/// `serde_json` will serialise non-finite floats (NaN/±Infinity) as `null`, which
/// breaks consumers that expect numeric arrays (and breaks round-tripping into
/// `Vec<f32>`).
///
/// We clamp ±Infinity to ±`f32::MAX` and map NaN to 0.0. This keeps exported
/// snapshots valid JSON while preserving sign/magnitude semantics as much as
/// possible for debugging.
pub(crate) fn json_safe_f32(value: f32) -> f32 {
    if value.is_finite() {
        value
    } else if value.is_nan() {
        0.0
    } else if value.is_sign_positive() {
        f32::MAX
    } else {
        -f32::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_squash_identity() {
        assert!((apply_squash("IDENTITY", 0.5) - 0.5).abs() < 1e-6);
        assert!((apply_squash("IDENTITY", -1.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_apply_squash_tanh() {
        assert!((apply_squash("TANH", 0.0) - 0.0).abs() < 1e-6);
        assert!((apply_squash("TANH", 1.0) - 1.0_f32.tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_apply_squash_relu() {
        assert!((apply_squash("RELU", 1.0) - 1.0).abs() < 1e-6);
        assert!((apply_squash("RELU", -1.0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_squash_hard_tanh() {
        assert!((apply_squash("HARD_TANH", 0.5) - 0.5).abs() < 1e-6);
        assert!((apply_squash("HARD_TANH", 2.0) - 1.0).abs() < 1e-6);
        assert!((apply_squash("HARD_TANH", -2.0) - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_empty() {
        let (mean, var, min, max) = compute_stats(&[]);
        assert_eq!(mean, 0.0);
        assert_eq!(var, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn test_compute_stats_single() {
        let (mean, var, min, max) = compute_stats(&[5.0]);
        assert!((mean - 5.0).abs() < 1e-6);
        assert!((var - 0.0).abs() < 1e-6);
        assert!((min - 5.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_multiple() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, var, min, max) = compute_stats(&values);
        assert!((mean - 3.0).abs() < 1e-6);
        assert!((var - 2.0).abs() < 1e-6); // variance of [1,2,3,4,5] = 2
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_ignores_non_finite_values() {
        // Mean/variance should be computed over finite values only.
        // This avoids skewing the result when recordings contain NaN/±Infinity.
        let values = [1.0_f32, 2.0, f32::NAN, 3.0];
        let (mean, var, min, max) = compute_stats(&values);

        assert!((mean - 2.0).abs() < 1e-6);
        assert!((var - (2.0 / 3.0)).abs() < 1e-6);
        assert!((min - 1.0).abs() < 1e-6);
        assert!((max - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_all_non_finite_returns_zeros() {
        // Returning ±Infinity here is semantically wrong and can break JSON output.
        let values = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let (mean, var, min, max) = compute_stats(&values);

        assert_eq!(mean, 0.0);
        assert_eq!(var, 0.0);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
    }
}
