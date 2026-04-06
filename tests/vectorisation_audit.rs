//! Integration tests for the vectorisation audit (Issue #1009).
//!
//! Verifies that the `SoA` (Structure-of-Arrays) reference implementations used in
//! the audit benchmarks produce results consistent with the production `AoS`
//! (Array-of-Structures) functions. This ensures the benchmark comparisons are
//! apples-to-apples and documents the expected equivalence.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Intentional numeric casts for GPU/neural network computation (Issue #873)

use neat_ai_discovery::analysis::samples::HelpfulSample;
use neat_ai_discovery::analysis::scoring::confidence::{
    compute_error_variance, compute_source_variance_confidence,
};
use neat_ai_discovery::analysis::scoring::error_distribution::ErrorDistribution;
use neat_ai_discovery::analysis::synapse::compute_synapse_improvement_and_count;

// =============================================================================
// Test Helpers
// =============================================================================

/// Generate test samples with known properties.
fn generate_test_samples(count: usize) -> Vec<HelpfulSample> {
    let mut samples = Vec::with_capacity(count);
    for i in 0..count {
        let phase = i as f32;
        let activation = if i % 50 == 0 {
            0.0
        } else {
            (phase * 0.1).sin() * 3.0 + (phase * 0.03).cos()
        };
        let avg_error = if i % 53 == 0 {
            f32::NAN
        } else if i % 71 == 0 {
            f32::INFINITY
        } else {
            (phase * 0.07).cos() * 0.5
        };
        let (target_value, target_activation) = if i % 5 == 0 {
            (None, None)
        } else {
            let tv = (phase * 0.05).sin() * 2.0;
            let ta = tv.tanh();
            (Some(tv), Some(ta))
        };
        samples.push(HelpfulSample {
            activation,
            avg_error,
            target_value,
            target_activation,
        });
    }
    samples
}

fn compute_baseline_error_sq(samples: &[HelpfulSample]) -> f32 {
    samples
        .iter()
        .map(|s| {
            if s.avg_error.is_finite() {
                s.avg_error * s.avg_error
            } else {
                0.0
            }
        })
        .sum()
}

// =============================================================================
// SoA Reference Implementations (duplicated from bench for test isolation)
// =============================================================================

const EPSILON: f32 = 1e-10;

fn soa_synapse_improvement_value_domain(
    activations: &[f32],
    errors: &[f32],
    weight: f32,
    total_baseline_error_sq: f32,
) -> (f32, u32, u32, u32) {
    if total_baseline_error_sq <= EPSILON || activations.is_empty() {
        return (0.0, 0, 0, activations.len() as u32);
    }

    let mut new_error_sq_sum = 0.0f32;
    let mut improved_count = 0u32;
    let mut worsened_count = 0u32;

    let len = activations.len().min(errors.len());
    for i in 0..len {
        let contribution = weight * activations[i];
        let baseline_error = errors[i];
        let new_error = baseline_error - contribution;

        if new_error.is_finite() {
            new_error_sq_sum += new_error * new_error;
        }

        if new_error.abs() + EPSILON < baseline_error.abs() {
            improved_count += 1;
        } else if new_error.abs() > baseline_error.abs() + EPSILON {
            worsened_count += 1;
        }
    }

    let improvement = if total_baseline_error_sq > EPSILON {
        (total_baseline_error_sq - new_error_sq_sum) / total_baseline_error_sq
    } else {
        0.0
    };
    let improvement = if improvement.is_finite() {
        improvement
    } else {
        0.0
    };

    (improvement, improved_count, worsened_count, len as u32)
}

fn soa_source_variance_confidence(activations: &[f32]) -> f32 {
    const MIN_CONFIDENT_STD_DEV: f32 = 0.05;

    if activations.len() < 2 {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0u32;

    for &a in activations {
        if a.is_finite() {
            let a64 = a as f64;
            sum += a64;
            sq_sum += a64 * a64;
            count += 1;
        }
    }

    if count < 2 {
        return 0.0;
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sq_sum / n) - (mean * mean);
    let std_dev = variance.max(0.0).sqrt() as f32;

    (std_dev / MIN_CONFIDENT_STD_DEV).clamp(0.0, 1.0)
}

fn soa_error_variance(errors: &[f32]) -> f32 {
    if errors.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut count = 0u32;

    for &e in errors {
        if e.is_finite() {
            let e64 = e as f64;
            sum += e64;
            sq_sum += e64 * e64;
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }

    let n = count as f64;
    let mean = sum / n;
    let variance = (sq_sum / n) - (mean * mean);

    variance.max(0.0) as f32
}

// =============================================================================
// Tests: SoA equivalence with AoS implementations
// =============================================================================

#[test]
fn test_soa_synapse_improvement_matches_aos_value_domain() {
    // The SoA implementation should produce the same results as AoS for the
    // value-domain (no target squash) path.
    for &size in &[10, 100, 1_000] {
        let samples = generate_test_samples(size);
        let baseline = compute_baseline_error_sq(&samples);
        let weight = 0.35;

        let (aos_imp, aos_improved, aos_worsened, aos_total) =
            compute_synapse_improvement_and_count(&samples, weight, baseline, None);

        let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();
        let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();
        let (soa_imp, soa_improved, soa_worsened, soa_total) =
            soa_synapse_improvement_value_domain(&activations, &errors, weight, baseline);

        assert_eq!(aos_total, soa_total, "Total count mismatch at size {size}");
        assert_eq!(
            aos_improved, soa_improved,
            "Improved count mismatch at size {size}: AoS={aos_improved}, SoA={soa_improved}"
        );
        assert_eq!(
            aos_worsened, soa_worsened,
            "Worsened count mismatch at size {size}: AoS={aos_worsened}, SoA={soa_worsened}"
        );
        assert!(
            (aos_imp - soa_imp).abs() < 1e-5,
            "Improvement mismatch at size {size}: AoS={aos_imp}, SoA={soa_imp}"
        );
    }
}

#[test]
fn test_soa_source_variance_matches_aos() {
    for &size in &[10, 100, 1_000] {
        let samples = generate_test_samples(size);

        let aos_result = compute_source_variance_confidence(&samples);

        let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();
        let soa_result = soa_source_variance_confidence(&activations);

        assert!(
            (aos_result - soa_result).abs() < 1e-6,
            "Variance confidence mismatch at size {size}: AoS={aos_result}, SoA={soa_result}"
        );
    }
}

#[test]
fn test_soa_error_variance_matches_aos() {
    for &size in &[10, 100, 1_000] {
        let samples = generate_test_samples(size);

        let aos_result = compute_error_variance(&samples);

        let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();
        let soa_result = soa_error_variance(&errors);

        assert!(
            (aos_result - soa_result).abs() < 1e-6,
            "Error variance mismatch at size {size}: AoS={aos_result}, SoA={soa_result}"
        );
    }
}

#[test]
fn test_error_distribution_from_samples_matches_from_errors() {
    // ErrorDistribution::from_samples extracts errors then calls from_errors.
    // Verify the extraction path doesn't alter results.
    let samples = generate_test_samples(500);

    let from_samples = ErrorDistribution::from_samples(&samples).unwrap();

    let errors: Vec<f32> = samples
        .iter()
        .map(|s| s.avg_error)
        .filter(|e| e.is_finite())
        .collect();
    let from_errors = ErrorDistribution::from_errors(&errors).unwrap();

    assert!(
        (from_samples.mean - from_errors.mean).abs() < 1e-6,
        "Mean mismatch: from_samples={}, from_errors={}",
        from_samples.mean,
        from_errors.mean
    );
    assert!(
        (from_samples.std_dev - from_errors.std_dev).abs() < 1e-6,
        "Std dev mismatch"
    );
    assert_eq!(from_samples.sample_count, from_errors.sample_count);
}

#[test]
fn test_field_extraction_preserves_data() {
    // Verify that extracting fields from HelpfulSample preserves values exactly.
    let samples = generate_test_samples(100);

    let activations: Vec<f32> = samples.iter().map(|s| s.activation).collect();
    let errors: Vec<f32> = samples.iter().map(|s| s.avg_error).collect();

    for (i, sample) in samples.iter().enumerate() {
        assert_eq!(
            sample.activation.to_bits(),
            activations[i].to_bits(),
            "Activation mismatch at index {i}"
        );
        assert_eq!(
            sample.avg_error.to_bits(),
            errors[i].to_bits(),
            "Error mismatch at index {i}"
        );
    }
}
