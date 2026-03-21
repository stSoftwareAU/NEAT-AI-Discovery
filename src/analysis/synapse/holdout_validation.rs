//! Hold-out validation for multi-weight search (Issue #893).
//!
//! The 9-variant multi-weight search picks the best-looking weight from 9
//! options, creating selection bias that causes overfitting. This module
//! provides deterministic train/validate splitting so that weights are
//! selected on training data and improvement is reported on held-out
//! validation data.

use crate::analysis::constants::{HOLDOUT_MIN_SAMPLE_COUNT, HOLDOUT_VALIDATION_FRACTION};
use crate::analysis::samples::HelpfulSample;

/// Result of a hold-out split.
pub(crate) struct HoldoutSplit<'a> {
    /// Training samples used for weight selection.
    pub train: Vec<&'a HelpfulSample>,
    /// Validation samples used for improvement reporting.
    pub validate: Vec<&'a HelpfulSample>,
}

/// Deterministically split samples into train and validate sets.
///
/// Uses a hash of (`source_uuid`, `target_uuid`) as the split seed to ensure
/// reproducibility across evaluations of the same synapse candidate.
///
/// Returns `None` if the sample count is below `HOLDOUT_MIN_SAMPLE_COUNT`,
/// signalling that the caller should fall back to full-sample evaluation.
pub(crate) fn split_samples_holdout<'a>(
    samples: &'a [HelpfulSample],
    source_uuid: &str,
    target_uuid: &str,
) -> Option<HoldoutSplit<'a>> {
    if samples.len() < HOLDOUT_MIN_SAMPLE_COUNT {
        return None;
    }

    let seed = deterministic_seed(source_uuid, target_uuid);
    // Compute validation count: fraction of total, at least 1.
    // Sample counts are bounded by practical sizes (well under 2^52).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    let validate_count = {
        let frac = (samples.len() as f64 * f64::from(HOLDOUT_VALIDATION_FRACTION)).round();
        frac.max(1.0) as usize
    };

    // Assign each sample to train or validate based on a deterministic hash
    // of its index combined with the seed.
    let mut train = Vec::with_capacity(samples.len() - validate_count);
    let mut validate = Vec::with_capacity(validate_count);

    for (i, sample) in samples.iter().enumerate() {
        if is_validation_sample(seed, i, samples.len(), validate_count) {
            validate.push(sample);
        } else {
            train.push(sample);
        }
    }

    // Safety: ensure both partitions are non-empty
    if train.is_empty() || validate.is_empty() {
        return None;
    }

    Some(HoldoutSplit { train, validate })
}

/// Compute a deterministic seed from source and target UUIDs using FNV-1a.
fn deterministic_seed(source_uuid: &str, target_uuid: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV-1a offset basis
    for b in source_uuid.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= 0xFF;
    hash = hash.wrapping_mul(0x100000001b3);
    for b in target_uuid.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Determine whether a sample at the given index belongs to the validation set.
///
/// Uses a stride-based selection seeded by the deterministic hash to ensure
/// even distribution across the sample range.
fn is_validation_sample(seed: u64, index: usize, total: usize, validate_count: usize) -> bool {
    // Use the seed to pick a deterministic offset, then select evenly-spaced
    // indices for the validation set.
    // total is from samples.len() which fits in u64; modulo guarantees result < total,
    // so the truncation to usize is safe.
    let offset = usize::try_from(seed % (total as u64)).unwrap_or(0);
    let shifted = (index + offset) % total;
    // Select the first `validate_count` positions in the shifted order
    shifted < validate_count
}

/// Collect owned samples from references.
pub(crate) fn collect_samples(refs: &[&HelpfulSample]) -> Vec<HelpfulSample> {
    refs.iter().copied().copied().collect()
}

/// Compute baseline error squared for a set of sample references.
pub(crate) fn baseline_error_sq(samples: &[&HelpfulSample]) -> f32 {
    let mut total = 0.0f32;
    for sample in samples {
        if sample.avg_error.is_finite() {
            total += sample.avg_error * sample.avg_error;
        }
    }
    total
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // Intentional for test data generation
mod tests {
    use super::*;

    fn make_samples(count: usize) -> Vec<HelpfulSample> {
        (0..count)
            .map(|i| HelpfulSample {
                activation: i as f32 * 0.1,
                avg_error: (i as f32 - count as f32 / 2.0) * 0.01,
                target_value: None,
                target_activation: None,
            })
            .collect()
    }

    #[test]
    fn test_split_returns_none_below_threshold() {
        let samples = make_samples(10);
        let result = split_samples_holdout(&samples, "source-1", "target-1");
        assert!(
            result.is_none(),
            "Should return None below HOLDOUT_MIN_SAMPLE_COUNT"
        );
    }

    #[test]
    fn test_split_produces_correct_partition_sizes() {
        let samples = make_samples(100);
        let split = split_samples_holdout(&samples, "source-1", "target-1").unwrap();
        // 30% of 100 = 30 validation samples
        assert_eq!(split.validate.len(), 30);
        assert_eq!(split.train.len(), 70);
        // All samples accounted for
        assert_eq!(split.train.len() + split.validate.len(), 100);
    }

    #[test]
    fn test_split_is_deterministic() {
        let samples = make_samples(50);
        let split1 = split_samples_holdout(&samples, "source-a", "target-b").unwrap();
        let split2 = split_samples_holdout(&samples, "source-a", "target-b").unwrap();

        // Same UUIDs must produce the same split
        assert_eq!(split1.train.len(), split2.train.len());
        assert_eq!(split1.validate.len(), split2.validate.len());

        for (a, b) in split1.train.iter().zip(split2.train.iter()) {
            assert_eq!(a.activation, b.activation);
            assert_eq!(a.avg_error, b.avg_error);
        }
    }

    #[test]
    fn test_different_uuids_produce_different_splits() {
        let samples = make_samples(50);
        let split1 = split_samples_holdout(&samples, "source-a", "target-b").unwrap();
        let split2 = split_samples_holdout(&samples, "source-c", "target-d").unwrap();

        // Different UUIDs should (almost certainly) produce different validation sets.
        // Compare the activations of the first validation sample.
        let v1_activations: Vec<f32> = split1.validate.iter().map(|s| s.activation).collect();
        let v2_activations: Vec<f32> = split2.validate.iter().map(|s| s.activation).collect();
        assert_ne!(
            v1_activations, v2_activations,
            "Different UUID pairs should produce different splits"
        );
    }

    #[test]
    fn test_no_sample_in_both_partitions() {
        let samples = make_samples(50);
        let split = split_samples_holdout(&samples, "source-1", "target-1").unwrap();

        // Collect all activation values (unique in our test data)
        let train_acts: std::collections::HashSet<u32> =
            split.train.iter().map(|s| s.activation.to_bits()).collect();
        let validate_acts: std::collections::HashSet<u32> = split
            .validate
            .iter()
            .map(|s| s.activation.to_bits())
            .collect();

        let overlap: Vec<_> = train_acts.intersection(&validate_acts).collect();
        assert!(
            overlap.is_empty(),
            "No sample should appear in both train and validate"
        );
    }

    #[test]
    fn test_baseline_error_sq_computation() {
        let samples = [
            HelpfulSample {
                activation: 1.0,
                avg_error: 3.0,
                target_value: None,
                target_activation: None,
            },
            HelpfulSample {
                activation: 2.0,
                avg_error: 4.0,
                target_value: None,
                target_activation: None,
            },
        ];
        let refs: Vec<&HelpfulSample> = samples.iter().collect();
        let result = baseline_error_sq(&refs);
        // 3^2 + 4^2 = 9 + 16 = 25
        assert!((result - 25.0).abs() < 1e-6);
    }

    #[test]
    fn test_collect_samples_roundtrip() {
        let samples = make_samples(10);
        let refs: Vec<&HelpfulSample> = samples.iter().collect();
        let collected = collect_samples(&refs);
        assert_eq!(collected.len(), 10);
        for (orig, coll) in samples.iter().zip(collected.iter()) {
            assert_eq!(orig.activation, coll.activation);
            assert_eq!(orig.avg_error, coll.avg_error);
        }
    }
}
