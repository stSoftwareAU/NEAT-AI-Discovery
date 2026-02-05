//! Tests for GPU batch evaluation functionality.
//!
//! Tests cover:
//! - GPU batch evaluation edge cases
//! - Batch result merging

use super::common::*;

/// TDD Test: evaluate_harmful_batch should handle empty batch gracefully.
#[test]
fn evaluate_harmful_batch_handles_empty_batch() {
    skip_if_no_gpu!();
    let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

    let empty_batch: Vec<(&[HelpfulSample], f32)> = vec![];
    let result = analyzer
        .evaluate_harmful_batch(&empty_batch)
        .expect("Empty batch should succeed");

    assert!(result.is_empty(), "Empty batch should return empty results");
}

/// TDD Test: evaluate_harmful_batch should handle batch with empty sample sets.
#[test]
fn evaluate_harmful_batch_handles_empty_sample_sets() {
    skip_if_no_gpu!();
    let analyzer = GpuAnalyzer::new().expect("GPU analyser creation should succeed in tests");

    let samples: Vec<HelpfulSample> = (0..20)
        .map(|i| HelpfulSample {
            activation: (i as f32) / 20.0,
            avg_error: 0.1,
            target_value: None,
            target_activation: None,
        })
        .collect();
    let empty_samples: Vec<HelpfulSample> = vec![];

    let batch_input = vec![
        (&samples[..], 0.5),
        (&empty_samples[..], 0.3), // Empty set in the middle
        (&samples[..], -0.2),
    ];

    let batched = analyzer
        .evaluate_harmful_batch(&batch_input)
        .expect("Batch with empty set should succeed");

    assert_eq!(batched.len(), 3, "Should return 3 results");

    // Middle result should be default (all zeros)
    assert_eq!(
        batched[1].harmful_count, 0,
        "Empty sample set should have 0 harmful_count"
    );
    assert_eq!(
        batched[1].helpful_count, 0,
        "Empty sample set should have 0 helpful_count"
    );
}

#[test]
fn merge_batch_results_preserves_order_with_empty_samples() {
    let flags = vec![false, true, false, true];
    let merged = GpuAnalyzer::merge_batch_results(
        &flags,
        vec![
            HelpfulStats {
                positive_count: 1,
                ..HelpfulStats::default()
            },
            HelpfulStats {
                positive_count: 2,
                ..HelpfulStats::default()
            },
        ],
    );

    assert_eq!(
        merged.len(),
        flags.len(),
        "Merged results should match input batch length"
    );
    assert_eq!(
        merged[0].positive_count, 1,
        "First non-empty sample should remain first"
    );
    assert_eq!(
        merged[1].positive_count, 0,
        "Empty samples should produce default stats"
    );
    assert_eq!(
        merged[2].positive_count, 2,
        "Second non-empty sample should remain in original position"
    );
    assert_eq!(
        merged[3].positive_count, 0,
        "Trailing empty samples should also produce defaults"
    );
}
