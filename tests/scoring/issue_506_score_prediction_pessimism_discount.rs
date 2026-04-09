//! Issue #506: Score prediction over-estimation causes 95% failure rate
//!
//! Verifies that the pessimism discount is applied to candidate score predictions
//! to reduce the over-estimation observed in production (creature b2ff6e45).
//!
//! Evidence: expected gains were 18,500× too high for the sole successful candidate,
//! and wrong-sign for 19 of 20 candidates. The fix applies a pessimism discount
//! based on the `improved_count / total_count` ratio to produce more realistic
//! predictions.

#![allow(clippy::cast_precision_loss, clippy::cast_sign_loss)] // Intentional numeric casts for GPU/neural network computation (Issue #873)
use crate::skip_without_gpu;
use neat_ai_discovery::analysis::synapse::apply_pessimism_discount;

/// Test that pessimism discount reduces predictions when not all samples improve.
#[test]
fn test_pessimism_discount_partial_improvement() {
    // Only 60% of samples improved
    let discounted = apply_pessimism_discount(0.05, 60, 100);

    // Issue #733: Changed from linear to concave curve.
    // discount = 0.15 + 0.85 × 0.6^0.6 ≈ 0.15 + 0.85 × 0.736 ≈ 0.776
    // discounted = 0.05 × 0.776 ≈ 0.0388
    assert!(
        discounted < 0.05,
        "Pessimism discount should reduce gain when only 60% of samples improved, got {discounted}"
    );
    assert!(
        discounted > 0.0,
        "Discounted gain should remain positive, got {discounted}"
    );
    // Concave curve gives higher result than old linear (0.033)
    assert!(
        discounted > 0.033,
        "Issue #733: Concave curve at 60% should give higher result than old linear (0.033), got {discounted}"
    );
}

/// Test that pessimism discount is neutral when all samples improve.
///
/// When every sample improves (ratio = 1.0), the discount factor is 1.0 — no
/// penalty is applied. The pessimism floor only kicks in when fewer samples
/// improve, so a perfect improvement ratio should preserve the full prediction.
#[test]
fn test_pessimism_discount_all_improved() {
    let discounted = apply_pessimism_discount(0.05, 100, 100);

    // discount = 0.15 + 0.85 × 1.0 = 1.0 → no reduction
    assert!(
        (discounted - 0.05).abs() < 0.001,
        "All samples improving should yield no discount, expected 0.05, got {discounted}"
    );
}

/// Test that pessimism discount is heavier when few samples improve.
#[test]
fn test_pessimism_discount_few_improved() {
    let high_ratio = apply_pessimism_discount(0.05, 90, 100);
    let low_ratio = apply_pessimism_discount(0.05, 30, 100);

    assert!(
        high_ratio > low_ratio,
        "Higher improved ratio should produce larger gain: high={high_ratio}, low={low_ratio}"
    );
}

/// Test that pessimism discount handles zero improved count gracefully.
#[test]
fn test_pessimism_discount_none_improved() {
    let discounted = apply_pessimism_discount(0.05, 0, 100);

    // discount = 0.15 + 0.85 × 0.0 = 0.15
    // discounted = 0.05 × 0.15 = 0.0075
    assert!(
        discounted > 0.0,
        "Discount should not zero-out prediction entirely, got {discounted}"
    );
    assert!(
        discounted < 0.01,
        "Zero improvements should produce a very small gain, got {discounted}"
    );
    assert!(
        (discounted - 0.0075).abs() < 0.001,
        "Expected ~0.0075 (floor discount), got {discounted}"
    );
}

/// Test that pessimism discount handles zero total count (degenerate case).
#[test]
fn test_pessimism_discount_zero_total() {
    let discounted = apply_pessimism_discount(0.05, 0, 0);

    // Should handle gracefully: uses floor only
    assert!(
        discounted >= 0.0,
        "Should not produce negative result, got {discounted}"
    );
    assert!(
        (discounted - 0.05 * 0.15).abs() < 0.001,
        "Zero total should use floor discount, got {discounted}"
    );
}

/// Test that pessimism discount handles negative gain (shouldn't change sign).
#[test]
fn test_pessimism_discount_negative_gain() {
    let discounted = apply_pessimism_discount(-0.03, 50, 100);

    // Negative gain should remain negative after discount
    assert!(
        discounted <= 0.0,
        "Negative gain should remain non-positive after discount, got {discounted}"
    );
}

/// Test that the discount provides meaningful reduction for partial improvement.
///
/// The issue reports a 95% failure rate with over-estimated predictions.
/// When only ~30% of samples improve (a common weak-signal scenario),
/// the discount should reduce the prediction substantially.
#[test]
fn test_pessimism_discount_reduces_weak_signal_significantly() {
    let raw_gain = 0.0205;
    // Only 30% of samples improved — weak signal
    let discounted = apply_pessimism_discount(raw_gain, 60, 200);

    // Issue #733: With concave curve, 30% ratio gives a higher discount than
    // the old linear formula, but the prediction should still be meaningfully
    // reduced (at least 40% reduction).
    assert!(
        discounted < raw_gain * 0.6,
        "Weak signal (30% improved) should reduce prediction by at least 40%: raw={raw_gain}, discounted={discounted}"
    );
    assert!(discounted > 0.0, "Discounted gain should remain positive");
}

/// Integration test: Verify that synapse analysis candidates include the pessimism discount.
///
/// Creates a minimal network with clear error correlation and verifies that
/// the pessimism discount is applied (gain is less than what it would be without
/// the discount). We compare candidates' gain against their `improved_count` ratio.
#[test]
fn test_issue_506_synapse_candidates_have_pessimism_discount() {
    skip_without_gpu!();

    use neat_ai_discovery::parquet_format::write_records_to_parquet;
    use neat_ai_discovery::types::DiscoverRecord;
    use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
    use tempfile::NamedTempFile;

    // Network: input-0 -> output-0 (IDENTITY)
    // input-1 is NOT connected (candidate for add-synapse)
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: vec![SynapseJson {
            from_uuid: "input-0".to_string(),
            to_uuid: "output-0".to_string(),
            weight: 0.5,
            synapse_type: None,
        }],
        input: 2,
        output: 1,
    };

    // Create data with moderate positive correlation between input-1 and output error.
    // Use enough samples for a reliable prediction.
    let mut records = Vec::new();
    for obs in 0..200 {
        let input0_activation = (obs as f32 * 0.1).sin();
        let input1_activation = (obs as f32 - 100.0) / 100.0; // -1 to 1
        let output_error = 0.1 * input1_activation; // Clear correlation
        let output_value = input0_activation * 0.5;

        records.push(DiscoverRecord::new(
            obs as u32,
            "input-0".to_string(),
            Some(input0_activation),
            input0_activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs as u32,
            "input-1".to_string(),
            Some(input1_activation),
            input1_activation,
            Vec::new(),
        ));
        records.push(DiscoverRecord::new(
            obs as u32,
            "output-0".to_string(),
            Some(output_value),
            output_value,
            vec![output_error],
        ));
    }

    let parquet_file = NamedTempFile::new().unwrap();
    let parquet_path = parquet_file.path().to_string_lossy().to_string();
    write_records_to_parquet(&parquet_path, &records).unwrap();

    let input = AnalyzeSynapsesInput {
        creature,
        parquet_file: parquet_path,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: None,
        random_seed: Some(42),
        module_outcome_tracker: None,
        temperature: 1.0,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input).unwrap();

    // Verify that pessimism discount was applied: the expected_creature_score_gain
    // should be consistent with the pessimism discount formula applied to each
    // candidate's improved_count / total_count ratio.
    for candidate in &result.helpful_synapses {
        assert!(
            candidate.expected_creature_score_gain > 0.0,
            "Issue #506: Candidate should have positive expected gain after discount",
        );

        // The pessimism discount is: floor + (1-floor) × (improved_count / total_count)
        // For a candidate where most samples improve, this should be close to 1.0.
        // For a candidate where few improve, it should be closer to the floor (0.15).
        let ratio = if candidate.total_count > 0 {
            candidate.improved_count as f32 / candidate.total_count as f32
        } else {
            0.0
        };

        // Verify the discount was actually applied: gain should be less than or equal
        // to what it would be without discount (when ratio < 1.0)
        if ratio < 0.99 {
            // With discount, gain should be reduced compared to no-discount case.
            // We verify this indirectly by checking the gain is less than
            // error_reduction × impact × boost_max (1.5 × 1.5 = 2.25).
            let max_undiscounted = candidate.expected_creature_error_reduction * 2.25;
            assert!(
                candidate.expected_creature_score_gain <= max_undiscounted + 0.001,
                "Issue #506: Gain {} should not exceed max undiscounted {} for ratio {}",
                candidate.expected_creature_score_gain,
                max_undiscounted,
                ratio,
            );
        }
    }
}
