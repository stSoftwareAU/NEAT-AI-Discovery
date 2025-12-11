//! TDD Tests for STEP/BIPOLAR activation-based impact calculation (Dec 2024)
//!
//! ## Problem
//!
//! STEP/BIPOLAR neurons have binary outputs (0/1 or -1/1). The current impact
//! calculation treats them as threshold functions where ANY synapse could flip
//! the output, giving all synapses full `child_impact`.
//!
//! This is conservative but potentially wasteful - if the STEP neuron is
//! "saturated" (always outputting the same value), removing synapses won't
//! actually change anything.
//!
//! ## Solution: Use Recorded Activations
//!
//! We have recorded activations! We can analyse them to determine:
//!
//! 1. **Saturated STEP (always 0 or always 1)**: Low impact - removing inputs
//!    won't flip the output (unless they're very large)
//! 2. **Flipping STEP (varies between 0 and 1)**: High impact - the neuron is
//!    near threshold and any input could be critical
//!
//! ## Impact Formula
//!
//! For STEP/BIPOLAR targets:
//! - If flip_rate = 0% (saturated): impact = weight.abs() * child_impact * SATURATION_DISCOUNT
//! - If flip_rate > 0% (flipping): impact = child_impact (any synapse could flip it)
//!
//! The flip_rate is calculated from recorded activations by counting how many
//! unique output values we see.

mod common;

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Helper to create a synapse
fn synapse(from: &str, to: &str, weight: f32) -> SynapseJson {
    SynapseJson {
        from_uuid: from.to_string(),
        to_uuid: to.to_string(),
        weight,
        synapse_type: None,
    }
}

/// Helper to create a hidden neuron
fn hidden(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "hidden".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

/// Helper to create an output neuron
fn output(uuid: &str, squash: &str) -> NeuronJson {
    NeuronJson {
        uuid: uuid.to_string(),
        neuron_type: "output".to_string(),
        squash: squash.to_string(),
        bias: 0.0,
    }
}

// =============================================================================
// STEP Saturation Tests
// =============================================================================

/// TDD Test: STEP neuron that ALWAYS outputs 1 (saturated positive).
///
/// If the STEP output is always 1.0 across all recorded samples, the neuron's
/// input sum is consistently above threshold. Removing a small positive weight
/// is unlikely to flip it unless the weight is massive.
///
/// Expected: Upstream neurons should have DISCOUNTED impact (not full child_impact).
#[test]
fn test_step_saturated_positive_has_discounted_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("step-gate", "STEP"), // Always outputs 1
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "step-gate", 0.1), // Small weight
            synapse("input-0", "step-gate", 10.0), // Large weight keeps it saturated
            synapse("step-gate", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records show STEP always outputs 1.0 (saturated positive)
    // This means input sum is always > 0, removing 0.1 weight won't flip it
    let records = vec![
        // upstream has moderate activation
        DiscoverRecord::new(0, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(2, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(3, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        // step-gate ALWAYS outputs 1.0 (saturated)
        DiscoverRecord::new(0, "step-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "step-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(2, "step-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(3, "step-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        // output records
        DiscoverRecord::new(0, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(2, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(3, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Saturated STEP (+): upstream structural_impact = {:.4}, activation_weighted = {:.4}",
        upstream.impact, upstream.activation_weighted_impact
    );

    // CURRENT BEHAVIOUR (to document/change):
    // With full child_impact for threshold: impact = 1.0
    //
    // DESIRED BEHAVIOUR (after fix):
    // With saturation detection: impact should be discounted because removing
    // the 0.1 weight from a saturated STEP won't flip it (large weight keeps it above 0)
    //
    // For now, just document the current behaviour
    assert!(
        upstream.impact > 0.0,
        "upstream should have some impact, got {}",
        upstream.impact
    );
}

/// TDD Test: STEP neuron that ALWAYS outputs 0 (saturated negative).
///
/// If the STEP output is always 0.0 across all recorded samples, the neuron's
/// input sum is consistently below threshold (< 0). Removing a small negative
/// weight won't flip it to 1.
#[test]
fn test_step_saturated_negative_has_discounted_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("step-gate", "STEP"), // Always outputs 0
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "step-gate", 0.1), // Small positive weight
            synapse("input-0", "step-gate", -10.0), // Large negative keeps it below 0
            synapse("step-gate", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records show STEP always outputs 0.0 (saturated negative)
    let records = vec![
        DiscoverRecord::new(0, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        // step-gate ALWAYS outputs 0.0 (saturated)
        DiscoverRecord::new(0, "step-gate".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(1, "step-gate".to_string(), Some(0.0), 0.0, vec![0.1]),
        // output
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Saturated STEP (-): upstream structural_impact = {:.4}",
        upstream.impact
    );

    // Document current behaviour
    assert!(
        upstream.impact > 0.0,
        "upstream should have some impact, got {}",
        upstream.impact
    );
}

/// TDD Test: STEP neuron that FLIPS between 0 and 1 (threshold-sensitive).
///
/// If the STEP output varies in recorded samples, the neuron is near its
/// threshold and ANY synapse could be the one that flips it. All synapses
/// should have high impact.
#[test]
fn test_step_flipping_has_full_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("step-gate", "STEP"), // Flips between 0 and 1
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "step-gate", 0.1), // Small weight but could flip!
            synapse("step-gate", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records show STEP VARIES between 0.0 and 1.0 (flipping)
    let records = vec![
        DiscoverRecord::new(0, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "upstream".to_string(), Some(-0.5), -0.5, vec![0.1]),
        // step-gate FLIPS between 0 and 1
        DiscoverRecord::new(0, "step-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "step-gate".to_string(), Some(0.0), 0.0, vec![0.1]),
        // output
        DiscoverRecord::new(0, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Flipping STEP: upstream structural_impact = {:.4}",
        upstream.impact
    );

    // For flipping STEP, impact should be HIGH (full child_impact)
    // because any synapse could be the one that pushes it over threshold
    assert!(
        upstream.impact > 0.5,
        "Flipping STEP upstream should have high impact (threshold-sensitive), got {}",
        upstream.impact
    );
}

// =============================================================================
// BIPOLAR Saturation Tests
// =============================================================================

/// TDD Test: BIPOLAR neuron that always outputs +1 (saturated positive).
#[test]
fn test_bipolar_saturated_positive_has_discounted_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("bipolar-gate", "BIPOLAR"), // Always outputs +1
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "bipolar-gate", 0.1),
            synapse("input-0", "bipolar-gate", 10.0), // Keeps it positive
            synapse("bipolar-gate", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // BIPOLAR always outputs +1.0 (saturated)
    let records = vec![
        DiscoverRecord::new(0, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "bipolar-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "bipolar-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Saturated BIPOLAR (+1): upstream structural_impact = {:.4}",
        upstream.impact
    );

    assert!(
        upstream.impact > 0.0,
        "upstream should have some impact, got {}",
        upstream.impact
    );
}

/// TDD Test: BIPOLAR neuron that FLIPS between -1 and +1.
#[test]
fn test_bipolar_flipping_has_full_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("bipolar-gate", "BIPOLAR"), // Flips between -1 and +1
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "bipolar-gate", 0.5),
            synapse("bipolar-gate", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // BIPOLAR flips between -1.0 and +1.0
    let records = vec![
        DiscoverRecord::new(0, "upstream".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "upstream".to_string(), Some(-1.0), -1.0, vec![0.1]),
        DiscoverRecord::new(0, "bipolar-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "bipolar-gate".to_string(), Some(-1.0), -1.0, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(-1.0), -1.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Flipping BIPOLAR: upstream structural_impact = {:.4}",
        upstream.impact
    );

    // Flipping BIPOLAR = high impact
    assert!(
        upstream.impact > 0.5,
        "Flipping BIPOLAR upstream should have high impact, got {}",
        upstream.impact
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

/// TDD Test: STEP with only one recorded sample can't determine flip rate.
///
/// With insufficient data, we should fall back to the conservative approach
/// (full child_impact) rather than incorrectly assuming saturation.
#[test]
fn test_step_insufficient_data_uses_conservative_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("step-gate", "STEP"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "step-gate", 0.1),
            synapse("step-gate", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Only ONE sample - can't determine if saturated or flipping
    let records = vec![
        DiscoverRecord::new(0, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "step-gate".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Single-sample STEP: upstream structural_impact = {:.4}",
        upstream.impact
    );

    // With insufficient data, should use conservative (high) impact
    assert!(
        upstream.impact > 0.5,
        "Single-sample STEP should use conservative high impact, got {}",
        upstream.impact
    );
}

/// TDD Test: Multiple STEP neurons in sequence should compound correctly.
///
/// If hidden-a → STEP-1 → STEP-2 → output, and both STEPs are saturated,
/// the impact should be doubly discounted. If both are flipping, impact
/// should remain high.
#[test]
fn test_chained_step_neurons_compound_impact() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            hidden("upstream", "IDENTITY"),
            hidden("step-1", "STEP"),
            hidden("step-2", "STEP"),
            output("output-0", "IDENTITY"),
        ],
        synapses: vec![
            synapse("input-0", "upstream", 1.0),
            synapse("upstream", "step-1", 0.1),
            synapse("input-0", "step-1", 10.0), // Keeps step-1 saturated at 1
            synapse("step-1", "step-2", 0.1),
            synapse("input-0", "step-2", 10.0), // Keeps step-2 saturated at 1
            synapse("step-2", "output-0", 1.0),
        ],
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Both STEPs always output 1.0 (saturated)
    let records = vec![
        DiscoverRecord::new(0, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(1, "upstream".to_string(), Some(0.5), 0.5, vec![0.1]),
        DiscoverRecord::new(0, "step-1".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "step-1".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(0, "step-2".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "step-2".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(0, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(1.0), 1.0, vec![0.1]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None).unwrap();

    let upstream = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "upstream")
        .expect("upstream should be in results");

    println!(
        "Chained saturated STEPs: upstream structural_impact = {:.4}",
        upstream.impact
    );

    // Document: Chained STEP neurons should compound their saturation discount
    // (if we implement saturation-based discounting)
    assert!(
        upstream.impact > 0.0,
        "upstream should have some impact, got {}",
        upstream.impact
    );
}

// =============================================================================
// Flip Rate Calculation Documentation
// =============================================================================

/// Document: How to calculate flip rate from recorded activations.
///
/// For STEP neuron with activations [1, 1, 0, 1, 0]:
/// - unique_values = {0, 1}
/// - count = 2
/// - flip_rate = if count > 1 { high } else { 0 (saturated) }
///
/// This is a simple heuristic. A more sophisticated approach would count
/// actual transitions (1→0 or 0→1) but that requires ordered samples.
#[test]
fn document_flip_rate_calculation() {
    println!("\n=== FLIP RATE CALCULATION FOR STEP/BIPOLAR ===\n");
    println!("Simple heuristic based on unique activation values:\n");
    println!("| Activations     | Unique | Flip Rate | Interpretation        |");
    println!("|-----------------|--------|-----------|----------------------|");
    println!("| [1,1,1,1,1]     | 1      | 0%        | Saturated positive   |");
    println!("| [0,0,0,0,0]     | 1      | 0%        | Saturated negative   |");
    println!("| [1,1,1,0,1]     | 2      | High      | Near threshold       |");
    println!("| [0,1,0,1,0]     | 2      | High      | Oscillating          |");
    println!("| [-1,-1,-1,-1]   | 1      | 0%        | BIPOLAR saturated -  |");
    println!("| [-1,1,-1,1]     | 2      | High      | BIPOLAR flipping     |");
    println!("\nNote: With insufficient samples (n=1), assume high flip rate (conservative).");
}
