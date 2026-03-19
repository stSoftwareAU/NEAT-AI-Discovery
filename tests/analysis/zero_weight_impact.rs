//! Test for Issue #130 extension: Zero-weight synapses causing NaN impact.
//!
//! When all inbound synapses to a neuron have weight == 0.0, the normalised
//! impact formula divides by zero: |w| / total → 0.0 / 0.0 → NaN.
//!
//! This test verifies the fix: impacts must always be finite numbers.

use neat_ai_discovery::focus::compute_impacts_public;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};

/// Test: Zero-weight synapses should not produce NaN impacts.
///
/// Network:
///   hidden → output (weight 0.0)
///
/// With the bug: `total_inbound_weight` = 0.0, so 0.0 / 0.0 = NaN
/// With the fix: impact should be 0.0 (zero contribution)
#[test]
fn test_zero_weight_synapse_does_not_produce_nan() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            // Input to hidden with normal weight
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Hidden to output with ZERO weight (the problematic case)
            SynapseJson {
                from_uuid: "hidden".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.0,
                synapse_type: None,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);

    // Hidden neuron's impact must be finite (not NaN or Inf)
    let hidden_impact = impacts.get("hidden").copied().unwrap_or(f32::NAN);
    assert!(
        hidden_impact.is_finite(),
        "hidden neuron impact should be finite, got {hidden_impact}"
    );

    // Zero-weight synapse means zero contribution → impact = 0.0
    assert!(
        (hidden_impact - 0.0).abs() < 0.001,
        "hidden neuron with zero-weight to output should have impact ~0.0, got {hidden_impact}"
    );

    // Output impact should always be 1.0
    let output_impact = impacts.get("output-0").copied().unwrap_or(f32::NAN);
    assert!(
        (output_impact - 1.0).abs() < 0.001,
        "output should have impact 1.0, got {output_impact}"
    );
}

/// Test: Multiple zero-weight synapses to a neuron.
///
/// Network:
///   hidden-1 ─┬→ output (weight 0.0)
///   hidden-2 ─┘   (weight 0.0)
///
/// Total inbound to output = 0.0, so both neurons compute 0.0 / 0.0 without fix.
#[test]
fn test_multiple_zero_weight_synapses_do_not_produce_nan() {
    let creature = CreatureJson {
        input: 2,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-1".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Both hidden neurons connect to output with ZERO weight
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.0,
                synapse_type: None,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);

    // All impacts must be finite
    for (uuid, impact) in &impacts {
        assert!(
            impact.is_finite(),
            "neuron {uuid} should have finite impact, got {impact}"
        );
    }

    // Hidden neurons with zero-weight should have zero impact
    let h1 = impacts.get("hidden-1").copied().unwrap_or(f32::NAN);
    let h2 = impacts.get("hidden-2").copied().unwrap_or(f32::NAN);
    assert!(
        (h1 - 0.0).abs() < 0.001,
        "hidden-1 with zero-weight should have impact ~0.0, got {h1}"
    );
    assert!(
        (h2 - 0.0).abs() < 0.001,
        "hidden-2 with zero-weight should have impact ~0.0, got {h2}"
    );
}

/// Test: Deep network with zero-weight in the chain.
///
/// Network:
///   hidden-1 → hidden-2 → output
///              (weight 0.0)
///
/// Even though hidden-1 has a non-zero weight to hidden-2, if hidden-2 → output
/// is zero, then hidden-1's effective impact should also be zero (0.0 × anything = 0.0).
#[test]
fn test_deep_network_zero_weight_propagates() {
    let creature = CreatureJson {
        input: 1,
        output: 1,
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-1".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-2".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-1".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-1".to_string(),
                to_uuid: "hidden-2".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            // Zero-weight in the chain
            SynapseJson {
                from_uuid: "hidden-2".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.0,
                synapse_type: None,
            },
        ],
    };

    let impacts = compute_impacts_public(&creature);

    // All impacts must be finite
    for (uuid, impact) in &impacts {
        assert!(
            impact.is_finite(),
            "neuron {uuid} should have finite impact, got {impact}"
        );
    }

    // hidden-2 connects to output with zero weight → impact = 0.0
    let h2 = impacts.get("hidden-2").copied().unwrap_or(f32::NAN);
    assert!(
        (h2 - 0.0).abs() < 0.001,
        "hidden-2 with zero-weight to output should have impact ~0.0, got {h2}"
    );

    // hidden-1's impact propagates through hidden-2, which has zero impact
    // So hidden-1's impact should also be ~0.0
    let h1 = impacts.get("hidden-1").copied().unwrap_or(f32::NAN);
    assert!(
        (h1 - 0.0).abs() < 0.001,
        "hidden-1 should have impact ~0.0 (propagates through zero), got {h1}"
    );
}
