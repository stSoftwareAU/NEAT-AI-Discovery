//! Issue #414: Remove-neuron (high error) discovery should be disabled.
//!
//! The high-error exploratory ablation discovery type had a 0% success rate (0/2 attempts)
//! because the underlying assumption is fundamentally flawed:
//!
//! **High error ≠ harmful neuron**
//!
//! A neuron with high recorded error is not necessarily a neuron that should be removed.
//! In fact, high-error neurons are often:
//! 1. Handling the most difficult samples (they're the only path for hard cases)
//! 2. Receiving bad inputs from upstream (the error is a symptom, not a cause)
//! 3. Fighting against incorrect biases elsewhere in the network
//!
//! Removing such neurons typically makes performance **worse** because:
//! - The difficult samples lose their only computation path
//! - The network loses the only neuron attempting to handle a specific pattern
//!
//! This test verifies that high-error neurons are NO LONGER returned as exploratory
//! ablation candidates, since the discovery type has been disabled.

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Test that high-error neurons are NOT returned as exploratory ablation candidates.
///
/// Previously, neurons with high error (≥10× max output error) were returned as
/// "exploratory ablation candidates". This was based on the flawed assumption that
/// high error means the neuron is harmful.
///
/// This test verifies the fix: high-error neurons are no longer suggested for removal.
#[test]
fn high_error_neuron_is_not_returned_as_exploratory_ablation_candidate() {
    // Simple creature:
    // input-0 -> high-error-hidden -> output-0
    //
    // The "high-error-hidden" neuron has:
    // - High structural impact (only path to output)
    // - High activation (not dormant)
    // - Extremely high recorded error
    //
    // Previously, this would be returned as an exploratory ablation candidate.
    // After the fix, it should NOT be suggested for removal.
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "high-error-hidden".to_string(),
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
                to_uuid: "high-error-hidden".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "high-error-hidden".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records:
    // - output has low error (so max_output_error stays small)
    // - high-error-hidden has HUGE error (100.0), which is 1000× the output error (0.1)
    //   This would have triggered the old EXPLORATORY_ERROR_MULTIPLIER of 10.0
    let records = vec![
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(
            0,
            "high-error-hidden".to_string(),
            Some(0.0),
            1.0, // High activation = high impact
            vec![100.0],
        ),
        DiscoverRecord::new(
            1,
            "high-error-hidden".to_string(),
            Some(0.0),
            1.0, // High activation = high impact
            vec![100.0],
        ),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Verify: high-error-hidden should NOT be in removal_candidates
    // because high-error exploratory ablation is now disabled.
    let high_error_in_removal = result
        .removal_candidates
        .iter()
        .any(|c| c.neuron_uuid == "high-error-hidden");

    assert!(
        !high_error_in_removal,
        "high-error-hidden should NOT be returned as a removal candidate. \
         High-error exploratory ablation has been disabled because error magnitude \
         does not correlate with whether removing a neuron improves the network. \
         Found removal candidates: {:?}",
        result
            .removal_candidates
            .iter()
            .map(|c| &c.neuron_uuid)
            .collect::<Vec<_>>()
    );
}

/// Test that low-impact neurons are STILL returned as removal candidates.
///
/// This ensures we haven't broken the legitimate removal candidate detection.
/// Low-impact neurons (`activation_weighted_impact` < costOfGrowth) should still
/// be suggested for removal because they genuinely don't contribute to the network.
#[test]
fn low_impact_neurons_still_returned_as_removal_candidates() {
    // Creature with a low-impact hidden neuron:
    // input-0 -> low-impact-hidden -> output-0
    //            (but dormant/low activation)
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "low-impact-hidden".to_string(),
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
                to_uuid: "low-impact-hidden".to_string(),
                weight: 1e-6, // Tiny weight
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "low-impact-hidden".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-6, // Tiny weight
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    // Records with near-zero activation = very low impact
    let records = vec![
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(
            0,
            "low-impact-hidden".to_string(),
            Some(0.0),
            1e-10, // Near-zero activation
            vec![0.01],
        ),
        DiscoverRecord::new(
            1,
            "low-impact-hidden".to_string(),
            Some(0.0),
            1e-10, // Near-zero activation
            vec![0.01],
        ),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Verify: low-impact-hidden SHOULD still be in removal_candidates
    // because it has genuinely low activation_weighted_impact
    let low_impact_candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "low-impact-hidden");

    assert!(
        low_impact_candidate.is_some(),
        "low-impact-hidden should be returned as a removal candidate. \
         Low-impact neurons (activation_weighted_impact < costOfGrowth) should \
         still be suggested for removal."
    );

    // Verify the reason does NOT mention "exploratory ablation"
    let candidate = low_impact_candidate.unwrap();
    assert!(
        !candidate.reason.to_lowercase().contains("exploratory"),
        "Low-impact removal candidates should not be labelled as 'exploratory'. \
         Found reason: {}",
        candidate.reason
    );
}

/// Test that the removal candidate reason reflects the correct basis for removal.
///
/// Removal candidates should be based on `activation_weighted_impact` being below
/// the costOfGrowth threshold, not on error magnitude.
#[test]
fn removal_candidate_reason_reflects_impact_not_error() {
    // Creature with a truly low-impact hidden neuron
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "dormant".to_string(),
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
                to_uuid: "dormant".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "dormant".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 1e-8, // Very tiny outgoing weight
                synapse_type: None,
            },
        ],
        input: 1,
        output: 1,
    };

    let temp_file = NamedTempFile::new().unwrap();
    let file_path = temp_file.path().to_str().unwrap();

    let records = vec![
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(
            0,
            "dormant".to_string(),
            Some(0.0),
            1e-10, // Near-zero activation
            vec![0.01],
        ),
        DiscoverRecord::new(
            1,
            "dormant".to_string(),
            Some(0.0),
            1e-10, // Near-zero activation
            vec![0.01],
        ),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    let dormant_candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "dormant");

    assert!(
        dormant_candidate.is_some(),
        "Dormant neuron should be a removal candidate"
    );

    let reason = &dormant_candidate.unwrap().reason;

    // The reason should mention impact-based metrics, not high error
    assert!(
        reason.to_lowercase().contains("impact")
            || reason.to_lowercase().contains("costofgrowth")
            || reason.to_lowercase().contains("cost of growth"),
        "Removal candidate reason should mention impact or costOfGrowth. Found: {reason}"
    );

    // The reason should NOT mention "high error" or "exploratory ablation"
    assert!(
        !reason.to_lowercase().contains("high error"),
        "Removal candidate reason should not mention 'high error' (disabled). Found: {reason}"
    );
    assert!(
        !reason.to_lowercase().contains("exploratory ablation"),
        "Removal candidate reason should not mention 'exploratory ablation' (disabled). Found: {reason}"
    );
}
