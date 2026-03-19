//! Exploratory ablation candidates (high error) — DISABLED (Issue #414).
//!
//! Previously, we returned "exploratory ablation" removal candidates for hidden neurons
//! with extremely high error (≥10× max output error), even when they were NOT low impact.
//!
//! This was based on the assumption that high error means the neuron is harmful.
//! Production data showed this had a 0% success rate (0/2 attempts) because:
//!
//! **High error ≠ harmful neuron**
//!
//! A neuron with high recorded error is often:
//! 1. Handling the most difficult samples (it's the only path for hard cases)
//! 2. Receiving bad inputs from upstream (the error is a symptom, not a cause)
//! 3. Fighting against incorrect biases elsewhere in the network
//!
//! Removing such neurons typically makes performance WORSE.
//!
//! This test verifies that high-error neurons are NO LONGER returned as exploratory
//! ablation candidates (Issue #414 fix). The legitimate "safe prune" removal candidates
//! (based on `activation_weighted_impact` < costOfGrowth) remain active with a 17.6% success rate.

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

/// Issue #414: High-error neurons should NOT be returned as removal candidates.
///
/// Previously, a hidden neuron with high error (≥10× max output error) would be returned
/// as an "exploratory ablation candidate". This test verifies that behaviour is now disabled.
#[test]
fn high_error_hidden_neuron_is_not_returned_as_exploratory_ablation_candidate() {
    // Simple creature:
    // input-0 -> bad (hidden) -> output-0
    //
    // The "bad" neuron has high structural impact and high activation, so it is NOT a safe
    // prune candidate. It also has extreme recorded error.
    //
    // BEFORE Issue #414: This would be returned as an exploratory ablation candidate.
    // AFTER Issue #414: This should NOT be returned as a removal candidate at all.
    let creature = CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "bad".to_string(),
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
                to_uuid: "bad".to_string(),
                weight: 1.0,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "bad".to_string(),
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
    // - bad has *huge* error, but also high activation (so it's high impact in practice)
    //
    // We include at least one record per selectable neuron. Inputs are implied.
    let records = vec![
        DiscoverRecord::new(0, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(1, "output-0".to_string(), Some(0.0), 0.0, vec![0.1]),
        DiscoverRecord::new(0, "bad".to_string(), Some(0.0), 1.0, vec![100.0]),
        DiscoverRecord::new(1, "bad".to_string(), Some(0.0), 1.0, vec![100.0]),
    ];
    write_records_to_parquet(file_path, &records).unwrap();

    let result = rank_focus_neurons(file_path, &creature, None, None).unwrap();

    // Sanity: "bad" should not be a safe prune candidate under costOfGrowth=1e-7.
    let bad_rank = result
        .neurons
        .iter()
        .find(|n| n.neuron_uuid == "bad")
        .expect("bad neuron should be ranked");
    assert!(
        bad_rank.activation_weighted_impact > 1e-6,
        "bad neuron should have high activation_weighted_impact, got {:.2e}",
        bad_rank.activation_weighted_impact
    );

    // Issue #414: "bad" should NOT be returned as a removal candidate.
    // High-error exploratory ablation has been disabled because error magnitude
    // does not correlate with whether removing a neuron improves the network.
    let bad_candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "bad");

    assert!(
        bad_candidate.is_none(),
        "High-error neurons should NOT be returned as removal candidates. \
         Found: {:?}",
        bad_candidate.map(|c| &c.reason)
    );
}
