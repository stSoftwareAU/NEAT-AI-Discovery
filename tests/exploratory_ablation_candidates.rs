//! Exploratory ablation candidates (high error).
//!
//! We already return "safe prune" removal candidates based on low activation-weighted
//! impact (activation_weighted_impact < costOfGrowth). This test verifies we ALSO return
//! an *extra* removal candidate when a hidden neuron has extremely high error, even when
//! it is NOT low impact.
//!
//! This supports the controller-side ablation test workflow: try removing a suspicious
//! high-error neuron and keep the mutation only if full-dataset score improves.

mod common;

use neat_ai_discovery::focus::rank_focus_neurons;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

#[test]
fn high_error_hidden_neuron_is_returned_as_exploratory_ablation_candidate() {
    // Simple creature:
    // input-0 -> bad (hidden) -> output-0
    //
    // The "bad" neuron has high structural impact and high activation, so it is NOT a safe
    // prune candidate. However, it has extreme recorded error, so we want it returned as an
    // exploratory ablation candidate (with a warning-style reason).
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

    // New behaviour: "bad" is still returned as a removal candidate, but marked as exploratory.
    let bad_candidate = result
        .removal_candidates
        .iter()
        .find(|c| c.neuron_uuid == "bad")
        .expect("bad neuron should be returned as an exploratory ablation candidate");

    assert_eq!(
        bad_candidate.expected_error_reduction, 0.0,
        "exploratory candidates should not claim an expected error reduction"
    );
    assert!(
        bad_candidate
            .reason
            .to_lowercase()
            .contains("exploratory ablation"),
        "expected reason to mention exploratory ablation, got: {}",
        bad_candidate.reason
    );
}
