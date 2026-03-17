//! Regression test for Issue #180 (9-Jan-2026).
//!
//! When adjusting an existing synapse's weight, the coordinated structural candidate should
//! use a single `setWeight` operation instead of the previous `removeSynapse` + `addSynapse`
//! pattern. This simplifies the representation and more directly expresses the intent:
//! "change the weight of this synapse" rather than "remove and recreate with new weight".
//!
//! Before (Issue #180):
//! ```json
//! {
//!   "operations": [
//!     { "type": "removeSynapse", "fromNeuronUuid": "...", "toNeuronUuid": "..." },
//!     { "type": "addSynapse", "fromNeuronUuid": "...", "toNeuronUuid": "...", "weight": 0.006 }
//!   ],
//!   "comment": "Adjust synapse weight (remove+add): old=-0.001, new=0.006, delta=0.007"
//! }
//! ```
//!
//! After (Issue #180):
//! ```json
//! {
//!   "operations": [
//!     { "type": "setWeight", "fromNeuronUuid": "...", "toNeuronUuid": "...", "weight": 0.006 }
//!   ],
//!   "comment": "Adjust synapse weight: old=-0.001, new=0.006, delta=0.007"
//! }
//! ```

use crate::skip_without_gpu;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{
    AnalyzeSynapsesInput, CoordinatedStructuralOpJson, CreatureJson, NeuronJson, SynapseJson,
};
use tempfile::NamedTempFile;

#[test]
fn issue_180_setweight_replaces_remove_add_pattern() {
    skip_without_gpu!();

    // Minimal creature:
    // - input-0 (implicit via creature.input)
    // - output-0 (explicit neuron)
    // - Existing synapse: input-0 → output-0 with a suboptimal weight
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
            weight: -0.001, // Suboptimal weight - discovery should suggest adjustment
            synapse_type: None,
        }],
        input: 1,
        output: 1,
    };

    // Records:
    // - input-0 has varying activation (not constant, so won't fold to setBias)
    // - output-0 has positive error that correlates with input activation
    //   (suggesting a positive weight adjustment would help)
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count: u32 = 64;
    for obs_index in 0..sample_count {
        // Varying input activation
        let input_activation = (obs_index as f32 / sample_count as f32) * 2.0 - 1.0; // [-1, 1]
        let output_activation = input_activation * -0.001; // Current output (with suboptimal weight)
        let output_value = output_activation; // IDENTITY squash
        // Error correlates with input: positive input should give positive output
        let error = input_activation * 0.1 - output_activation;

        records.push(DiscoverRecord::new(
            obs_index,
            "input-0".to_string(),
            Some(input_activation),
            input_activation,
            Vec::new(),
        ));

        records.push(DiscoverRecord::new(
            obs_index,
            "output-0".to_string(),
            Some(output_value),
            output_activation,
            vec![error],
        ));
    }

    let temp_file = NamedTempFile::new().expect("Failed to create temp parquet file");
    let parquet_file = temp_file
        .path()
        .to_str()
        .expect("temp path should be valid UTF-8")
        .to_string();
    write_records_to_parquet(&parquet_file, &records).expect("Failed to write parquet test data");

    let input = AnalyzeSynapsesInput {
        parquet_file,
        creature,
        focus_neurons: vec!["output-0".to_string()],
        max_candidates: Some(10),
        analysis_deadline_ms: Some(30_000),
        random_seed: None,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input)
        .expect("Synapse analysis should succeed");

    // Expected behaviour (post-fix):
    // - A coordinated-structural candidate with a single setWeight operation
    // - NOT a removeSynapse + addSynapse pair
    assert!(
        !result.coordinated_structural_candidates.is_empty(),
        "expected at least one coordinated structural candidate for weight adjustment"
    );

    // Find the candidate that adjusts the weight on the input-0 → output-0 synapse
    let weight_adjustment_candidate = result.coordinated_structural_candidates.iter().find(|c| {
        c.operations.iter().any(|op| match op {
            CoordinatedStructuralOpJson::SetWeight {
                from_neuron_uuid,
                to_neuron_uuid,
                ..
            } => from_neuron_uuid == "input-0" && to_neuron_uuid == "output-0",
            _ => false,
        })
    });

    assert!(
        weight_adjustment_candidate.is_some(),
        "expected a setWeight candidate for input-0 → output-0 synapse, but found none.\n\
         Coordinated candidates: {:?}",
        result.coordinated_structural_candidates
    );

    let candidate = weight_adjustment_candidate.unwrap();

    // Should be a single operation, not remove+add
    assert_eq!(
        candidate.operations.len(),
        1,
        "expected a single setWeight operation, not remove+add pattern"
    );

    match &candidate.operations[0] {
        CoordinatedStructuralOpJson::SetWeight {
            from_neuron_uuid,
            to_neuron_uuid,
            weight,
        } => {
            assert_eq!(from_neuron_uuid, "input-0");
            assert_eq!(to_neuron_uuid, "output-0");
            // The new weight should be different from the old weight (-0.001)
            // and should move in the positive direction based on the error pattern
            assert!(
                *weight > -0.001,
                "expected weight to be adjusted upward from -0.001, got {weight}"
            );
        }
        other => panic!("expected setWeight operation, got {other:?}"),
    }

    // Verify the comment no longer says "remove+add"
    if let Some(comment) = &candidate.comment {
        assert!(
            !comment.contains("remove+add"),
            "comment should not reference remove+add pattern: {comment}"
        );
        assert!(
            comment.contains("Adjust synapse weight"),
            "comment should describe weight adjustment: {comment}"
        );
    }
}

#[test]
fn setweight_json_serialization_format() {
    // Test that SetWeight serializes to the expected JSON format
    let op = CoordinatedStructuralOpJson::SetWeight {
        from_neuron_uuid: "input-0".to_string(),
        to_neuron_uuid: "output-0".to_string(),
        weight: 0.006,
    };

    let json = serde_json::to_string(&op).expect("serialization should succeed");

    // Should serialize with type="setWeight" and camelCase field names
    assert!(
        json.contains(r#""type":"setWeight""#),
        "expected type=setWeight, got: {json}"
    );
    assert!(
        json.contains(r#""fromNeuronUuid":"input-0""#),
        "expected fromNeuronUuid, got: {json}"
    );
    assert!(
        json.contains(r#""toNeuronUuid":"output-0""#),
        "expected toNeuronUuid, got: {json}"
    );
    assert!(
        json.contains(r#""weight":0.006"#) || json.contains(r#""weight":0.006"#),
        "expected weight, got: {json}"
    );
}
