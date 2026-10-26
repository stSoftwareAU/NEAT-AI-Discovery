//! Issue #2346: exercise the output ceiling through the API available before
//! the fix, so this regression can run against both base and head source.

#![allow(clippy::cast_precision_loss)] // Small fixture sample indices fit exactly in f32.

use neat_ai_discovery::analysis::detection::correlated_error::detect_correlated_error_patterns;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{CreatureJson, NeuronJson};

fn correlated_outputs(output_count: usize) -> (CreatureJson, Vec<(String, Vec<DiscoverRecord>)>) {
    let neurons: Vec<NeuronJson> = (0..output_count)
        .map(|i| NeuronJson {
            uuid: format!("output-{i}"),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        })
        .collect();
    let records = neurons
        .iter()
        .map(|neuron| {
            let samples = (0..20)
                .map(|i| {
                    DiscoverRecord::new(
                        i,
                        neuron.uuid.clone(),
                        Some(0.5),
                        0.5,
                        vec![(i + 1) as f32 * 0.1],
                    )
                })
                .collect();
            (neuron.uuid.clone(), samples)
        })
        .collect();
    let creature = CreatureJson {
        neurons,
        synapses: vec![],
        input: 0,
        output: output_count,
    };
    (creature, records)
}

#[test]
fn rejects_output_count_above_ceiling_without_building_the_matrix() {
    let (control, control_records) = correlated_outputs(3);
    assert!(
        !detect_correlated_error_patterns(&control, &control_records).is_empty(),
        "the correlated fixture must produce groups below the output ceiling"
    );

    // Pin the 1,000-output contract without importing a constant added by the
    // fix. This bounded fixture is also safe to execute on the unfixed source.
    let (creature, records) = correlated_outputs(1_001);
    assert!(
        detect_correlated_error_patterns(&creature, &records).is_empty(),
        "1,001 eligible outputs must be rejected by the 1,000-output ceiling"
    );
}
