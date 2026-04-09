//! Regression test for Issue #178 (7-Jan-2026).
//!
//! When a source activation is constant (activation range ≈ 0), an add-synapse from that
//! source behaves like a bias adjustment on the downstream neuron:
//!   contribution = weight * activation ≈ constant
//!
//! In this case we should emit a coordinated-structural `setBias` operation rather than
//! proposing a new synapse, which would pay complexity cost for what is effectively an
//! intercept shift.

use crate::skip_without_gpu;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeSynapsesInput, CreatureJson, NeuronJson, SynapseJson};
use tempfile::NamedTempFile;

#[test]
fn issue_178_constant_source_becomes_setbias_candidate() {
    skip_without_gpu!();

    // Minimal creature:
    // - input-0 (implicit via creature.input)
    // - output-0 (explicit neuron)
    //
    // No existing synapse input-0 → output-0, so synapse analysis would normally propose it.
    let creature = CreatureJson {
        neurons: vec![NeuronJson {
            uuid: "output-0".to_string(),
            neuron_type: "output".to_string(),
            squash: "IDENTITY".to_string(),
            bias: 0.0,
        }],
        synapses: Vec::<SynapseJson>::new(),
        input: 1,
        output: 1,
    };

    // Records:
    // - input-0 activation is constant (range = 0)
    // - output-0 error is constant and positive (so a constant offset helps)
    let mut records: Vec<DiscoverRecord> = Vec::new();
    let sample_count: u32 = 32;
    for obs_index in 0..sample_count {
        let input_activation = 1.0f32; // constant
        let output_activation = 0.0f32;
        let output_value = 0.0f32;
        let error = 0.2f32; // constant VALUE-domain error

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
        module_outcome_tracker: None,
        temperature: 1.0,
    };

    let result = neat_ai_discovery::analysis::analyze_synapses(&input)
        .expect("Synapse analysis should succeed");

    // Expected behaviour (post-fix):
    // - no direct helpful synapse from constant source
    // - a coordinated-structural candidate that sets output bias instead
    assert!(
        result.helpful_synapses.is_empty(),
        "expected no helpful synapses; constant sources should be folded into setBias"
    );

    assert_eq!(
        result.coordinated_structural_candidates.len(),
        1,
        "expected a single coordinated candidate"
    );

    let candidate = &result.coordinated_structural_candidates[0];
    assert_eq!(
        candidate.operations.len(),
        1,
        "expected a single setBias operation"
    );

    // Bias should be nudged upward (direction matters); exact value is determined by the
    // synapse weight calculation and clamping, but must be > 0 for positive error.
    match &candidate.operations[0] {
        neat_ai_discovery::CoordinatedStructuralOpJson::SetBias { neuron_uuid, bias } => {
            assert_eq!(neuron_uuid, "output-0");
            assert!(
                *bias > 0.0,
                "expected positive bias adjustment for positive constant error, got {bias}"
            );
        }
        other => panic!("expected setBias op, got {other:?}"),
    }
}
