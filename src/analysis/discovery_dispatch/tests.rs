use super::*;
use crate::analysis::shared;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

/// A mock discovery module that returns a fixed set of candidates.
struct MockModule {
    detection_count: usize,
    candidates: Vec<CoordinatedStructuralCandidateJson>,
}

impl DiscoveryModule for MockModule {
    fn name(&self) -> &'static str {
        "mock"
    }

    fn phase_name(&self) -> &'static str {
        "mock_detection"
    }

    fn detect_and_convert(
        &self,
        _creature: &CreatureJson,
        _shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        (self.detection_count, self.candidates.clone())
    }
}

/// A mock module that returns no candidates.
struct EmptyModule;

impl DiscoveryModule for EmptyModule {
    fn name(&self) -> &'static str {
        "empty"
    }

    fn phase_name(&self) -> &'static str {
        "empty_detection"
    }

    fn detect_and_convert(
        &self,
        _creature: &CreatureJson,
        _shared_cache: &Arc<RecordCache>,
    ) -> (usize, Vec<CoordinatedStructuralCandidateJson>) {
        (0, Vec::new())
    }
}

fn make_test_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            crate::NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            crate::NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "LOGISTIC".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![],
        input: 1,
        output: 1,
    }
}

fn make_test_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        operations: vec![CoordinatedStructuralOpJson::SetBias {
            neuron_uuid: "output-1".to_string(),
            bias: 0.1,
        }],
        expected_creature_score_gain: gain,
        comment: Some("test candidate".to_string()),
    }
}

fn make_empty_synapse_result() -> shared::AnalyzeSynapsesResult {
    shared::AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: shared::SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    }
}

fn make_test_cache() -> Arc<RecordCache> {
    Arc::new(RecordCache::new_empty_for_test())
}

#[test]
fn dispatch_module_merges_candidates_into_synapse_result() {
    let _lock = crate::watchdog::lock_for_test_serialisation();

    let module = MockModule {
        detection_count: 2,
        candidates: vec![make_test_candidate(0.5), make_test_candidate(0.3)],
    };

    let creature = make_test_creature();
    let cache = make_test_cache();
    let mut syn = make_empty_synapse_result();

    dispatch_discovery_module(&module, &creature, &cache, &mut syn, None, false);

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "dispatch should merge candidates into synapse result"
    );
}

#[test]
fn dispatch_empty_module_leaves_synapse_result_unchanged() {
    let _lock = crate::watchdog::lock_for_test_serialisation();

    let module = EmptyModule;
    let creature = make_test_creature();
    let cache = make_test_cache();
    let mut syn = make_empty_synapse_result();

    dispatch_discovery_module(&module, &creature, &cache, &mut syn, None, false);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "dispatch of empty module should not add candidates"
    );
}

#[test]
fn dispatch_all_modules_runs_every_module() {
    let _lock = crate::watchdog::lock_for_test_serialisation();

    let module_a = MockModule {
        detection_count: 1,
        candidates: vec![make_test_candidate(0.5)],
    };
    let module_b = MockModule {
        detection_count: 1,
        candidates: vec![make_test_candidate(0.3)],
    };
    let empty = EmptyModule;

    let modules: Vec<&dyn DiscoveryModule> = vec![&module_a, &module_b, &empty];

    let creature = make_test_creature();
    let cache = make_test_cache();
    let mut syn = make_empty_synapse_result();

    dispatch_all_discovery_modules(&modules, &creature, &cache, &mut syn, None, false);

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "dispatch_all should accumulate candidates from all modules"
    );
}

#[test]
fn dispatch_respects_max_synapse_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();

    let module = MockModule {
        detection_count: 3,
        candidates: vec![
            make_test_candidate(0.5),
            make_test_candidate(0.3),
            make_test_candidate(0.1),
        ],
    };

    let creature = make_test_creature();
    let cache = make_test_cache();
    let mut syn = make_empty_synapse_result();

    dispatch_discovery_module(&module, &creature, &cache, &mut syn, Some(1), false);

    let total = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();
    assert!(
        total <= 1,
        "dispatch should respect max_synapse_candidates limit"
    );
}

#[test]
fn capitalise_first_works_for_various_inputs() {
    assert_eq!(capitalise_first("saturation"), "Saturation");
    assert_eq!(capitalise_first("multi-hop"), "Multi-hop");
    assert_eq!(capitalise_first(""), "");
    assert_eq!(capitalise_first("A"), "A");
}

#[test]
fn collect_records_for_uuids_returns_entries_with_empty_records() {
    let cache = make_test_cache();
    let uuids = vec!["nonexistent-1".to_string(), "nonexistent-2".to_string()];
    let records = collect_records_for_uuids(&uuids, &cache);

    // The empty test cache returns Ok(empty vec) for any UUID,
    // so we get entries back but with empty record vectors.
    assert_eq!(records.len(), 2, "should return an entry per UUID");
    for (_, recs) in &records {
        assert!(recs.is_empty(), "records should be empty for missing UUIDs");
    }
}

#[test]
fn collect_records_for_hidden_neurons_returns_entries_with_empty_records() {
    let cache = make_test_cache();
    let neurons = vec![("h1".to_string(), "LOGISTIC".to_string(), 0.0f32)];
    let records = collect_records_for_hidden_neurons(&neurons, &cache);

    assert_eq!(records.len(), 1, "should return an entry per neuron");
    assert!(
        records[0].1.is_empty(),
        "records should be empty for missing neurons"
    );
}
