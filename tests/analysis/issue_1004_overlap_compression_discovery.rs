//! Integration tests for Issue #1004: Overlap candidate compression with
//! discovery module detection phase.
//!
//! Verifies that:
//! 1. The split detect + merge pattern for discovery modules produces identical
//!    results to the combined `run_discovery_modules_parallel`.
//! 2. `detect_discovery_modules_parallel` correctly captures all detection results.
//! 3. `merge_discovery_module_results` correctly merges into synapse results.
//! 4. Compression and discovery detection can overlap via `rayon::join` without
//!    affecting correctness.

use neat_ai_discovery::analysis::candidate_compression::{
    compress_identity_candidates, compress_nonlinear_candidates,
};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleDetectionEntry, DiscoveryModuleDetectionResults,
    DiscoveryModuleSpec, detect_discovery_modules_parallel, merge_discovery_module_results,
    run_discovery_modules_parallel,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::shared::{self, SynapseAnalysisMetadata};
use neat_ai_discovery::{
    CandidateSynapseJson, CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson,
    CreatureJson, NeuronJson, SynapseJson,
};

fn empty_synapse_result() -> shared::AnalyzeSynapsesResult {
    shared::AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata {
            candidates_found: 0,
            candidates_returned: 0,
            ..Default::default()
        },
    }
}

fn make_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some(format!("test gain={gain}")),
    }
}

fn make_module(
    name: &str,
    candidates: Option<Vec<CoordinatedStructuralCandidateJson>>,
) -> DiscoveryModuleSpec {
    let detected_count = candidates.as_ref().map_or(0, std::vec::Vec::len);
    DiscoveryModuleSpec {
        module_name: name.to_string(),
        phase_name: "test_phase",
        max_candidates: 0,
        detect_fn: Box::new(move || {
            candidates.map(|c| DiscoveryDetectionResult {
                detected_count,
                candidates: c,
            })
        }),
    }
}

fn synapse_candidate(from: &str, to: &str, weight: f32, gain: f32) -> CandidateSynapseJson {
    CandidateSynapseJson {
        from_neuron_uuid: from.to_string(),
        to_neuron_uuid: to.to_string(),
        from_neuron_index: None,
        to_neuron_index: None,
        weight,
        target_neuron_impact: 1.0,
        expected_creature_error_reduction: gain,
        expected_creature_score_gain: gain,
        improved_count: 80,
        total_count: 100,
        improvement_magnitude_ratio: None,
        target_neuron_stats: None,
        outlier_reduction_info: None,
        prediction_confidence: 0.8,
        expected_score_gain_confidence_interval: [gain * 0.5, gain * 1.5],
        comment: None,
        variant_key: None,
    }
}

fn make_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-a".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-b".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-c".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-1".to_string(),
                neuron_type: "output".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-a".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.3,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-b".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "input-c".to_string(),
                to_uuid: "output-1".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 3,
        output: 1,
    }
}

// =============================================================================
// Issue #1004: Split detect + merge equivalence tests
// =============================================================================

/// Verify that `detect_discovery_modules_parallel` returns all detection results
/// in the original module order.
#[test]
fn detect_returns_results_in_original_order() {
    let modules = vec![
        make_module("mod_a", Some(vec![make_candidate(1.0)])),
        make_module("mod_b", None),
        make_module(
            "mod_c",
            Some(vec![make_candidate(2.0), make_candidate(3.0)]),
        ),
    ];

    let results = detect_discovery_modules_parallel(modules, None, None);

    assert_eq!(results.entries.len(), 3, "should have 3 entries");
    assert_eq!(results.entries[0].module_name, "mod_a");
    assert_eq!(results.entries[1].module_name, "mod_b");
    assert_eq!(results.entries[2].module_name, "mod_c");

    // mod_a: 1 candidate
    assert!(results.entries[0].result.is_some());
    assert_eq!(
        results.entries[0].result.as_ref().unwrap().candidates.len(),
        1
    );

    // mod_b: None
    assert!(results.entries[1].result.is_none());

    // mod_c: 2 candidates
    assert!(results.entries[2].result.is_some());
    assert_eq!(
        results.entries[2].result.as_ref().unwrap().candidates.len(),
        2
    );
}

/// Verify that split detect + merge produces the same synapse result as the
/// combined `run_discovery_modules_parallel`.
#[test]
fn split_detect_merge_matches_combined() {
    // Combined approach.
    let mut syn_combined = empty_synapse_result();
    let modules_combined = vec![
        make_module("mod_a", Some(vec![make_candidate(2.0)])),
        make_module(
            "mod_b",
            Some(vec![make_candidate(1.0), make_candidate(0.5)]),
        ),
        make_module("mod_c", None),
    ];
    let mut tracker_combined = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(
        &mut syn_combined,
        modules_combined,
        None,
        false,
        &mut tracker_combined,
    );

    // Split approach.
    let mut syn_split = empty_synapse_result();
    let modules_split = vec![
        make_module("mod_a", Some(vec![make_candidate(2.0)])),
        make_module(
            "mod_b",
            Some(vec![make_candidate(1.0), make_candidate(0.5)]),
        ),
        make_module("mod_c", None),
    ];
    let detection_results = detect_discovery_modules_parallel(modules_split, None, None);
    let mut tracker_split = ModuleOutcomeTracker::new();
    merge_discovery_module_results(
        &mut syn_split,
        detection_results,
        None,
        false,
        &mut tracker_split,
    );

    // Both should produce identical results.
    assert_eq!(
        syn_combined.coordinated_structural_candidates.len(),
        syn_split.coordinated_structural_candidates.len(),
        "candidate count must match"
    );
    for (combined, split) in syn_combined
        .coordinated_structural_candidates
        .iter()
        .zip(syn_split.coordinated_structural_candidates.iter())
    {
        assert!(
            (combined.expected_creature_score_gain - split.expected_creature_score_gain).abs()
                < 1e-6,
            "gains must match"
        );
    }
    assert_eq!(
        syn_combined.metadata.candidates_returned, syn_split.metadata.candidates_returned,
        "metadata must match"
    );
    assert_eq!(
        syn_combined.metadata.discovery_module_stats.len(),
        syn_split.metadata.discovery_module_stats.len(),
        "per-module stats count must match"
    );
}

/// Verify that `merge_discovery_module_results` handles empty detection results.
#[test]
fn merge_empty_detection_results_is_noop() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();
    let empty_results = DiscoveryModuleDetectionResults {
        entries: Vec::new(),
    };
    merge_discovery_module_results(&mut syn, empty_results, None, false, &mut tracker);

    assert!(syn.coordinated_structural_candidates.is_empty());
    assert_eq!(syn.metadata.candidates_returned, 0);
}

/// Verify that detect returns empty results for empty module list.
#[test]
fn detect_empty_modules_returns_empty() {
    let results = detect_discovery_modules_parallel(Vec::new(), None, None);
    assert!(results.entries.is_empty());
}

// =============================================================================
// Issue #1004: Overlapped compression + discovery detection tests
// =============================================================================

/// Verify that compression and discovery detection can overlap via `rayon::join`
/// without affecting the correctness of either phase.
#[test]
fn overlapped_compression_and_detection_produces_correct_results() {
    let creature = make_creature();
    let helpful_synapses = vec![
        synapse_candidate("input-a", "output-1", 0.3, 0.05),
        synapse_candidate("input-b", "output-1", 0.5, 0.06),
        synapse_candidate("input-c", "output-1", 0.4, 0.04),
    ];

    // Sequential: compression first, then detection.
    let seq_identity = compress_identity_candidates(&helpful_synapses, &creature);
    let seq_nonlinear = compress_nonlinear_candidates(&helpful_synapses, &creature);
    let mut seq_compressed = seq_identity;
    seq_compressed.extend(seq_nonlinear);

    let modules_seq = vec![
        make_module("mod_a", Some(vec![make_candidate(1.0)])),
        make_module("mod_b", Some(vec![make_candidate(0.5)])),
    ];
    let mut syn_seq = empty_synapse_result();
    let mut tracker_seq = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn_seq, modules_seq, None, false, &mut tracker_seq);

    // Overlapped: compression and detection run concurrently.
    let (par_compressed, par_detection) = rayon::join(
        || {
            let (identity, nonlinear) = rayon::join(
                || compress_identity_candidates(&helpful_synapses, &creature),
                || compress_nonlinear_candidates(&helpful_synapses, &creature),
            );
            let mut all = identity;
            all.extend(nonlinear);
            all
        },
        || {
            let modules = vec![
                make_module("mod_a", Some(vec![make_candidate(1.0)])),
                make_module("mod_b", Some(vec![make_candidate(0.5)])),
            ];
            detect_discovery_modules_parallel(modules, None, None)
        },
    );

    let mut syn_par = empty_synapse_result();
    let mut tracker_par = ModuleOutcomeTracker::new();
    merge_discovery_module_results(&mut syn_par, par_detection, None, false, &mut tracker_par);

    // Compression results should be identical.
    assert_eq!(
        seq_compressed.len(),
        par_compressed.len(),
        "compression candidate count must match"
    );
    for (seq, par) in seq_compressed.iter().zip(par_compressed.iter()) {
        assert!(
            (seq.expected_creature_score_gain - par.expected_creature_score_gain).abs() < 1e-6,
            "compression gains must match"
        );
    }

    // Discovery results should be identical.
    assert_eq!(
        syn_seq.coordinated_structural_candidates.len(),
        syn_par.coordinated_structural_candidates.len(),
        "discovery candidate count must match"
    );
}

/// Verify overlapped execution multiple times to expose potential race conditions.
#[test]
fn overlapped_execution_is_deterministic_across_runs() {
    let creature = make_creature();
    let helpful_synapses = vec![
        synapse_candidate("input-a", "output-1", 0.3, 0.05),
        synapse_candidate("input-b", "output-1", 0.5, 0.06),
    ];

    let mut all_compression_counts: Vec<usize> = Vec::new();
    let mut all_detection_counts: Vec<usize> = Vec::new();

    for _ in 0..10 {
        let snapshot = helpful_synapses.clone();
        let creature_clone = creature.clone();

        let (compressed, detection) = rayon::join(
            || {
                let (identity, nonlinear) = rayon::join(
                    || compress_identity_candidates(&snapshot, &creature_clone),
                    || compress_nonlinear_candidates(&snapshot, &creature_clone),
                );
                let mut all = identity;
                all.extend(nonlinear);
                all
            },
            || {
                let modules = vec![
                    make_module("mod_1", Some(vec![make_candidate(1.0)])),
                    make_module("mod_2", Some(vec![make_candidate(2.0)])),
                    make_module("mod_3", Some(vec![make_candidate(3.0)])),
                ];
                detect_discovery_modules_parallel(modules, None, None)
            },
        );

        all_compression_counts.push(compressed.len());
        all_detection_counts.push(detection.entries.len());
    }

    // All runs should produce identical counts.
    let first_compression = all_compression_counts[0];
    let first_detection = all_detection_counts[0];
    for (i, (&c, &d)) in all_compression_counts
        .iter()
        .zip(all_detection_counts.iter())
        .enumerate()
    {
        assert_eq!(c, first_compression, "compression count differs on run {i}");
        assert_eq!(d, first_detection, "detection count differs on run {i}");
    }
}

/// Verify that merge ordering is deterministic: compression results merged first,
/// then discovery results, matching the sequential pipeline order.
#[test]
fn merge_ordering_compression_before_discovery() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();

    // First merge compression results (gain = 5.0) — simulated by adding directly.
    syn.coordinated_structural_candidates
        .push(CoordinatedStructuralCandidateJson {
            remove_neuron_compensation: None,
            constant_neuron_bias_fold: None,
            operations: vec![CoordinatedStructuralOpJson::AddNeuron {
                neuron_uuid: "compress-test".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
                insert_before_neuron_uuid: None,
            }],
            expected_creature_score_gain: 5.0,
            comment: Some("compression candidate".to_string()),
        });

    // Then merge discovery results (gain = 3.0).
    let discovery_results = DiscoveryModuleDetectionResults {
        entries: vec![DiscoveryModuleDetectionEntry {
            module_name: "test_module".to_string(),
            phase_name: "test_phase",
            max_candidates: 0,
            result: Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(3.0)],
            }),
        }],
    };
    merge_discovery_module_results(&mut syn, discovery_results, None, false, &mut tracker);

    // Both candidates should be present.
    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "should have both compression and discovery candidates"
    );

    // In non-diversify mode, candidates are sorted by gain descending.
    // The compression candidate (5.0) should come first.
    assert!(
        syn.coordinated_structural_candidates[0].expected_creature_score_gain >= 5.0 - 1e-6,
        "compression candidate (gain=5.0) should be first"
    );
    assert!(
        syn.coordinated_structural_candidates[1].expected_creature_score_gain >= 3.0 - 1e-6,
        "discovery candidate (gain=3.0) should be second"
    );
}

/// Verify that the split detect + merge respects `max_synapse_candidates`.
#[test]
fn split_detect_merge_respects_max_candidates() {
    let mut tracker = ModuleOutcomeTracker::new();
    let mut syn = empty_synapse_result();

    let modules = vec![
        make_module(
            "mod_a",
            Some(vec![make_candidate(3.0), make_candidate(2.0)]),
        ),
        make_module("mod_b", Some(vec![make_candidate(1.0)])),
    ];

    let detection_results = detect_discovery_modules_parallel(modules, None, None);
    merge_discovery_module_results(&mut syn, detection_results, Some(2), false, &mut tracker);

    let total = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();
    assert!(
        total <= 2,
        "expected truncation to max_synapse_candidates=2, got {total}"
    );
}
