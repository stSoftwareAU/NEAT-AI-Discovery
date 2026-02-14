use crate::analysis::shared;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

use super::{DiscoveryDetectionResult, DiscoveryModuleSpec, run_discovery_modules_parallel};

fn empty_synapse_result() -> shared::AnalyzeSynapsesResult {
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

fn make_candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
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
    let detected_count = candidates.as_ref().map(|c| c.len()).unwrap_or(0);
    DiscoveryModuleSpec {
        module_name: name.to_string(),
        phase_name: "test_phase",
        detect_fn: Box::new(move || {
            candidates.map(|c| DiscoveryDetectionResult {
                detected_count,
                candidates: c,
            })
        }),
    }
}

#[test]
fn parallel_dispatch_merges_candidates_from_multiple_modules() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    let modules = vec![
        make_module("module_a", Some(vec![make_candidate(2.0)])),
        make_module(
            "module_b",
            Some(vec![make_candidate(1.0), make_candidate(0.5)]),
        ),
    ];

    run_discovery_modules_parallel(&mut syn, modules, None, false);

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        3,
        "expected all candidates from both modules to be merged"
    );
    assert_eq!(syn.metadata.candidates_returned, 3);
}

#[test]
fn parallel_dispatch_handles_mix_of_none_and_some() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    let modules = vec![
        make_module("returns_none", None),
        make_module("returns_some", Some(vec![make_candidate(1.0)])),
        make_module("returns_empty", Some(Vec::new())),
        make_module("returns_more", Some(vec![make_candidate(0.5)])),
    ];

    run_discovery_modules_parallel(&mut syn, modules, None, false);

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "expected only non-empty Some results to be merged"
    );
}

#[test]
fn parallel_dispatch_respects_max_synapse_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    let modules = vec![
        make_module(
            "mod_a",
            Some(vec![make_candidate(3.0), make_candidate(2.0)]),
        ),
        make_module("mod_b", Some(vec![make_candidate(1.0)])),
    ];

    run_discovery_modules_parallel(&mut syn, modules, Some(2), false);

    let total = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();
    assert!(
        total <= 2,
        "expected truncation to max_synapse_candidates=2, got {total}"
    );
}

#[test]
fn parallel_dispatch_with_empty_modules_is_noop() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_modules_parallel(&mut syn, Vec::new(), None, false);

    assert!(syn.coordinated_structural_candidates.is_empty());
    assert_eq!(syn.metadata.candidates_returned, 0);
}

#[test]
fn parallel_dispatch_preserves_deterministic_ordering() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    // Run 10 times and check all produce the same result.
    // Diversify mode is enabled to skip sorting and expose raw merge ordering.
    let mut all_gains: Vec<Vec<f32>> = Vec::new();

    for _ in 0..10 {
        let mut syn = empty_synapse_result();

        let modules = vec![
            make_module("mod_1", Some(vec![make_candidate(1.0)])),
            make_module("mod_2", Some(vec![make_candidate(2.0)])),
            make_module("mod_3", Some(vec![make_candidate(3.0)])),
            make_module("mod_4", Some(vec![make_candidate(4.0)])),
            make_module("mod_5", Some(vec![make_candidate(5.0)])),
        ];

        run_discovery_modules_parallel(&mut syn, modules, None, true);

        let gains: Vec<f32> = syn
            .coordinated_structural_candidates
            .iter()
            .map(|c| c.expected_creature_score_gain)
            .collect();
        all_gains.push(gains);
    }

    // All runs should produce identical ordering
    for (i, gains) in all_gains.iter().enumerate().skip(1) {
        assert_eq!(
            &all_gains[0], gains,
            "run {i} produced different ordering than run 0"
        );
    }
}

#[test]
fn parallel_dispatch_single_module_matches_sequential_behaviour() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    // Parallel with one module
    let mut syn_parallel = empty_synapse_result();
    let modules = vec![make_module(
        "single",
        Some(vec![make_candidate(1.5), make_candidate(0.7)]),
    )];
    run_discovery_modules_parallel(&mut syn_parallel, modules, None, false);

    // Sequential with one module (using existing run_discovery_module)
    let mut syn_sequential = empty_synapse_result();
    super::run_discovery_module(
        &mut syn_sequential,
        "single",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 2,
                candidates: vec![make_candidate(1.5), make_candidate(0.7)],
            })
        },
    );

    assert_eq!(
        syn_parallel.coordinated_structural_candidates.len(),
        syn_sequential.coordinated_structural_candidates.len(),
        "parallel and sequential should produce same number of candidates"
    );

    for (p, s) in syn_parallel
        .coordinated_structural_candidates
        .iter()
        .zip(syn_sequential.coordinated_structural_candidates.iter())
    {
        assert_eq!(
            p.expected_creature_score_gain, s.expected_creature_score_gain,
            "parallel and sequential candidates should have same gains"
        );
    }

    assert_eq!(
        syn_parallel.metadata.candidates_returned,
        syn_sequential.metadata.candidates_returned
    );
}

// =============================================================================
// Issue #557: Positive gain filtering tests
// =============================================================================

#[test]
fn parallel_dispatch_filters_zero_gain_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    let modules = vec![make_module(
        "zero_gain",
        Some(vec![
            make_candidate(0.5),
            make_candidate(0.0), // should be filtered
            make_candidate(0.3),
        ]),
    )];

    run_discovery_modules_parallel(&mut syn, modules, None, false);

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "candidates with zero gain should be filtered out"
    );
    for c in &syn.coordinated_structural_candidates {
        assert!(
            c.expected_creature_score_gain > 0.0,
            "all candidates must have positive gain, got {}",
            c.expected_creature_score_gain
        );
    }
}

#[test]
fn parallel_dispatch_filters_negative_gain_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    let modules = vec![make_module(
        "negative_gain",
        Some(vec![
            make_candidate(1.0),
            make_candidate(-0.5),   // should be filtered
            make_candidate(-0.001), // should be filtered
        ]),
    )];

    run_discovery_modules_parallel(&mut syn, modules, None, false);

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        1,
        "candidates with negative gain should be filtered out"
    );
    assert!(syn.coordinated_structural_candidates[0].expected_creature_score_gain > 0.0);
}

#[test]
fn parallel_dispatch_filters_all_non_positive_returns_empty() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    let modules = vec![make_module(
        "all_non_positive",
        Some(vec![make_candidate(0.0), make_candidate(-1.0)]),
    )];

    run_discovery_modules_parallel(&mut syn, modules, None, false);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "all-non-positive module should produce zero candidates"
    );
}
