use crate::analysis::shared;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

use super::{DiscoveryDetectionResult, run_discovery_module};

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
        comment: Some("test".to_string()),
    }
}

#[test]
fn run_discovery_module_merges_candidates_into_synapse_result() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(&mut syn, "test module", "test_phase", None, false, || {
        Some(DiscoveryDetectionResult {
            detected_count: 2,
            candidates: vec![make_candidate(1.0), make_candidate(0.5)],
        })
    });

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "expected both candidates to be merged"
    );
    assert_eq!(syn.metadata.candidates_returned, 2);
}

#[test]
fn run_discovery_module_does_nothing_when_detect_returns_none() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(&mut syn, "empty module", "test_phase", None, false, || None);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "expected no candidates when detect_fn returns None"
    );
    assert_eq!(syn.metadata.candidates_returned, 0);
}

#[test]
fn run_discovery_module_does_nothing_when_candidates_empty() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(&mut syn, "no candidates", "test_phase", None, false, || {
        Some(DiscoveryDetectionResult {
            detected_count: 0,
            candidates: Vec::new(),
        })
    });

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "expected no candidates when result has empty candidates vec"
    );
}

#[test]
fn run_discovery_module_respects_max_synapse_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "capped module",
        "test_phase",
        Some(1),
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 3,
                candidates: vec![
                    make_candidate(3.0),
                    make_candidate(2.0),
                    make_candidate(1.0),
                ],
            })
        },
    );

    let total = syn.helpful_synapses.len()
        + syn.harmful_synapses.len()
        + syn.coordinated_structural_candidates.len();
    assert_eq!(total, 1, "expected truncation to max_synapse_candidates=1");
}

#[test]
fn run_discovery_module_beats_watchdog_start_and_finish() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(&mut syn, "watchdog test", "test_phase", None, false, || {
        None
    });

    // After the module runs, the last watchdog beat should be the "finished" one
    assert_eq!(
        crate::watchdog::active_stage_for_test().as_deref(),
        Some("analysis::analyze_all → watchdog test finished")
    );
}

#[test]
fn run_discovery_module_accumulates_across_multiple_calls() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    // First module adds 1 candidate
    run_discovery_module(&mut syn, "module A", "phase_a", None, false, || {
        Some(DiscoveryDetectionResult {
            detected_count: 1,
            candidates: vec![make_candidate(2.0)],
        })
    });

    // Second module adds 2 more candidates
    run_discovery_module(&mut syn, "module B", "phase_b", None, false, || {
        Some(DiscoveryDetectionResult {
            detected_count: 2,
            candidates: vec![make_candidate(1.0), make_candidate(0.5)],
        })
    });

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        3,
        "expected candidates from both modules to accumulate"
    );
    assert_eq!(syn.metadata.candidates_returned, 3);
}
