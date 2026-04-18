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

// =============================================================================
// Issue #1110: Coordinated minimum expected-gain floor tests
// =============================================================================

#[test]
fn coordinated_min_expected_gain_constant_is_1e_minus_5() {
    use crate::analysis::constants::COORDINATED_MIN_EXPECTED_GAIN;
    assert!(
        (COORDINATED_MIN_EXPECTED_GAIN - 1e-5).abs() < f32::EPSILON,
        "COORDINATED_MIN_EXPECTED_GAIN should be 1e-5, got {COORDINATED_MIN_EXPECTED_GAIN}",
    );
}

/// Production failure data shows gains at ~8e-8 produce negative actual
/// outcomes. These noise-level candidates must be rejected.
#[test]
fn noise_level_gain_8e_minus_8_is_rejected() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "noise level module",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 2,
                candidates: vec![
                    make_candidate(8e-8), // noise — should be rejected
                    make_candidate(1e-4), // genuine — should be accepted
                ],
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        1,
        "noise-level candidate at 8e-8 should be filtered, genuine at 1e-4 kept"
    );
    assert!(syn.coordinated_structural_candidates[0].expected_creature_score_gain >= 1e-5);
}

/// Candidates at 1e-4 are well above the floor and should be accepted.
#[test]
fn genuine_gain_1e_minus_4_is_accepted() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "genuine gain module",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(1e-4)],
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        1,
        "candidate at 1e-4 should be accepted"
    );
}

/// Candidates exactly at the threshold (1e-5) should be accepted (>= check).
#[test]
fn gain_exactly_at_threshold_is_accepted() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "exact threshold module",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(1e-5)],
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        1,
        "candidate exactly at threshold 1e-5 should be accepted"
    );
}

/// Candidates just below the threshold should be rejected.
#[test]
fn gain_just_below_threshold_is_rejected() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "below threshold module",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(9e-6)],
            })
        },
    );

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "candidate at 9e-6 (below 1e-5 threshold) should be rejected"
    );
}

// =============================================================================
// Issue #557: Positive gain filtering tests
// =============================================================================

#[test]
fn run_discovery_module_filters_zero_gain_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "zero gain module",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 3,
                candidates: vec![
                    make_candidate(1.0),
                    make_candidate(0.0), // should be filtered
                    make_candidate(0.5),
                ],
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        2,
        "zero-gain candidates should be filtered"
    );
    for c in &syn.coordinated_structural_candidates {
        assert!(c.expected_creature_score_gain > 0.0);
    }
}

#[test]
fn run_discovery_module_filters_negative_gain_candidates() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "negative gain module",
        "test_phase",
        None,
        false,
        || {
            Some(DiscoveryDetectionResult {
                detected_count: 2,
                candidates: vec![make_candidate(-0.1), make_candidate(-0.5)],
            })
        },
    );

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "all-negative module should produce zero candidates"
    );
}
