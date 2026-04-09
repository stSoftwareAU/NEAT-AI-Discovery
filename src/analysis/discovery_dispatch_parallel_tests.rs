use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, SystemTime};

use crate::analysis::shared;
use crate::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

use super::{
    DiscoveryDetectionResult, DiscoveryModuleSpec, ModuleOutcomeTracker,
    detect_discovery_modules_parallel, run_discovery_modules_parallel,
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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, modules, Some(2), false, &mut tracker);

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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, Vec::new(), None, false, &mut tracker);

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

        let mut tracker = ModuleOutcomeTracker::new();
        run_discovery_modules_parallel(&mut syn, modules, None, true, &mut tracker);

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
    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn_parallel, modules, None, false, &mut tracker);

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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

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

    let mut tracker = ModuleOutcomeTracker::new();
    run_discovery_modules_parallel(&mut syn, modules, None, false, &mut tracker);

    assert!(
        syn.coordinated_structural_candidates.is_empty(),
        "all-non-positive module should produce zero candidates"
    );
}

// =============================================================================
// Issue #1029: Deadline-aware discovery module detection tests
// =============================================================================

#[test]
fn parallel_detection_skips_modules_when_deadline_already_passed() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: Duration::from_secs(60),
        abort_delay: Duration::from_secs(1),
    });

    // Track how many detection closures actually execute.
    let execution_count = Arc::new(AtomicUsize::new(0));

    let modules: Vec<DiscoveryModuleSpec> = (0..5)
        .map(|i| {
            let counter = Arc::clone(&execution_count);
            DiscoveryModuleSpec {
                module_name: format!("module_{i}"),
                phase_name: "test_phase",
                max_candidates: 0,
                detect_fn: Box::new(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Some(DiscoveryDetectionResult {
                        detected_count: 1,
                        candidates: vec![make_candidate(1.0)],
                    })
                }),
            }
        })
        .collect();

    // Deadline is already in the past — all modules should be skipped.
    let past_deadline = Some(SystemTime::now() - Duration::from_secs(10));
    let results = detect_discovery_modules_parallel(modules, past_deadline, None);

    // All entries should have result = None (skipped).
    assert_eq!(results.entries.len(), 5);
    for entry in &results.entries {
        assert!(
            entry.result.is_none(),
            "Module '{}' should have been skipped due to past deadline",
            entry.module_name
        );
    }

    // No detection closures should have executed.
    assert_eq!(
        execution_count.load(Ordering::Relaxed),
        0,
        "No detection closures should execute when deadline has already passed"
    );
}

#[test]
fn parallel_detection_runs_all_modules_when_no_deadline() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: Duration::from_secs(60),
        abort_delay: Duration::from_secs(1),
    });

    let execution_count = Arc::new(AtomicUsize::new(0));

    let modules: Vec<DiscoveryModuleSpec> = (0..3)
        .map(|i| {
            let counter = Arc::clone(&execution_count);
            DiscoveryModuleSpec {
                module_name: format!("module_{i}"),
                phase_name: "test_phase",
                max_candidates: 0,
                detect_fn: Box::new(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Some(DiscoveryDetectionResult {
                        detected_count: 1,
                        candidates: vec![make_candidate(1.0)],
                    })
                }),
            }
        })
        .collect();

    // No deadline — all modules should run.
    let results = detect_discovery_modules_parallel(modules, None, None);

    assert_eq!(results.entries.len(), 3);
    assert_eq!(
        execution_count.load(Ordering::Relaxed),
        3,
        "All detection closures should execute when no deadline is set"
    );
    for entry in &results.entries {
        assert!(
            entry.result.is_some(),
            "Module '{}' should have produced results without a deadline",
            entry.module_name
        );
    }
}

#[test]
fn parallel_detection_runs_all_modules_when_deadline_is_far_future() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: Duration::from_secs(60),
        abort_delay: Duration::from_secs(1),
    });

    let execution_count = Arc::new(AtomicUsize::new(0));

    let modules: Vec<DiscoveryModuleSpec> = (0..3)
        .map(|i| {
            let counter = Arc::clone(&execution_count);
            DiscoveryModuleSpec {
                module_name: format!("module_{i}"),
                phase_name: "test_phase",
                max_candidates: 0,
                detect_fn: Box::new(move || {
                    counter.fetch_add(1, Ordering::Relaxed);
                    Some(DiscoveryDetectionResult {
                        detected_count: 1,
                        candidates: vec![make_candidate(1.0)],
                    })
                }),
            }
        })
        .collect();

    // Deadline far in the future — all modules should run.
    let future_deadline = Some(SystemTime::now() + Duration::from_secs(3600));
    let results = detect_discovery_modules_parallel(modules, future_deadline, None);

    assert_eq!(results.entries.len(), 3);
    assert_eq!(
        execution_count.load(Ordering::Relaxed),
        3,
        "All detection closures should execute when deadline is far in the future"
    );
}

#[test]
fn parallel_detection_preserves_module_metadata_when_skipped() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: Duration::from_secs(60),
        abort_delay: Duration::from_secs(1),
    });

    let modules = vec![
        DiscoveryModuleSpec {
            module_name: "alpha".to_string(),
            phase_name: "phase_alpha",
            max_candidates: 10,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(1.0)],
                })
            }),
        },
        DiscoveryModuleSpec {
            module_name: "beta".to_string(),
            phase_name: "phase_beta",
            max_candidates: 20,
            detect_fn: Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![make_candidate(2.0)],
                })
            }),
        },
    ];

    let past_deadline = Some(SystemTime::now() - Duration::from_secs(10));
    let results = detect_discovery_modules_parallel(modules, past_deadline, None);

    // Module metadata (name, phase, budget) should be preserved even when skipped.
    assert_eq!(results.entries[0].module_name, "alpha");
    assert_eq!(results.entries[0].phase_name, "phase_alpha");
    assert_eq!(results.entries[0].max_candidates, 10);
    assert!(results.entries[0].result.is_none());

    assert_eq!(results.entries[1].module_name, "beta");
    assert_eq!(results.entries[1].phase_name, "phase_beta");
    assert_eq!(results.entries[1].max_candidates, 20);
    assert!(results.entries[1].result.is_none());
}
