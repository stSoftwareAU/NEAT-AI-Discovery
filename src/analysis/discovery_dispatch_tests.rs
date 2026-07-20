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
        remove_neuron_compensation: None,
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

// =============================================================================
// Issue #1271: Per-final-target coordinated-structural cap tests
// =============================================================================

/// Build a coordinated-structural candidate whose final operation targets
/// `target_uuid` (an `addSynapse` op into `target_uuid`), with the supplied
/// expected gain.
fn make_candidate_with_target(target_uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::AddSynapse {
            from_neuron_uuid: format!("source-{gain}"),
            to_neuron_uuid: target_uuid.to_string(),
            weight: 0.5,
        }],
        expected_creature_score_gain: gain,
        comment: Some("test".to_string()),
    }
}

/// Acceptance: synthetic batch of 10 candidates targeting the same final
/// neuron is reduced to 3 by the per-final-target cap.
#[test]
fn coordinated_per_target_cap_reduces_ten_same_target_to_three() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "ten same target",
        "test_phase",
        None,
        false,
        || {
            let target = "shared-output-neuron";
            let mut candidates = Vec::with_capacity(10);
            for i in 0..10 {
                // Each candidate has a distinct gain so we can assert the
                // top-K is retained.
                #[allow(clippy::cast_precision_loss)]
                let gain = 0.1 + (i as f32) * 0.01;
                candidates.push(make_candidate_with_target(target, gain));
            }
            Some(DiscoveryDetectionResult {
                detected_count: 10,
                candidates,
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        3,
        "10 candidates against the same final target should be capped at 3"
    );

    // The cap retains the highest-gain candidates.
    let mut gains: Vec<f32> = syn
        .coordinated_structural_candidates
        .iter()
        .map(|c| c.expected_creature_score_gain)
        .collect();
    gains.sort_by(|a, b| b.total_cmp(a));
    assert!(
        gains[0] >= gains[1] && gains[1] >= gains[2],
        "retained candidates should be the top-3 by gain"
    );
    assert!(
        (gains[0] - 0.19).abs() < 1e-5,
        "highest retained gain should be 0.19, got {}",
        gains[0]
    );

    // Rejection breakdown surfaces the dropped count under the new reason.
    let counts = syn.metadata.rejection_breakdown.counts();
    assert_eq!(
        counts
            .get(crate::analysis::diagnostics::rejection_reasons::REJECTION_COORDINATED_TARGET_CAP_EXCEEDED),
        Some(&7),
        "7 of the 10 same-target candidates should be recorded as cap drops"
    );
}

/// Regression: the 41-consecutive-failure pattern from creature `bcbca347`
/// (GRQ-sampler commit `e85c5d2`) would have been capped at 3 instead of
/// monopolising the whole batch budget against a single output neuron.
#[test]
fn coordinated_per_target_cap_regression_bcbca347_41_same_target() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "bcbca347 regression",
        "test_phase",
        None,
        false,
        || {
            // All 41 failures had final addSynapse(... -> shared output target).
            let target = "533d8616-037c-4278-b95c-3a2a1ce15ee6";
            let mut candidates = Vec::with_capacity(41);
            for i in 0..41 {
                #[allow(clippy::cast_precision_loss)]
                let gain = 0.05 + (i as f32) * 0.001;
                candidates.push(make_candidate_with_target(target, gain));
            }
            Some(DiscoveryDetectionResult {
                detected_count: 41,
                candidates,
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        3,
        "the 41-failure bcbca347 pattern must be capped at 3 by Issue #1271"
    );
    let counts = syn.metadata.rejection_breakdown.counts();
    assert_eq!(
        counts
            .get(crate::analysis::diagnostics::rejection_reasons::REJECTION_COORDINATED_TARGET_CAP_EXCEEDED),
        Some(&38),
        "38 of the 41 same-target candidates should be recorded as cap drops"
    );
}

/// Multiple targets within the same batch are each capped independently.
#[test]
fn coordinated_per_target_cap_admits_full_quota_per_distinct_target() {
    let _lock = crate::watchdog::lock_for_test_serialisation();
    let _wd = crate::watchdog::Watchdog::start(crate::watchdog::WatchdogConfig {
        stall_timeout: std::time::Duration::from_secs(60),
        abort_delay: std::time::Duration::from_secs(1),
    });

    let mut syn = empty_synapse_result();

    run_discovery_module(
        &mut syn,
        "two targets cap",
        "test_phase",
        None,
        false,
        || {
            // Two targets, four candidates each — both should be capped to 3.
            let mut candidates = Vec::new();
            for i in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let gain = 0.10 + (i as f32) * 0.01;
                candidates.push(make_candidate_with_target("target-A", gain));
            }
            for i in 0..4 {
                #[allow(clippy::cast_precision_loss)]
                let gain = 0.20 + (i as f32) * 0.01;
                candidates.push(make_candidate_with_target("target-B", gain));
            }
            Some(DiscoveryDetectionResult {
                detected_count: candidates.len(),
                candidates,
            })
        },
    );

    assert_eq!(
        syn.coordinated_structural_candidates.len(),
        6,
        "two targets with four candidates each should keep 3+3 = 6"
    );
    let counts = syn.metadata.rejection_breakdown.counts();
    assert_eq!(
        counts
            .get(crate::analysis::diagnostics::rejection_reasons::REJECTION_COORDINATED_TARGET_CAP_EXCEEDED),
        Some(&2),
        "exactly 2 candidates (one per target) should be reported as cap drops"
    );
}

/// Constant value sanity check.
#[test]
fn max_coordinated_per_target_output_default_is_three() {
    use crate::analysis::constants::MAX_COORDINATED_PER_TARGET_OUTPUT;
    assert_eq!(
        MAX_COORDINATED_PER_TARGET_OUTPUT, 3,
        "default cap should be 3 to match Issue #1271 specification"
    );
}

/// Env-var override clamps the effective cap.
#[test]
#[serial_test::serial]
fn coordinated_per_target_cap_env_override() {
    // Hold the same serialisation lock the cap-reading tests use so this
    // env-var mutation can never overlap a concurrent reader under
    // `--test-threads>1` (those tests read the cap, this one mutates it).
    let _lock = crate::watchdog::lock_for_test_serialisation();

    // SAFETY: env access is serialised via `#[serial]`.
    unsafe {
        std::env::set_var("NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET", "1");
    }
    let effective = crate::analysis::constants::max_coordinated_per_target_output();
    assert_eq!(effective, 1, "env override of 1 should clamp the cap to 1");

    // SAFETY: env access is serialised via `#[serial]`.
    unsafe {
        std::env::remove_var("NEAT_AI_DISCOVERY_MAX_COORDINATED_PER_TARGET");
    }
    assert_eq!(
        crate::analysis::constants::max_coordinated_per_target_output(),
        crate::analysis::constants::MAX_COORDINATED_PER_TARGET_OUTPUT,
    );
}

// =============================================================================
// Issue #1530: honest propagation-aware remove-neuron gain override
// =============================================================================

use super::apply_honest_remove_neuron_gain;
use crate::CreatureJson;

/// A creature whose hidden neuron `deep` sits behind a heavily-shared output
/// inbound budget (weight 1 of 20), so its propagation-aware influence — and
/// hence its honest remove-gain magnitude — is a small `0.05`, well below the
/// retired `[0.1, 0.5]` placeholder floor. `deep` carries real, if small,
/// downstream influence, so its honest gain is strictly negative.
fn creature_with_deep_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "constant"},
                {"uuid": "deep", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "deep", "weight": 1.0},
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "deep", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 19.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

fn remove_neuron_candidate(uuid: &str, gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveNeuron {
            neuron_uuid: uuid.to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some("fabricated placeholder gain".to_string()),
    }
}

/// The core wiring: a single-op `RemoveNeuron` candidate carrying a fabricated
/// placeholder gain is overwritten with the honest, propagation-aware estimate
/// (which is non-positive), not left echoing the supplied value.
#[test]
fn honest_gain_overrides_fabricated_remove_neuron_gain() {
    let creature = creature_with_deep_neuron();
    // Deliberately wrong, fabricated placeholder-range gain.
    let mut candidates = vec![remove_neuron_candidate("deep", 0.17879)];

    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(
        overridden, 1,
        "the single RemoveNeuron candidate is overridden"
    );

    let honest = crate::analysis::estimate_remove_neuron_gain(&creature, "deep")
        .expect("estimator gain for the hidden neuron");
    let reported = f64::from(candidates[0].expected_creature_score_gain);
    assert!(
        (reported - honest).abs() < 1e-6,
        "reported gain {reported} must equal the honest estimate {honest}, not the placeholder"
    );
    assert!(
        reported <= 0.0,
        "honest remove-neuron gain must be non-positive, got {reported}"
    );
    assert!(
        !(0.1..=0.5).contains(&reported.abs()),
        "gain {reported} must not land in the retired fabricated floor range [0.1, 0.5]"
    );
}

/// Multi-operation coordinated candidates reflect the whole atomic group's
/// effect, so their gain is left untouched by the bare-removal estimator.
#[test]
fn multi_op_candidate_gain_is_not_overridden() {
    let creature = creature_with_deep_neuron();
    let mut candidates = vec![CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "deep".to_string(),
            },
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "deep".to_string(),
            },
        ],
        expected_creature_score_gain: 0.05,
        comment: None,
    }];

    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(overridden, 0, "multi-op candidates are not touched");
    assert!(
        (candidates[0].expected_creature_score_gain - 0.05).abs() < f32::EPSILON,
        "multi-op gain must be preserved"
    );
}

/// Non-RemoveNeuron single-op candidates are left untouched.
#[test]
fn non_remove_neuron_candidate_is_not_overridden() {
    let creature = creature_with_deep_neuron();
    let mut candidates = vec![make_candidate(0.42)]; // a RemoveSynapse candidate

    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(overridden, 0, "non-RemoveNeuron candidates are not touched");
    assert!((candidates[0].expected_creature_score_gain - 0.42).abs() < f32::EPSILON);
}

/// A `RemoveNeuron` candidate targeting an output or an absent neuron yields no
/// estimate (`None`), so its gain is left untouched rather than being zeroed.
#[test]
fn output_and_absent_neuron_candidates_are_not_overridden() {
    let creature = creature_with_deep_neuron();
    let mut candidates = vec![
        remove_neuron_candidate("out-0", 0.9),
        remove_neuron_candidate("does-not-exist", 0.8),
    ];

    let overridden = apply_honest_remove_neuron_gain(&creature, &mut candidates);
    assert_eq!(overridden, 0, "output/absent neurons yield no estimate");
    assert!((candidates[0].expected_creature_score_gain - 0.9).abs() < f32::EPSILON);
    assert!((candidates[1].expected_creature_score_gain - 0.8).abs() < f32::EPSILON);
}

// =============================================================================
// Issue #1689: wire #1559 covariance stats + weight-redistribution into dispatch
// =============================================================================

use super::apply_remove_neuron_compensation;
use crate::types::DiscoverRecord;

fn record(obs: u32, uuid: &str, activation: f32) -> DiscoverRecord {
    DiscoverRecord::new(obs, uuid.to_string(), None, activation, vec![])
}

/// A creature where a **constant** hidden-class neuron `konst` and a
/// variance-carrying survivor `sib` both feed the shared output `out-0`. Used to
/// prove that routing is by neuron *class*: `konst` is left for the #1623 bias
/// fold even though a shared-target survivor exists.
fn creature_with_constant_neuron() -> CreatureJson {
    serde_json::from_str(
        r#"{
            "input": 1, "output": 1,
            "neurons": [
                {"uuid": "input-0", "type": "input"},
                {"uuid": "konst", "type": "constant", "squash": "IDENTITY"},
                {"uuid": "sib", "type": "hidden", "squash": "IDENTITY"},
                {"uuid": "out-0", "type": "output", "squash": "IDENTITY"}
            ],
            "synapses": [
                {"fromUUID": "input-0", "toUUID": "sib", "weight": 1.0},
                {"fromUUID": "konst", "toUUID": "out-0", "weight": 1.0},
                {"fromUUID": "sib", "toUUID": "out-0", "weight": 1.0}
            ]
        }"#,
    )
    .expect("valid creature JSON")
}

/// Perfectly-correlated per-sample activations for two neurons, aligned on
/// `obs_index`. The candidate genuinely varies (carries a per-sample signal) and
/// the survivor tracks it exactly (ρ = 1).
fn correlated_records(candidate_uuid: &str, survivor_uuid: &str) -> Vec<DiscoverRecord> {
    let activations = [1.0_f32, 2.0, 3.0];
    let mut records = Vec::new();
    for (obs, &a) in activations.iter().enumerate() {
        let obs = u32::try_from(obs).expect("small index");
        records.push(record(obs, candidate_uuid, a));
        records.push(record(obs, survivor_uuid, a));
    }
    records
}

/// The core wiring: a sole-op `RemoveNeuron` candidate whose removed neuron
/// carries per-sample variance is emitted with covariance stats and a
/// weight-redistribution remedy (optimal `delta_weight`, residual variances,
/// `fully_compensable`).
#[test]
fn remove_neuron_candidate_carries_covariance_and_redistribution() {
    let creature = creature_with_deep_neuron();
    let records = correlated_records("deep", "sib");
    let mut candidates = vec![remove_neuron_candidate("deep", -0.05)];

    let attached = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    assert_eq!(
        attached, 1,
        "the variance-carrying candidate is compensated"
    );

    let comp = candidates[0]
        .remove_neuron_compensation
        .as_ref()
        .expect("compensation attached to the candidate");

    assert_eq!(comp.survivor_neuron_uuid, "sib");
    assert_eq!(comp.target_neuron_uuid, "out-0");
    assert_eq!(comp.sample_count, 3);
    // Δw = w_c · cov / var(a_s) = 1 · (2/3) / (2/3) = 1.0.
    assert!(
        (comp.delta_weight - 1.0).abs() < 1e-5,
        "optimal weight bump must be 1.0, got {}",
        comp.delta_weight
    );
    // Perfect correlation ⇒ the removal becomes non-regressive.
    assert!(
        comp.fully_compensable,
        "a perfectly-correlated survivor makes the removal fully compensable"
    );
    assert!(
        (comp.correlation - 1.0).abs() < 1e-5,
        "correlation must be ~1.0, got {}",
        comp.correlation
    );
    assert!(
        comp.variance_recovered > 0.0,
        "redistribution must recover per-sample variance, got {}",
        comp.variance_recovered
    );
    assert!(
        comp.redistributed_residual_variance <= comp.bias_only_residual_variance + 1e-6,
        "redistribution must not add residual variance over the mean-only fold"
    );
}

/// Routing test: a constant-neuron candidate is **not** given a redistribution
/// remedy here — it routes to the #1623 bias-fold path — even when a
/// shared-target survivor with correlated records exists.
#[test]
fn constant_neuron_candidate_is_not_given_redistribution() {
    let creature = creature_with_constant_neuron();
    let records = correlated_records("konst", "sib");
    let mut candidates = vec![remove_neuron_candidate("konst", -0.01)];

    let attached = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    assert_eq!(attached, 0, "constant neurons route to the #1623 bias fold");
    assert!(
        candidates[0].remove_neuron_compensation.is_none(),
        "constant-neuron candidate must carry no redistribution remedy"
    );
}

/// Multi-operation coordinated candidates are not bare neuron removals, so they
/// receive no compensation.
#[test]
fn multi_op_candidate_gets_no_compensation() {
    let creature = creature_with_deep_neuron();
    let records = correlated_records("deep", "sib");
    let mut candidates = vec![CoordinatedStructuralCandidateJson {
        operations: vec![
            CoordinatedStructuralOpJson::RemoveSynapse {
                from_neuron_uuid: "input-0".to_string(),
                to_neuron_uuid: "deep".to_string(),
            },
            CoordinatedStructuralOpJson::RemoveNeuron {
                neuron_uuid: "deep".to_string(),
            },
        ],
        expected_creature_score_gain: -0.05,
        comment: None,
        remove_neuron_compensation: None,
    }];

    let attached = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    assert_eq!(attached, 0, "multi-op candidates are not compensated");
    assert!(candidates[0].remove_neuron_compensation.is_none());
}

/// Non-`RemoveNeuron` single-op candidates receive no compensation.
#[test]
fn non_remove_neuron_candidate_gets_no_compensation() {
    let creature = creature_with_deep_neuron();
    let records = correlated_records("deep", "sib");
    let mut candidates = vec![make_candidate(0.42)]; // a RemoveSynapse candidate

    let attached = apply_remove_neuron_compensation(&creature, &records, &mut candidates);
    assert_eq!(
        attached, 0,
        "non-RemoveNeuron candidates are not compensated"
    );
    assert!(candidates[0].remove_neuron_compensation.is_none());
}

/// A variance-carrying candidate with no shared-target survivor records cannot
/// evaluate counterfactual (d), so no remedy is fabricated — the field stays
/// `None` (the applier flags such removals rather than folding the mean).
#[test]
fn variance_candidate_without_records_gets_no_compensation() {
    let creature = creature_with_deep_neuron();
    let mut candidates = vec![remove_neuron_candidate("deep", -0.05)];

    // No per-sample records supplied.
    let attached = apply_remove_neuron_compensation(&creature, &[], &mut candidates);
    assert_eq!(attached, 0, "no records ⇒ (d) cannot be evaluated");
    assert!(
        candidates[0].remove_neuron_compensation.is_none(),
        "no remedy is fabricated without per-sample data"
    );
}
