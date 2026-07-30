//! Issue #1793 — `ModuleStarvationTracker` is deleted, not wired.
//!
//! The tracker (Issue #1273) was never populated in production: the only
//! production caller reached it through a wrapper that hard-coded
//! `starvation_tracker = None, current_epoch = 0`, so the detection-time
//! starvation gate could never fire and `starvedModuleCount` in the drought
//! diagnostic was structurally always `0`.
//!
//! Decision (B) — delete. These tests pin the removal behaviourally so a
//! partial resurrection is caught:
//!
//! 1. The surviving dispatch entry point still applies the two *live* gates
//!    (the population-wide module gate, Issue #1060, and the deadline gate,
//!    Issue #1029) and skips nothing else.
//! 2. `module_starved` is gone from the rejection vocabulary — with no
//!    recorder left it could only ever have been a permanent zero.
//! 3. The drought diagnostic no longer serialises `starvedModuleCount`.
//! 4. The two dead operator levers are gone from the canonical env-var
//!    reference.

use neat_ai_discovery::analysis::candidate_starvation::UPSTREAM_REJECTION_REASONS;
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::ALL_REJECTION_REASONS;
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleSpec, detect_discovery_modules_parallel,
};
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::drought_diagnostic::{DroughtInputs, emit_drought_diagnostic};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::{CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson};

const CONFIGURATION: &str = include_str!("../docs/CONFIGURATION.md");

/// The removed per-(creature, module) starvation reason.
const REMOVED_REJECTION_REASON: &str = "module_starved";

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

fn spec(name: &str, phase: &'static str) -> DiscoveryModuleSpec {
    DiscoveryModuleSpec {
        module_name: name.to_string(),
        phase_name: phase,
        max_candidates: 0,
        detect_fn: Box::new(|| {
            Some(DiscoveryDetectionResult {
                detected_count: 1,
                candidates: vec![make_candidate(1.0)],
            })
        }),
    }
}

/// With the starvation gate removed, a module with a long consecutive-failure
/// history that is *not* gated by the population-wide success-rate threshold
/// must still have its `detect_fn` executed. Previously this was the only
/// module class the (dead) starvation cooldown would have suppressed.
#[test]
fn repeatedly_failing_module_still_runs_after_starvation_gate_removal() {
    let mut tracker = ModuleOutcomeTracker::new();
    // A long failure streak, but paired with enough successes to stay above
    // MODULE_GATE_THRESHOLD — exactly the shape the starvation cooldown used to
    // catch and the module gate does not.
    for _ in 0..20 {
        tracker.record("streaky-module", true);
        tracker.record("streaky-module", false);
    }

    let results = detect_discovery_modules_parallel(
        vec![spec("streaky-module", "test_streaky")],
        None,
        Some(&tracker),
    );

    assert_eq!(results.entries.len(), 1);
    assert!(
        results.entries[0].result.is_some(),
        "no starvation cooldown survives: an ungated module must always run"
    );
}

/// The live module gate (Issue #1060) must survive the collapse of the
/// `_with_starvation` wrapper into the single entry point.
#[test]
fn module_gate_survives_wrapper_collapse() {
    let mut tracker = ModuleOutcomeTracker::new();
    // Bayesian rate 1/502 ≈ 0.002 < MODULE_GATE_THRESHOLD (0.005).
    for _ in 0..500 {
        tracker.record("failing-module", false);
    }
    for _ in 0..20 {
        tracker.record("healthy-module", true);
        tracker.record("healthy-module", false);
    }

    let results = detect_discovery_modules_parallel(
        vec![
            spec("failing-module", "test_fail"),
            spec("healthy-module", "test_healthy"),
        ],
        None,
        Some(&tracker),
    );

    assert_eq!(results.entries.len(), 2);
    assert!(
        results.entries[0].result.is_none(),
        "gated module must still be skipped"
    );
    assert!(
        results.entries[1].result.is_some(),
        "healthy module must still run"
    );
}

/// The deadline gate (Issue #1029) must also survive the collapse.
#[test]
fn deadline_gate_survives_wrapper_collapse() {
    let past = Some(std::time::SystemTime::now() - std::time::Duration::from_secs(60));

    let results =
        detect_discovery_modules_parallel(vec![spec("any-module", "test_any")], past, None);

    assert_eq!(results.entries.len(), 1);
    assert!(
        results.entries[0].result.is_none(),
        "an expired deadline must still skip the module"
    );
}

/// `module_starved` had exactly one recorder — the merge phase reacting to the
/// starvation flag. With the tracker gone it can never be counted again, so it
/// must not remain in the documented vocabulary or in the starvation
/// classifier's upstream partition.
#[test]
fn module_starved_is_no_longer_a_rejection_reason() {
    assert!(
        !ALL_REJECTION_REASONS.contains(&REMOVED_REJECTION_REASON),
        "`{REMOVED_REJECTION_REASON}` has no recorder left and must be removed \
         from ALL_REJECTION_REASONS"
    );
    assert!(
        !UPSTREAM_REJECTION_REASONS.contains(&REMOVED_REJECTION_REASON),
        "`{REMOVED_REJECTION_REASON}` must not remain in the candidate-starvation \
         upstream partition"
    );
}

/// The drought diagnostic must not carry a `starvedModuleCount` field that can
/// only ever serialise as `0` (the same treatment `candidateCacheSize` got in
/// Issue #1792).
#[test]
fn drought_diagnostic_json_has_no_starved_module_count() {
    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many("below_threshold", 3);

    let inputs = DroughtInputs {
        consecutive_failures: 9,
        rolling_success_rate: 0.0,
        discovery_mode: DiscoveryMode::Conservative,
        target_tracker: None,
        current_epoch: 4,
        target_cooldown_skipped: 0,
        rejection_breakdown: &breakdown,
        candidates_returned: 0,
    };

    let diagnostic = emit_drought_diagnostic(&inputs, 5).expect("drought is active");
    let json = serde_json::to_value(&diagnostic).expect("diagnostic serialises");
    let object = json.as_object().expect("diagnostic is a JSON object");

    assert!(
        !object.contains_key("starvedModuleCount"),
        "starvedModuleCount was structurally always 0 and must not be serialised; got {object:?}"
    );
    // The surviving suppression counters are untouched.
    assert!(object.contains_key("targetCooldownActiveCount"));
    assert!(object.contains_key("dominantRejectionReason"));
}

/// The two operator levers configured a tracker that no longer exists. A lever
/// that silently does nothing is worse than no lever, so they must be gone from
/// the canonical reference.
#[test]
fn starvation_env_vars_are_removed_from_the_canonical_reference() {
    for var in [
        "NEAT_AI_DISCOVERY_MODULE_STARVATION_FAILURE_STREAK",
        "NEAT_AI_DISCOVERY_MODULE_STARVATION_COOLDOWN_EPOCHS",
    ] {
        assert!(
            !CONFIGURATION.contains(var),
            "`{var}` configures a deleted tracker and must not remain documented \
             in docs/CONFIGURATION.md"
        );
    }
}
