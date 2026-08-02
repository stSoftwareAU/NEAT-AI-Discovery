//! Issue #1925 — a barren discovery pass must say which gate silenced it.
//!
//! The #1920 cache study found 57% of production runs cache nothing and that
//! four strategies contribute twelve records between them across 47 days. The
//! study could not say *why*, because the cache only holds candidates the
//! controller evaluated: "candidates rejected inside discovery never appear, so
//! this corpus cannot say why a run proposed nothing".
//!
//! Discovery already counted every candidate-level rejection (#1129), but the
//! four gates that suppress a **whole discovery module** — the historical
//! success-rate gate (#1060), the deadline skip (#1029), a caught detection
//! panic (#1087) and creature-scale tiering (#1547) — recorded nothing at all.
//! A module that was never asked looked exactly like a module that ran and
//! found nothing, which is precisely the distinction the issue needs.
//!
//! These tests pin the new attribution behaviourally:
//!
//! 1. Each module-level skip is counted under its own stable reason and
//!    classified as upstream (generation-side) evidence.
//! 2. A module that ran and found nothing is **not** counted as a skip.
//! 3. `zeroCandidateSummary` reports the starvation classification and the
//!    generation-signal counts behind it.

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use neat_ai_discovery::analysis::candidate_starvation::{
    GenerationSignals, StarvationClass, UPSTREAM_REJECTION_REASONS, signals_from_breakdown,
};
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    ALL_REJECTION_REASONS, REJECTION_MODULE_DEADLINE_SKIPPED, REJECTION_MODULE_GATED_LOW_SUCCESS,
    REJECTION_MODULE_PANICKED, REJECTION_MODULE_TIERED_OUT,
};
use neat_ai_discovery::analysis::discovery_dispatch::{
    DiscoveryDetectionResult, DiscoveryModuleDetectionEntry, DiscoveryModuleDetectionResults,
    DiscoveryModuleSpec, ModuleSkipReason, detect_discovery_modules_parallel,
    merge_discovery_module_results,
};
use neat_ai_discovery::analysis::module_weights::ModuleOutcomeTracker;
use neat_ai_discovery::analysis::shared::{AnalyzeSynapsesResult, SynapseAnalysisMetadata};
use neat_ai_discovery::{
    CoordinatedStructuralCandidateJson, CoordinatedStructuralOpJson, EnvironmentalGatesJson,
    build_zero_candidate_summary,
};

const MODULE: &str = "test-module";

fn empty_synapse_result() -> AnalyzeSynapsesResult {
    AnalyzeSynapsesResult {
        helpful_synapses: Vec::new(),
        harmful_synapses: Vec::new(),
        synapse_weight_updates: Vec::new(),
        coordinated_structural_candidates: Vec::new(),
        candidate_clusters: Vec::new(),
        gpu_used: false,
        no_candidate_reasons: Vec::new(),
        metadata: SynapseAnalysisMetadata::default(),
    }
}

fn candidate(gain: f32) -> CoordinatedStructuralCandidateJson {
    CoordinatedStructuralCandidateJson {
        remove_neuron_compensation: None,
        constant_neuron_bias_fold: None,
        operations: vec![CoordinatedStructuralOpJson::RemoveSynapse {
            from_neuron_uuid: "a".to_string(),
            to_neuron_uuid: "b".to_string(),
        }],
        expected_creature_score_gain: gain,
        comment: Some("test".to_string()),
    }
}

fn spec(
    detect_fn: Box<dyn FnOnce() -> Option<DiscoveryDetectionResult> + Send>,
) -> Vec<DiscoveryModuleSpec> {
    vec![DiscoveryModuleSpec {
        module_name: MODULE.to_string(),
        phase_name: "issue_1925_test_phase",
        max_candidates: 0,
        detect_fn,
    }]
}

/// A tracker whose only module sits below `MODULE_GATE_THRESHOLD` (0.005).
///
/// Ten real ablation attempts with no successes plus accumulated pre-filtering
/// soft failures drive the Bayesian rate to `1 / 212 ≈ 0.0047`. This is the
/// production shape of the gate: a prolific module whose own truncations feed
/// the soft-failure count that eventually silences it.
fn gated_tracker() -> ModuleOutcomeTracker {
    let mut tracker = ModuleOutcomeTracker::new();
    for _ in 0..10 {
        tracker.record(MODULE, false);
    }
    tracker.record_soft_failures(MODULE, 400, 0.5);
    assert!(
        tracker.is_gated_default(MODULE),
        "fixture must actually be gated, rate was {}",
        tracker.stats(MODULE).success_rate()
    );
    tracker
}

/// Run detection + merge and return the resulting rejection counts.
fn breakdown_after_dispatch(
    modules: Vec<DiscoveryModuleSpec>,
    tracker: &mut ModuleOutcomeTracker,
    deadline: Option<SystemTime>,
) -> (AnalyzeSynapsesResult, HashMap<String, u32>) {
    let mut syn = empty_synapse_result();
    let detection = detect_discovery_modules_parallel(modules, deadline, Some(tracker));
    merge_discovery_module_results(&mut syn, detection, None, false, tracker);
    let counts = syn.metadata.rejection_breakdown.counts().clone();
    (syn, counts)
}

/// Every module-level skip reason must be a documented reason and must be
/// classified as upstream — a module that never ran cannot be evidence that the
/// accept gate over-rejected.
#[test]
fn module_skip_reasons_are_documented_and_classified_upstream() {
    for reason in [
        REJECTION_MODULE_GATED_LOW_SUCCESS,
        REJECTION_MODULE_DEADLINE_SKIPPED,
        REJECTION_MODULE_PANICKED,
        REJECTION_MODULE_TIERED_OUT,
    ] {
        assert!(
            ALL_REJECTION_REASONS.contains(&reason),
            "{reason} must be listed in ALL_REJECTION_REASONS"
        );
        assert!(
            UPSTREAM_REJECTION_REASONS.contains(&reason),
            "{reason} must classify as upstream (generation-side) evidence"
        );
    }
}

/// The success-rate gate (#1060) is the starvation ratchet named in the issue:
/// a gated module proposes nothing, so it earns no new ablation attempts and can
/// never recover. The suppression must appear in the breakdown and against the
/// module in `discoveryModuleStats`.
#[test]
fn gated_module_is_counted_and_named_in_module_stats() {
    let mut tracker = gated_tracker();
    let modules = spec(Box::new(|| {
        panic!("a gated module's detection closure must never run");
    }));

    let (syn, counts) = breakdown_after_dispatch(modules, &mut tracker, None);

    assert_eq!(
        counts.get(REJECTION_MODULE_GATED_LOW_SUCCESS).copied(),
        Some(1),
        "one count per suppressed module, got {counts:?}"
    );
    let stats = syn
        .metadata
        .discovery_module_stats
        .iter()
        .find(|s| s.module_name == MODULE)
        .expect("the gated module must still be listed in discoveryModuleStats");
    assert_eq!(
        stats.skipped.as_deref(),
        Some(REJECTION_MODULE_GATED_LOW_SUCCESS),
        "the module's own stats must name why it never ran"
    );
    assert_eq!(stats.candidates_produced, 0);
}

/// A module skipped because the analysis deadline had already passed is a
/// budget failure, not search exhaustion, and must be counted as such.
#[test]
fn deadline_skipped_module_is_counted() {
    let mut tracker = ModuleOutcomeTracker::new();
    let modules = spec(Box::new(|| {
        panic!("a deadline-skipped module's detection closure must never run");
    }));
    let past = SystemTime::now() - Duration::from_secs(60);

    let (_syn, counts) = breakdown_after_dispatch(modules, &mut tracker, Some(past));

    assert_eq!(
        counts.get(REJECTION_MODULE_DEADLINE_SKIPPED).copied(),
        Some(1),
        "deadline skip must be counted, got {counts:?}"
    );
}

/// A caught detection panic (#1087) yields an empty result. Without its own
/// count that is indistinguishable from a module that ran cleanly and found
/// nothing — the failure would be silent.
#[test]
fn panicking_module_is_counted_rather_than_silently_empty() {
    let mut tracker = ModuleOutcomeTracker::new();
    let modules = spec(Box::new(|| panic!("deliberate detection panic")));

    let (_syn, counts) = breakdown_after_dispatch(modules, &mut tracker, None);

    assert_eq!(
        counts.get(REJECTION_MODULE_PANICKED).copied(),
        Some(1),
        "a caught panic must be counted, got {counts:?}"
    );
}

/// Tiered-out modules (#1547) are removed before dispatch, so they reach the
/// merge phase as pre-built skipped entries. Merging one must count it.
#[test]
fn tiered_out_module_is_counted() {
    let mut syn = empty_synapse_result();
    let mut tracker = ModuleOutcomeTracker::new();
    let detection = DiscoveryModuleDetectionResults {
        entries: vec![DiscoveryModuleDetectionEntry::skipped(
            "multi-hop analysis".to_string(),
            "issue_1925_test_phase",
            ModuleSkipReason::TieredOut,
        )],
    };

    merge_discovery_module_results(&mut syn, detection, None, false, &mut tracker);

    assert_eq!(
        syn.metadata
            .rejection_breakdown
            .counts()
            .get(REJECTION_MODULE_TIERED_OUT)
            .copied(),
        Some(1),
        "a tiered-out module must be counted, not merely logged"
    );
}

/// The counterpart guard: a module that genuinely ran must never be counted as
/// skipped, whether it found candidates or not. Conflating the two would make
/// every quiet pass look suppressed.
#[test]
fn module_that_ran_is_never_counted_as_skipped() {
    let skip_reasons = [
        REJECTION_MODULE_GATED_LOW_SUCCESS,
        REJECTION_MODULE_DEADLINE_SKIPPED,
        REJECTION_MODULE_PANICKED,
        REJECTION_MODULE_TIERED_OUT,
    ];

    for (label, detect) in [
        (
            "found nothing",
            Box::new(|| None) as Box<dyn FnOnce() -> Option<DiscoveryDetectionResult> + Send>,
        ),
        (
            "found a candidate",
            Box::new(|| {
                Some(DiscoveryDetectionResult {
                    detected_count: 1,
                    candidates: vec![candidate(1.0)],
                })
            }),
        ),
    ] {
        let mut tracker = ModuleOutcomeTracker::new();
        let (syn, counts) = breakdown_after_dispatch(spec(detect), &mut tracker, None);

        for reason in skip_reasons {
            assert!(
                !counts.contains_key(reason),
                "module that {label} must not record {reason}, got {counts:?}"
            );
        }
        let stats = syn
            .metadata
            .discovery_module_stats
            .iter()
            .find(|s| s.module_name == MODULE)
            .expect("module must be listed");
        assert!(
            stats.skipped.is_none(),
            "module that {label} must carry no skip reason"
        );
    }
}

/// Module-level skips must reach the starvation classifier as upstream drops,
/// so a pass silenced entirely by suppressed modules classifies as starved
/// rather than over-rejected.
#[test]
fn a_pass_silenced_by_module_skips_classifies_as_candidate_starved() {
    let mut breakdown = RejectionBreakdown::new();
    breakdown.record_many(REJECTION_MODULE_TIERED_OUT, 7);
    breakdown.record_many(REJECTION_MODULE_GATED_LOW_SUCCESS, 3);

    let signals = signals_from_breakdown(&breakdown, 0);
    assert_eq!(signals.upstream_rejections, 10);
    assert_eq!(signals.proposals_formed(), 0);

    let class = neat_ai_discovery::analysis::candidate_starvation::classify(
        &signals,
        &neat_ai_discovery::analysis::candidate_starvation::StarvationConfig::default(),
    );
    assert_eq!(class, StarvationClass::CandidateStarved);
}

/// A barren pass must report *which failure mode* it is in and the counts
/// behind that verdict — the classification has been computed since #1739 but
/// was thrown away instead of being shown to the operator.
#[test]
fn zero_candidate_summary_reports_starvation_class_and_signals() {
    let mut pass_breakdown = RejectionBreakdown::new();
    pass_breakdown.record_many(REJECTION_MODULE_TIERED_OUT, 7);

    let signals = GenerationSignals {
        accepted: 0,
        gate_side_rejections: 2,
        upstream_rejections: 7,
        abundance_rejections: 1,
    };

    let summary = build_zero_candidate_summary(
        None,
        None,
        &pass_breakdown,
        EnvironmentalGatesJson {
            memory_budget_exceeded: false,
            memory_pressure_cancelled: false,
            cancelled: false,
            environmentally_disabled: None,
        },
        signals,
        StarvationClass::CandidateStarved,
    );

    assert_eq!(summary.starvation_class, "candidateStarved");
    assert_eq!(summary.generation_signals.upstream_rejections, 7);
    assert_eq!(summary.generation_signals.reaching_gate, 2);
    assert_eq!(summary.generation_signals.proposals_formed, 3);
    assert_eq!(
        summary.dominant_rejection_reason.as_deref(),
        Some(REJECTION_MODULE_TIERED_OUT)
    );

    let json = serde_json::to_value(&summary).expect("summary serialises");
    assert_eq!(json["starvationClass"], "candidateStarved");
    assert_eq!(json["generationSignals"]["proposalsFormed"], 3);
    assert_eq!(json["generationSignals"]["upstreamRejections"], 7);
    assert_eq!(json["rejectionBreakdown"][REJECTION_MODULE_TIERED_OUT], 7);
}
