//! Issue #1545 — squash-aware `ACTIVATION_SPECS` pruning for hidden add-neuron
//! targets (GRQ scale).
//!
//! Acceptance criteria exercised here:
//! - (a) A creature with **zero** historical success for a squash family gets
//!   that family pruned from hidden scans (cold-start core set).
//! - (b) Drought / novelty escalation restores the **full** scan set — the
//!   guard against a filter that only helps toy networks (permanent pruning
//!   with no recovery path).
//! - (c) The config count per (source, target) respects the top-N cap.
//!
//! These mirror the pattern of `issue_1315_role_task_aware_scan.rs` and pin the
//! public planner contract that the neuron-analysis wiring
//! (`build_neuron_scan_plan`) is built on.

use neat_ai_discovery::analysis::activation::{
    ACTIVATION_SPECS, CORE_HIDDEN_SCAN_NAMES, HiddenScanContext, SquashScanPlan, build_scan_configs,
};
use std::collections::HashSet;

fn names(
    specs: &[&'static neat_ai_discovery::analysis::activation::ActivationCandidateSpec],
) -> Vec<&'static str> {
    specs.iter().map(|s| s.name).collect()
}

/// (a) Cold start: no recorded history ⇒ the hidden scan is exactly the core
/// set, so every non-core squash family (with zero historical success) is
/// pruned.
#[test]
fn cold_start_prunes_families_with_zero_history() {
    let history = HashSet::new();
    let ctx = HiddenScanContext {
        successful_squashes: &history,
        escalated: false,
    };
    let got = names(&ctx.scan_specs());

    // Core families present.
    for core in CORE_HIDDEN_SCAN_NAMES {
        assert!(
            got.contains(core),
            "core family {core} must remain, got {got:?}"
        );
    }
    // Non-core families with no history pruned.
    for pruned in [
        "BIPOLAR", "CLIPPED", "ArcTan", "ABSOLUTE", "Softplus", "LOGISTIC",
    ] {
        assert!(
            !got.contains(&pruned),
            "family {pruned} with zero history must be pruned, got {got:?}",
        );
    }
    assert!(
        ctx.scan_specs().len() < ACTIVATION_SPECS.len(),
        "cold-start scan must be strictly smaller than the full set",
    );
}

/// (a) History-aware widening: a family the creature has adopted is scanned in
/// addition to the core set, while other non-core families stay pruned.
#[test]
fn history_widens_only_adopted_families() {
    let mut history = HashSet::new();
    history.insert("BIPOLAR".to_string());
    let ctx = HiddenScanContext {
        successful_squashes: &history,
        escalated: false,
    };
    let got = names(&ctx.scan_specs());

    assert!(
        got.contains(&"BIPOLAR"),
        "adopted family must be scanned: {got:?}"
    );
    assert!(
        !got.contains(&"CLIPPED"),
        "unadopted family must stay pruned: {got:?}"
    );
    // Core still present.
    assert!(got.contains(&"IDENTITY"));
}

/// (a) History matching is case-insensitive: creature squashes are uppercased
/// at deserialisation (Issue #753) while some spec names are mixed-case.
#[test]
fn history_match_is_case_insensitive() {
    // ArcTan is a non-core, mixed-case spec name; a creature that adopted it is
    // stored uppercase as "ARCTAN".
    let mut history = HashSet::new();
    history.insert("ARCTAN".to_string());
    let ctx = HiddenScanContext {
        successful_squashes: &history,
        escalated: false,
    };
    let got = names(&ctx.scan_specs());
    assert!(
        got.contains(&"ArcTan"),
        "uppercase history must match mixed-case spec: {got:?}"
    );
}

/// (b) Escalation restores the full set — the critical guard against permanent
/// over-pruning. A plateaued creature widens back to every family regardless of
/// its (empty) history.
#[test]
fn escalation_restores_full_scan_set() {
    let history = HashSet::new();
    let ctx = HiddenScanContext {
        successful_squashes: &history,
        escalated: true,
    };
    assert_eq!(
        ctx.scan_specs().len(),
        ACTIVATION_SPECS.len(),
        "escalated hidden scan must restore the full ACTIVATION_SPECS set",
    );
}

/// (c) The per-(source, target) config cap bounds the emitted config count and
/// keeps the most central scales.
#[test]
fn config_cap_bounds_per_pair_count() {
    // Full set under escalation ⇒ the largest possible config count.
    let history = HashSet::new();
    let full = HiddenScanContext {
        successful_squashes: &history,
        escalated: true,
    }
    .scan_specs();

    let uncapped = build_scan_configs(&full, 0);
    assert!(
        uncapped.len() > 100,
        "sanity: full cross-product is large: {}",
        uncapped.len()
    );

    let cap = 20;
    let capped = build_scan_configs(&full, cap);
    assert_eq!(
        capped.len(),
        cap,
        "cap must bound the per-pair config count"
    );

    // The retained scales must be the ones closest to 1.0 — extreme scales are
    // dropped first (they are flagged numerically unstable in the spec docs).
    let max_retained = capped
        .iter()
        .map(|c| (c.scale.ln()).abs())
        .fold(0.0_f32, f32::max);
    let min_dropped = build_scan_configs(&full, 0)
        .iter()
        .filter(|c| {
            !capped
                .iter()
                .any(|k| (k.scale - c.scale).abs() < f32::EPSILON)
        })
        .map(|c| (c.scale.ln()).abs())
        .fold(f32::INFINITY, f32::min);
    assert!(
        max_retained <= min_dropped + f32::EPSILON,
        "retained scales ({max_retained}) must be at least as central as dropped ({min_dropped})",
    );
}

/// The cold-start plan is a genuine reduction versus the full plan — this is the
/// "measurable drop in activation GPU configs" success criterion, computed
/// deterministically without a GPU.
#[test]
fn cold_start_plan_reduces_gpu_config_count() {
    let history = HashSet::new();
    let cold = SquashScanPlan::for_hidden(
        &HiddenScanContext {
            successful_squashes: &history,
            escalated: false,
        },
        0,
    );
    let full = SquashScanPlan::full();

    let cold_configs = cold.configs().len();
    let full_configs = full.configs().len();

    assert!(
        cold_configs < full_configs,
        "cold-start plan ({cold_configs}) must emit fewer configs than full ({full_configs})",
    );
    // Guard the headline number: the cross-product should shrink by at least a
    // quarter at cold start (integer comparison avoids lossy float casts).
    assert!(
        cold_configs * 4 < full_configs * 3,
        "cold-start config count ({cold_configs}) should be < 75% of full ({full_configs})",
    );
}
