//! Fail-loud candidate reconciliation through the real dispatch path
//! (Issue #1802).
//!
//! Each sub-issue of #1782 wired one silent drop path into the rejection
//! breakdown, but nothing stopped the *next* one being added silently — all six
//! paths the diagnosis found were bare `continue`s that compiled, passed tests
//! and shipped. These tests pin the enforced invariant that replaces that
//! convention:
//!
//! ```text
//! considered == accounted   (per surface, per pass)
//! ```
//!
//! The primary cases drive `analyze_all` — the real discovery-pass entry point —
//! with a seeded creature and assert *positively* that each surface reconciled
//! with zero unaccounted candidates, rather than inferring success from the
//! absence of a failure marker. The companion cases guard the guard: a
//! deliberately un-counted drop injected into a ledger fixture must fail with a
//! message naming the surface and the delta, and must fail loudly under strict
//! mode.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::candidate_reconciliation::{
    CandidateLedger, Reconciliation, SURFACE_NEURON, SURFACE_SYNAPSE, StrictModeGuard, reconcile,
    strict_mode,
};
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    ALL_REJECTION_REASONS, REJECTION_UNACCOUNTED_DROP,
};
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::analysis::shared::AnalyzeAllResult;
use neat_ai_discovery::analysis::target_failure_tracker::reset_global_tracker;
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Fixture: the same shape as the #1795 suppression-wiring guard — a small
// forward-only creature whose recorded errors drive real candidate generation.
// ---------------------------------------------------------------------------

fn build_creature() -> CreatureJson {
    CreatureJson {
        neurons: vec![
            NeuronJson {
                uuid: "input-0".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "input-1".to_string(),
                neuron_type: "input".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "hidden-0".to_string(),
                neuron_type: "hidden".to_string(),
                squash: "TANH".to_string(),
                bias: 0.0,
            },
            NeuronJson {
                uuid: "output-0".to_string(),
                neuron_type: "output".to_string(),
                squash: "IDENTITY".to_string(),
                bias: 0.0,
            },
        ],
        synapses: vec![
            SynapseJson {
                from_uuid: "input-0".to_string(),
                to_uuid: "hidden-0".to_string(),
                weight: 0.5,
                synapse_type: None,
            },
            SynapseJson {
                from_uuid: "hidden-0".to_string(),
                to_uuid: "output-0".to_string(),
                weight: 0.4,
                synapse_type: None,
            },
        ],
        input: 2,
        output: 1,
    }
}

fn build_records(creature: &CreatureJson, count: usize, error_scale: f32) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..count as u32 {
        let t = obs as f32 / count as f32;
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (act, Some(act), vec![error_scale * (t * 3.0).cos()])
                }
                "output" => {
                    let error = error_scale * (t * std::f32::consts::PI).cos();
                    (0.5 + 0.2 * t, Some(0.5 + 0.2 * t), vec![error])
                }
                _ => continue,
            };
            records.push(DiscoverRecord {
                obs_index: obs,
                neuron_uuid: neuron.uuid.clone(),
                value,
                activation,
                errors,
            });
        }
    }
    records
}

fn write_fixture_parquet(error_scale: f32) -> (TempDir, String) {
    let creature = build_creature();
    let records = build_records(&creature, 60, error_scale);
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let parquet_path = temp_dir.path().join("candidate_reconciliation.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("parquet path is valid UTF-8")
        .to_string();
    write_records_to_parquet(&parquet_file, &records).expect("write parquet");
    (temp_dir, parquet_file)
}

fn make_input(parquet_file: String) -> AnalyzeAllInput {
    let creature = build_creature();
    let focus_neurons: Vec<String> = creature
        .neurons
        .iter()
        .filter(|n| n.neuron_type == "output" || n.neuron_type == "hidden")
        .map(|n| n.uuid.clone())
        .collect();

    AnalyzeAllInput {
        parquet_file,
        creature,
        focus_neurons,
        max_synapse_candidates: None,
        max_neuron_candidates: None,
        analysis_deadline_ms: None,
        include_synapse_analysis: Some(true),
        include_neuron_analysis: Some(true),
        random_seed: Some(42),
        previous_neuron_fingerprints: None,
        module_outcome_tracker: None,
        max_analysis_memory_mb: None,
        max_discovery_wall_clock_minutes: None,
        temperature: 1.0,
        failure_cache: None,
        discovery_outcome_log: None,
        cost_name: None,
    }
}

/// Skip on GPU-less hosts: without a GPU no candidate is ever formed, so the
/// ledger has nothing to reconcile and the guard would pass vacuously.
fn gpu_or_skip(test: &str) -> bool {
    if GpuAnalyzer::gpu_is_available() {
        return true;
    }
    eprintln!("Skipping {test}: no GPU available");
    false
}

/// Both surfaces' reconciliations from one pass, each paired with the
/// surface's final rejection breakdown.
fn reconciliations(result: &AnalyzeAllResult) -> Vec<(Reconciliation, &RejectionBreakdown)> {
    let mut out = Vec::new();
    if let Some(synapse) = result.synapse.as_ref() {
        let reconciliation = synapse
            .metadata
            .candidate_reconciliation
            .expect("the synapse surface must attach its reconciliation (Issue #1802)");
        out.push((reconciliation, &synapse.metadata.rejection_breakdown));
    }
    if let Some(neuron) = result.neuron.as_ref() {
        let reconciliation = neuron
            .metadata
            .candidate_reconciliation
            .expect("the neuron surface must attach its reconciliation (Issue #1802)");
        out.push((reconciliation, &neuron.metadata.rejection_breakdown));
    }
    out
}

fn assert_balanced(reconciliation: Reconciliation, breakdown: &RejectionBreakdown) {
    assert!(reconciliation.balanced(), "{}", reconciliation.message());
    assert_eq!(
        reconciliation.unaccounted, 0,
        "surface {} lost {} candidate(s) with no recorded verdict",
        reconciliation.surface, reconciliation.unaccounted
    );
    assert_eq!(
        breakdown.counts().get(REJECTION_UNACCOUNTED_DROP),
        None,
        "a balanced pass must not record `{REJECTION_UNACCOUNTED_DROP}` on the {} surface",
        reconciliation.surface
    );
}

// ---------------------------------------------------------------------------
// Primary guards — the real dispatch path.
// ---------------------------------------------------------------------------

/// A converged creature (errors in the ~1e-7 noise band) generates candidates
/// that are all rejected. Every one of those rejections must be accounted for.
#[test]
#[serial]
fn converged_pass_accounts_for_every_candidate() {
    if !gpu_or_skip("converged_pass_accounts_for_every_candidate") {
        return;
    }
    reset_global_tracker();
    let (_temp_dir, parquet_file) = write_fixture_parquet(1e-7);

    let result = analyze_all(&make_input(parquet_file)).expect("analyze_all");

    let observed = reconciliations(&result);
    assert!(
        !observed.is_empty(),
        "the pass must reconcile at least one surface"
    );
    for (reconciliation, breakdown) in observed {
        assert_balanced(reconciliation, breakdown);
    }
}

/// An improvable creature accepts some candidates and rejects others, so both
/// sides of the identity are non-trivially populated. This is the case that
/// proves the ledger is genuinely wired rather than balancing at zero.
#[test]
#[serial]
fn improvable_pass_accounts_for_every_candidate() {
    if !gpu_or_skip("improvable_pass_accounts_for_every_candidate") {
        return;
    }
    reset_global_tracker();
    let (_temp_dir, parquet_file) = write_fixture_parquet(0.1);

    let result = analyze_all(&make_input(parquet_file)).expect("analyze_all");

    let observed = reconciliations(&result);
    let mut total_considered = 0u32;
    for (reconciliation, breakdown) in observed {
        assert_balanced(reconciliation, breakdown);
        total_considered = total_considered.saturating_add(reconciliation.considered);
    }
    assert!(
        total_considered > 0,
        "the ledger recorded no candidates at all, so the invariant held vacuously — \
         `record_considered` is no longer reached from the real dispatch path (Issue #1802)"
    );
}

/// Strict mode is the CI form of the invariant: with it on, a real pass must
/// still complete, proving the wiring balances under the assertion rather than
/// only under the warn.
#[test]
#[serial]
fn real_pass_survives_strict_mode() {
    if !gpu_or_skip("real_pass_survives_strict_mode") {
        return;
    }
    reset_global_tracker();
    let _strict = StrictModeGuard::new(true);
    assert!(strict_mode(), "the guard must force strict mode on");
    let (_temp_dir, parquet_file) = write_fixture_parquet(0.1);

    let result = analyze_all(&make_input(parquet_file)).expect("analyze_all under strict mode");

    for (reconciliation, breakdown) in reconciliations(&result) {
        assert_balanced(reconciliation, breakdown);
    }
}

// ---------------------------------------------------------------------------
// Guarding the guard — a deliberately un-counted drop.
// ---------------------------------------------------------------------------

/// A drop path that forgets to record its verdict must be caught, and the
/// diagnostic must name the surface and the unaccounted delta.
#[test]
#[serial]
fn uncounted_drop_fails_with_surface_and_delta() {
    let _strict = StrictModeGuard::new(false);
    let ledger = CandidateLedger::new();
    // A batch of 9 candidates is formed; 4 are dropped by a hypothetical bare
    // `continue` that records nothing.
    ledger.record_considered(9);
    ledger.record_accounted(5);

    let mut breakdown = RejectionBreakdown::new();
    let reconciliation = reconcile(SURFACE_NEURON, &ledger, &mut breakdown);

    assert!(!reconciliation.balanced());
    assert_eq!(reconciliation.unaccounted, 4);
    assert_eq!(reconciliation.signed_delta(), -4);

    let message = reconciliation.message();
    assert!(
        message.contains(SURFACE_NEURON),
        "the diagnostic must name the surface: {message}"
    );
    assert!(
        message.contains('4'),
        "the diagnostic must name the unaccounted delta: {message}"
    );

    // Fail loud on a release build too: the residual is visible in the
    // breakdown even where the assertion is compiled out.
    assert_eq!(
        breakdown.counts().get(REJECTION_UNACCOUNTED_DROP),
        Some(&4),
        "the unaccounted residual must surface in the rejection breakdown"
    );
}

/// Under strict mode the same un-counted drop panics, so CI fails the PR that
/// introduces it instead of merely logging.
#[test]
#[serial]
#[cfg(debug_assertions)]
#[should_panic(expected = "candidate reconciliation failed on the synapse surface")]
fn strict_mode_fails_ci_on_uncounted_drop() {
    let _strict = StrictModeGuard::new(true);
    let ledger = CandidateLedger::new();
    ledger.record_considered(3);
    ledger.record_accounted(1);
    let mut breakdown = RejectionBreakdown::new();
    let _ = reconcile(SURFACE_SYNAPSE, &ledger, &mut breakdown);
}

/// The residual reason must be a documented, stable name like every other
/// rejection reason, so downstream tooling can key on it.
#[test]
fn unaccounted_drop_is_a_documented_reason() {
    assert_eq!(REJECTION_UNACCOUNTED_DROP, "unaccounted_drop");
    assert!(
        ALL_REJECTION_REASONS.contains(&REJECTION_UNACCOUNTED_DROP),
        "the residual reason must be registered in ALL_REJECTION_REASONS"
    );
}
