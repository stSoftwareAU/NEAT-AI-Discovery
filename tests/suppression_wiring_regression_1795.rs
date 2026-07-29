//! Production-shaped regression guard for the suppression layer (Issue #1795).
//!
//! The whole #1780 bug class — suppression state that is read but never written
//! — was invisible because every store had thorough unit tests that
//! **constructed the store directly**. A unit test that builds its own subject
//! cannot detect a missing production caller, so `TargetFailureTracker` stayed
//! green while having zero production writers.
//!
//! These tests therefore drive `analyze_all` — the real discovery-pass entry
//! point — and never construct a tracker by hand. If a refactor drops the
//! `advance_global_epoch` call (#1790), the `flush_target_pass_outcomes` call
//! (#1791), the cooldown filter, or reverts `DroughtInputs::target_tracker` to
//! a literal `None`, one of these goes red with a message naming the
//! suppression store that went dead.
//!
//! Scope note: #1792 deleted `CandidateOutcomeCache` and #1793 deleted
//! `ModuleStarvationTracker`, so `TargetFailureTracker` is the only surviving
//! suppression store and the only one guarded here.
//!
//! Isolation: `global_tracker()` is process-global, so every test resets it and
//! runs `#[serial]`.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use neat_ai_discovery::analysis::analyze_all;
use neat_ai_discovery::analysis::discovery_mode::DiscoveryOutcomeLog;
use neat_ai_discovery::analysis::gpu::GpuAnalyzer;
use neat_ai_discovery::analysis::shared::AnalyzeAllResult;
use neat_ai_discovery::analysis::target_failure_tracker::{
    TargetFailureTracker, global_tracker, reset_global_tracker,
};
use neat_ai_discovery::parquet_format::write_records_to_parquet;
use neat_ai_discovery::types::DiscoverRecord;
use neat_ai_discovery::{AnalyzeAllInput, CreatureJson, NeuronJson, SynapseJson};
use serial_test::serial;
use tempfile::TempDir;

/// Name of the suppression store these guards protect. Every failure message
/// carries it so a future regression is diagnosable without re-deriving #1780.
const STORE: &str = "TargetFailureTracker";

// ---------------------------------------------------------------------------
// Global-tracker probes (read-only; production recovers poisoned locks the
// same way).
// ---------------------------------------------------------------------------

fn with_tracker<T>(f: impl FnOnce(&TargetFailureTracker) -> T) -> T {
    let guard = global_tracker()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f(&guard)
}

fn tracked_targets() -> usize {
    with_tracker(TargetFailureTracker::len)
}

fn current_epoch() -> u64 {
    with_tracker(TargetFailureTracker::current_epoch)
}

fn cooldown_threshold() -> u32 {
    with_tracker(TargetFailureTracker::cooldown_consecutive_failures)
}

fn active_cooldowns() -> usize {
    let epoch = current_epoch();
    with_tracker(|t| t.active_cooldown_count(epoch))
}

// ---------------------------------------------------------------------------
// Fixture: a tiny converged creature whose candidates are all rejected, so
// every evaluated target records a pass failure.
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

/// Recorded errors sit at `~1e-7`, the achievable-but-below-floor band the
/// #1740 threshold review confirmed is genuine noise for a converged creature.
/// Every candidate is therefore generated and then rejected at the gain gate,
/// so each focus target is genuinely *evaluated* and records a pass failure —
/// exactly the production shape the guard needs. A larger error (e.g. `0.1`)
/// yields accepted candidates, which reset the streak instead.
const CONVERGED_ERROR_SCALE: f32 = 1e-7;

fn build_records(creature: &CreatureJson, count: usize) -> Vec<DiscoverRecord> {
    let mut records = Vec::new();
    for obs in 0..count as u32 {
        let t = obs as f32 / count as f32;
        for neuron in &creature.neurons {
            let (activation, value, errors) = match neuron.neuron_type.as_str() {
                "input" => ((t * std::f32::consts::TAU).sin(), None, Vec::new()),
                "hidden" => {
                    let act = 0.3 + 0.4 * (t * std::f32::consts::TAU).sin();
                    (
                        act,
                        Some(act),
                        vec![CONVERGED_ERROR_SCALE * (t * 3.0).cos()],
                    )
                }
                "output" => {
                    let error = CONVERGED_ERROR_SCALE * (t * std::f32::consts::PI).cos();
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

/// Write the fixture's recording once; the `TempDir` must outlive every pass.
fn write_fixture_parquet() -> (TempDir, String) {
    let creature = build_creature();
    let records = build_records(&creature, 60);
    let temp_dir = tempfile::tempdir().expect("temp dir");
    let parquet_path = temp_dir.path().join("suppression_wiring.parquet");
    let parquet_file = parquet_path
        .to_str()
        .expect("parquet path is valid UTF-8")
        .to_string();
    write_records_to_parquet(&parquet_file, &records).expect("write parquet");
    (temp_dir, parquet_file)
}

fn make_input(parquet_file: String, outcome_log: Option<DiscoveryOutcomeLog>) -> AnalyzeAllInput {
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
        discovery_outcome_log: outcome_log,
        cost_name: None,
    }
}

/// Total targets dropped by `apply_target_cooldown` across both phases in one
/// pass, read off the production metadata surface rather than a test double.
fn cooldown_skipped(result: &AnalyzeAllResult) -> u32 {
    let synapse = result
        .synapse
        .as_ref()
        .map_or(0, |s| s.metadata.target_cooldown_skipped);
    let neuron = result
        .neuron
        .as_ref()
        .map_or(0, |n| n.metadata.target_cooldown_skipped);
    synapse.saturating_add(neuron)
}

/// Skip on GPU-less hosts: without a GPU no candidate is evaluated, so no
/// target verdict exists to record and the guard has nothing to observe.
fn gpu_or_skip(test: &str) -> bool {
    if GpuAnalyzer::gpu_is_available() {
        return true;
    }
    eprintln!("Skipping {test}: no GPU available");
    false
}

// ---------------------------------------------------------------------------
// Guards
// ---------------------------------------------------------------------------

/// A single discovery pass through `analyze_all` must leave the process-global
/// tracker populated and advance its epoch by exactly one (#1790 + #1791
/// verified through the production path, not the unit path).
#[test]
#[serial]
fn discovery_pass_populates_global_tracker() {
    if !gpu_or_skip("discovery_pass_populates_global_tracker") {
        return;
    }
    reset_global_tracker();
    let (_temp_dir, parquet_file) = write_fixture_parquet();
    let input = make_input(parquet_file, None);

    let epoch_before = current_epoch();
    analyze_all(&input).expect("analyze_all");

    assert_eq!(
        current_epoch(),
        epoch_before + 1,
        "{STORE} went dead: one discovery pass must advance the global epoch by exactly 1 \
         (Issue #1790). A frozen epoch means no cooldown can ever expire; an epoch moving by 2 \
         means the advance leaked into the per-phase preparation layers."
    );
    assert!(
        tracked_targets() > 0,
        "{STORE} went dead: a discovery pass over a rejected-only creature left the global \
         tracker empty, so it is read but never written (Issue #1791). Check that \
         `analyze_all` still calls `flush_target_pass_outcomes` with the merged neuron and \
         synapse `target_pass_outcomes`."
    );
}

/// Repeating the pass past `cooldown_consecutive_failures` must make
/// `apply_target_cooldown` genuinely drop a target — observed via the
/// production metadata counter, not by calling the filter directly.
#[test]
#[serial]
fn cooldown_drops_target_through_real_path() {
    if !gpu_or_skip("cooldown_drops_target_through_real_path") {
        return;
    }
    reset_global_tracker();
    let (_temp_dir, parquet_file) = write_fixture_parquet();
    let input = make_input(parquet_file, None);

    let threshold = cooldown_threshold();
    // One extra pass past the threshold: the streak reaches `threshold` on the
    // final build-up pass, so suppression can only be observed on the pass
    // after it.
    let passes = threshold + 1;
    let mut observed_skips = 0u32;
    for _ in 0..passes {
        let result = analyze_all(&input).expect("analyze_all");
        observed_skips = observed_skips.saturating_add(cooldown_skipped(&result));
    }

    assert!(
        observed_skips > 0,
        "{STORE} went dead: {passes} consecutive rejected-only passes (cooldown threshold \
         {threshold}) dropped no targets at all. Either the per-target failures are no longer \
         flushed to the global tracker (Issue #1791) or `apply_target_cooldown` no longer \
         consults it (Issue #1130)."
    );
    assert!(
        active_cooldowns() > 0,
        "{STORE} went dead: no target is in cooldown after {passes} rejected-only passes, so \
         the suppression layer is inert."
    );
}

/// Past `drought_reset_after_epochs`, the orchestrator's
/// `maybe_perform_drought_reset` must clear a **non-zero** number of cooldown
/// entries. A structurally-zero clear is the #1780 signature: the lever fires
/// against state nothing ever populated.
#[test]
#[serial]
fn drought_reset_clears_nonzero() {
    if !gpu_or_skip("drought_reset_clears_nonzero") {
        return;
    }
    reset_global_tracker();
    let (_temp_dir, parquet_file) = write_fixture_parquet();

    // Build up real cooldown entries through the production path.
    let build_up = make_input(parquet_file.clone(), None);
    for _ in 0..=cooldown_threshold() {
        analyze_all(&build_up).expect("analyze_all");
    }
    let cooled_before = active_cooldowns();
    assert!(
        cooled_before > 0,
        "{STORE} went dead: the build-up passes left nothing in cooldown, so this guard cannot \
         prove the drought reset clears anything (Issue #1791)."
    );

    // A trailing-failure streak long enough to arm the operator escape hatch
    // (`NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS`, armed by default).
    let drought_log = DiscoveryOutcomeLog::from_outcomes(vec![false; 120]);
    let drought_pass = make_input(parquet_file, Some(drought_log));
    analyze_all(&drought_pass).expect("analyze_all");

    let cleared = cooled_before.saturating_sub(active_cooldowns());
    assert!(
        cleared > 0,
        "{STORE} went dead: the drought reset cleared 0 of {cooled_before} active cooldown \
         entries. `maybe_perform_drought_reset` must operate on the live global tracker \
         (Issue #1205), not a snapshot or an empty stand-in."
    );
}

/// `DroughtInputs::target_tracker` must never be a literal `None`.
///
/// This is the behavioural form of the no-permanently-`None` guard: with a live
/// tracker the diagnostic reports a non-zero `target_cooldown_active_count`;
/// with `None` the field is structurally `0`, reading as "nothing is in
/// cooldown" rather than "cooldown is not tracked".
#[test]
#[serial]
fn drought_diagnostic_reports_live_tracker() {
    if !gpu_or_skip("drought_diagnostic_reports_live_tracker") {
        return;
    }
    reset_global_tracker();
    let (_temp_dir, parquet_file) = write_fixture_parquet();

    let build_up = make_input(parquet_file.clone(), None);
    for _ in 0..=cooldown_threshold() {
        analyze_all(&build_up).expect("analyze_all");
    }
    assert!(
        active_cooldowns() > 0,
        "{STORE} went dead: no cooldown entries to report (Issue #1791)."
    );

    // Above the drought log threshold but below the reset threshold, so the
    // diagnostic fires while the entries are still present.
    let drought_log = DiscoveryOutcomeLog::from_outcomes(vec![false; 10]);
    let result = analyze_all(&make_input(parquet_file, Some(drought_log))).expect("analyze_all");

    let diagnostic = result
        .synapse
        .as_ref()
        .and_then(|s| s.metadata.drought_diagnostic.clone())
        .or_else(|| {
            result
                .neuron
                .as_ref()
                .and_then(|n| n.metadata.drought_diagnostic.clone())
        })
        .expect("drought diagnostic must be attached once the streak crosses the threshold");

    assert!(
        diagnostic.target_cooldown_active_count > 0,
        "{STORE} went dead: the drought diagnostic reported 0 active cooldowns while the global \
         tracker holds {}. `DroughtInputs.target_tracker` has regressed to a literal `None` \
         (Issue #1795).",
        active_cooldowns()
    );
}
