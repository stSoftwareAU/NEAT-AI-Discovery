//! Tests for diagnostics and rejection tracking functionality.
//!
//! Tests cover:
//! - `TargetDiagnostics` preference and candidate selection
//! - `NeuronDiagnostics` load failure tracking and reporting
//! - Focus target filtering for threshold activations
//! - Hidden/input/constant neuron filtering

use super::common::*;

#[test]
fn diagnostics_prefers_higher_expected_improvement() {
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
    diagnostics.set_target_record_count("output-0", 1_500);
    diagnostics.record_candidate_attempt("output-0", false);
    diagnostics.record_no_samples("output-0", "input-0", 0);

    diagnostics.record_candidate_attempt("output-0", true);
    diagnostics.record_below_threshold(
        "output-0",
        "hidden-1",
        ThresholdContext {
            sample_count: 42,
            expected_improvement: 0.05,
            threshold: 0.1,
            improved_count: 30,
            worsened_count: 12,
            weight: -0.25,
        },
    );

    let entry = diagnostics
        .entry_for("output-0")
        .expect("diagnostics entry should exist");
    let reason = entry.best_rejection.as_ref().map(|detail| detail.reason);
    assert!(
        matches!(reason, Some(RejectionReason::BelowThreshold)),
        "Expected below-threshold reason to persist when it has the highest score"
    );
}

#[test]
fn diagnostics_marks_candidate_selection() {
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
    diagnostics.mark_candidate_selected("output-0");
    let entry = diagnostics
        .entry_for("output-0")
        .expect("diagnostics entry should exist");
    assert!(
        entry.had_candidate,
        "Entry should record that a candidate was selected"
    );
}

#[test]
fn neuron_diagnostics_tracks_load_failures() {
    // Test that when eligible sources exist but all fail to load, we report
    // NoSamples rather than NoEligibleSources
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
    diagnostics.set_target_record_count("output-0", 100);
    diagnostics.set_total_eligible_sources("output-0", 10); // 10 eligible sources exist
    // All 10 sources fail to load
    for _ in 0..10 {
        diagnostics.record_load_failure("output-0");
    }
    // No record_candidate_attempt calls (because all failed to load)

    let summaries = diagnostics.no_candidate_summaries();
    assert_eq!(summaries.len(), 1);
    let summary = &summaries[0];

    // Should NOT report "no eligible sources" - sources existed but failed to load
    assert!(
        !matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
        "Should not report NoEligibleSources when sources existed but failed to load"
    );
    // Should report NoSamples (as a catchall for sources existing but not being usable)
    assert!(
        matches!(summary.reason, NeuronNoCandidateReason::NoSamples),
        "Should report NoSamples when eligible sources exist but none were evaluated"
    );

    // Verify entry tracking
    let entry = diagnostics.entry_for("output-0").unwrap();
    assert_eq!(entry.total_eligible_sources, 10);
    assert_eq!(entry.record_load_failures, 10);
    assert_eq!(entry.evaluated_sources, 0);
}

#[test]
fn neuron_diagnostics_reports_genuine_no_eligible_sources() {
    // Test that when there are genuinely no eligible sources, we correctly report that
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0"]);
    diagnostics.set_target_record_count("output-0", 100);
    diagnostics.set_total_eligible_sources("output-0", 0); // No eligible sources

    let summaries = diagnostics.no_candidate_summaries();
    assert_eq!(summaries.len(), 1);
    let summary = &summaries[0];

    // Should correctly report no eligible sources
    assert!(
        matches!(summary.reason, NeuronNoCandidateReason::NoEligibleSources),
        "Should report NoEligibleSources when genuinely no sources exist"
    );

    // Verify entry tracking
    let entry = diagnostics.entry_for("output-0").unwrap();
    assert_eq!(entry.total_eligible_sources, 0);
    assert_eq!(entry.record_load_failures, 0);
    assert_eq!(entry.evaluated_sources, 0);
}

#[test]
fn target_diagnostics_reports_no_samples_reason() {
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = TargetDiagnostics::new_for_tests(&["output-0"]);
    diagnostics.set_target_record_count("output-0", 25);
    diagnostics.record_candidate_attempt("output-0", false);
    diagnostics.record_no_samples("output-0", "input-0", 8);

    let summaries = diagnostics.no_candidate_summaries();
    assert_eq!(
        summaries.len(),
        1,
        "Expected a single diagnostic summary for target without candidates"
    );

    let summary = &summaries[0];
    assert_eq!(
        summary.target_uuid, "output-0",
        "Target UUID should be preserved in summary"
    );
    assert!(
        matches!(summary.reason, SynapseNoCandidateReason::NoSamples),
        "Expected no-samples reason"
    );
}

#[test]
fn test_filter_focus_targets_tracks_threshold_targets_for_hidden_and_unknown_when_allowed() {
    // Regression coverage (29-Dec-2025): when output/hidden handling was split, the
    // STEP/BIPOLAR tracking was accidentally only applied to output targets.
    let focus = [
        "hidden-step".to_string(),
        "output-tanh".to_string(),
        "unknown-bipolar".to_string(),
    ];
    let unique_focus: Vec<&String> = focus.iter().collect();

    let mut neuron_type_map: HashMap<String, String> = HashMap::new();
    neuron_type_map.insert("hidden-step".to_string(), "hidden".to_string());
    neuron_type_map.insert("output-tanh".to_string(), "output".to_string());
    neuron_type_map.insert("unknown-bipolar".to_string(), "mystery".to_string());

    let mut neuron_squash_map: HashMap<String, String> = HashMap::new();
    neuron_squash_map.insert("hidden-step".to_string(), "STEP".to_string());
    neuron_squash_map.insert("output-tanh".to_string(), "TANH".to_string());
    neuron_squash_map.insert("unknown-bipolar".to_string(), "BIPOLAR".to_string());

    let result = filter_focus_targets_for_neuron_analysis(
        &unique_focus,
        &neuron_type_map,
        &neuron_squash_map,
        false,
    );

    assert!(
        result.focus_order.contains(&"hidden-step".to_string()),
        "hidden-step should be included when output-only mode is disabled"
    );
    assert!(
        result.focus_order.contains(&"unknown-bipolar".to_string()),
        "unknown-bipolar should be included when output-only mode is disabled (treated as hidden)"
    );
    assert!(
        result
            .threshold_targets
            .contains(&"hidden-step".to_string()),
        "hidden-step (STEP) should be tracked as a threshold target"
    );
    assert!(
        result
            .threshold_targets
            .contains(&"unknown-bipolar".to_string()),
        "unknown-bipolar (BIPOLAR) should be tracked as a threshold target"
    );
    assert!(
        !result
            .threshold_targets
            .contains(&"output-tanh".to_string()),
        "output-tanh (TANH) should not be tracked as a threshold target"
    );
}

#[test]
fn test_filter_focus_targets_respects_output_only_mode_for_hidden_and_unknown() {
    let focus = [
        "hidden-relu".to_string(),
        "output-tanh".to_string(),
        "unknown-identity".to_string(),
    ];
    let unique_focus: Vec<&String> = focus.iter().collect();

    let mut neuron_type_map: HashMap<String, String> = HashMap::new();
    neuron_type_map.insert("hidden-relu".to_string(), "hidden".to_string());
    neuron_type_map.insert("output-tanh".to_string(), "output".to_string());
    neuron_type_map.insert("unknown-identity".to_string(), "mystery".to_string());

    let mut neuron_squash_map: HashMap<String, String> = HashMap::new();
    neuron_squash_map.insert("hidden-relu".to_string(), "RELU".to_string());
    neuron_squash_map.insert("output-tanh".to_string(), "TANH".to_string());
    neuron_squash_map.insert("unknown-identity".to_string(), "IDENTITY".to_string());

    // With output-only mode enabled
    let result = filter_focus_targets_for_neuron_analysis(
        &unique_focus,
        &neuron_type_map,
        &neuron_squash_map,
        true, // output_only = true
    );

    assert!(
        result.focus_order.contains(&"output-tanh".to_string()),
        "output-tanh should be included in output-only mode"
    );
    assert!(
        !result.focus_order.contains(&"hidden-relu".to_string()),
        "hidden-relu should be excluded in output-only mode"
    );
    assert!(
        !result.focus_order.contains(&"unknown-identity".to_string()),
        "unknown-identity should be excluded in output-only mode"
    );
    assert!(
        result.skipped_hidden.contains(&"hidden-relu".to_string()),
        "hidden-relu should be in skipped_hidden"
    );
    assert!(
        result
            .skipped_hidden
            .contains(&"unknown-identity".to_string()),
        "unknown-identity should be in skipped_hidden (treated as hidden)"
    );
}

#[test]
fn neuron_diagnostics_reports_hidden_neuron_filtered_in_mixed_focus_list() {
    // Test that when a focus list contains BOTH output AND hidden neurons,
    // the hidden neurons get HiddenNeuronFiltered reason (not NoEligibleSources).
    //
    // Bug scenario: When focus_order is NOT empty (some output neurons exist),
    // the skipped_hidden neurons were never merged into diagnostics, so they
    // appeared with misleading reasons like NoEligibleSources.
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "hidden-1"]);

    // Mark hidden-1 as filtered (this is what should happen in normal flow)
    diagnostics.mark_hidden_filtered("hidden-1");

    // Simulate output-0 being processed normally but finding no candidate
    diagnostics.set_target_record_count("output-0", 100);
    diagnostics.set_total_eligible_sources("output-0", 5);
    diagnostics.record_candidate_attempt("output-0", false);

    let summaries = diagnostics.no_candidate_summaries();

    // Both should have summaries
    assert_eq!(
        summaries.len(),
        2,
        "Expected 2 summaries (one for output, one for hidden)"
    );

    // Find the hidden neuron summary
    let hidden_summary = summaries
        .iter()
        .find(|s| s.target_uuid == "hidden-1")
        .expect("Should have summary for hidden-1");

    // The hidden neuron should have HiddenNeuronFiltered reason
    assert!(
        matches!(
            hidden_summary.reason,
            NeuronNoCandidateReason::HiddenNeuronFiltered
        ),
        "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
        hidden_summary.reason
    );
}

#[test]
fn neuron_diagnostics_reports_input_neuron_filtered_not_hidden() {
    // Test that when an input neuron is in the focus list, it gets
    // InputNeuronFiltered reason (not HiddenNeuronFiltered).
    //
    // Bug scenario: The neuron_type_map was only built from creature.neurons
    // and didn't include input neurons. When an input neuron (e.g. "input-0")
    // was in the focus list, neuron_type_map.get() returned None, and
    // `neuron_type != Some("output")` evaluated to true. The input neuron
    // was incorrectly added to skipped_hidden and reported with
    // HiddenNeuronFiltered reason.
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "input-1", "hidden-2"]);

    // Mark input-1 as filtered because it's an input neuron
    diagnostics.mark_input_filtered("input-1");

    // Mark hidden-2 as filtered because it's a hidden neuron
    diagnostics.mark_hidden_filtered("hidden-2");

    // Simulate output-0 being processed normally but finding no candidate
    diagnostics.set_target_record_count("output-0", 100);
    diagnostics.set_total_eligible_sources("output-0", 5);
    diagnostics.record_candidate_attempt("output-0", false);

    let summaries = diagnostics.no_candidate_summaries();

    // All three should have summaries
    assert_eq!(
        summaries.len(),
        3,
        "Expected 3 summaries (one for output, one for input, one for hidden)"
    );

    // Find the input neuron summary - should have InputNeuronFiltered reason
    let input_summary = summaries
        .iter()
        .find(|s| s.target_uuid == "input-1")
        .expect("Should have summary for input-1");
    assert!(
        matches!(
            input_summary.reason,
            NeuronNoCandidateReason::InputNeuronFiltered
        ),
        "Input neuron should report InputNeuronFiltered, not {:?}",
        input_summary.reason
    );

    // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
    let hidden_summary = summaries
        .iter()
        .find(|s| s.target_uuid == "hidden-2")
        .expect("Should have summary for hidden-2");
    assert!(
        matches!(
            hidden_summary.reason,
            NeuronNoCandidateReason::HiddenNeuronFiltered
        ),
        "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
        hidden_summary.reason
    );
}

#[test]
fn neuron_diagnostics_reports_constant_neuron_filtered_not_hidden() {
    // Test that when a constant neuron is in the focus list, it gets
    // ConstantNeuronFiltered reason (not HiddenNeuronFiltered).
    //
    // Bug scenario (v0.1.124): Constant neurons were pushed to skipped_hidden
    // but reported with HiddenNeuronFiltered reason. This is semantically
    // incorrect - constant neurons don't receive inputs because they always
    // output a fixed value, which is different from hidden neurons whose
    // backpropagated errors don't reliably predict output error.
    // Issue #216: Methods now take &self, not &mut self
    let diagnostics = NeuronDiagnostics::new_for_tests(&["output-0", "constant-1", "hidden-2"]);

    // Mark constant-1 as filtered because it's a constant neuron
    diagnostics.mark_constant_filtered("constant-1");

    // Mark hidden-2 as filtered because it's a hidden neuron
    diagnostics.mark_hidden_filtered("hidden-2");

    // Simulate output-0 being processed normally but finding no candidate
    diagnostics.set_target_record_count("output-0", 100);
    diagnostics.set_total_eligible_sources("output-0", 5);
    diagnostics.record_candidate_attempt("output-0", false);

    let summaries = diagnostics.no_candidate_summaries();

    // All three should have summaries
    assert_eq!(
        summaries.len(),
        3,
        "Expected 3 summaries (one for output, one for constant, one for hidden)"
    );

    // Find the constant neuron summary - should have ConstantNeuronFiltered reason
    let constant_summary = summaries
        .iter()
        .find(|s| s.target_uuid == "constant-1")
        .expect("Should have summary for constant-1");
    assert!(
        matches!(
            constant_summary.reason,
            NeuronNoCandidateReason::ConstantNeuronFiltered
        ),
        "Constant neuron should report ConstantNeuronFiltered, not {:?}",
        constant_summary.reason
    );

    // Find the hidden neuron summary - should still have HiddenNeuronFiltered reason
    let hidden_summary = summaries
        .iter()
        .find(|s| s.target_uuid == "hidden-2")
        .expect("Should have summary for hidden-2");
    assert!(
        matches!(
            hidden_summary.reason,
            NeuronNoCandidateReason::HiddenNeuronFiltered
        ),
        "Hidden neuron should report HiddenNeuronFiltered, not {:?}",
        hidden_summary.reason
    );
}
