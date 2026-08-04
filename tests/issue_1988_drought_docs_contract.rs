//! Issue #1988 — the drought docs must not describe diagnostic surfaces or
//! rejection-reason names the code cannot emit.
//!
//! `docs/DROUGHT_PLAYBOOK.md` is the operator's incident runbook: an operator
//! mid-drought greps the live diagnostic for a field the doc names, matches
//! nothing, and burns time doubting the harness. Every test below first proves
//! what the code actually emits, then asserts the prose agrees.

use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::ALL_REJECTION_REASONS;
use neat_ai_discovery::analysis::discovery_mode::{
    DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS, DEFAULT_LOW_SUCCESS_RATE_THRESHOLD, DiscoveryMode,
    DiscoveryOutcomeLog, decide_mode,
};
use neat_ai_discovery::analysis::drought_diagnostic::{DroughtInputs, emit_drought_diagnostic};
use neat_ai_discovery::analysis::drought_reset::maybe_perform_drought_reset;
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;
use neat_ai_discovery::config::DEFAULT_DROUGHT_RESET_AFTER_EPOCHS;

const PLAYBOOK: &str = include_str!("../docs/DROUGHT_PLAYBOOK.md");
const CONFIGURATION: &str = include_str!("../docs/CONFIGURATION.md");

/// The six fields Issue #1274 proposed and Issue #1937 struck from `FFI_API.md`.
/// `DroughtDiagnostic` has never carried any of them.
const PHANTOM_FIELDS: &[&str] = &[
    "dominantFailedModule",
    "dominantFailedModuleShare",
    "dominantFailedTargetUuid",
    "dominantFailedTargetShare",
    "dominantOperationCount",
    "predictedVsActualGapP50",
];

/// Text of the markdown section introduced by `heading`, up to the next heading
/// of the same or a higher level.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let start = doc
        .find(heading)
        .unwrap_or_else(|| panic!("doc must contain the heading {heading:?}"));
    let level = heading.chars().filter(|c| *c == '#').count();
    let body = &doc[start + heading.len()..];
    body.match_indices("\n#")
        .find(|(idx, _)| body[idx + 1..].chars().take_while(|c| *c == '#').count() <= level)
        .map_or(body, |(idx, _)| &body[..idx])
}

/// The real wire keys of `droughtDiagnostic`, taken from a diagnostic the
/// library actually emitted.
fn emitted_wire_keys() -> Vec<String> {
    let mut breakdown = RejectionBreakdown::default();
    breakdown.record_many("no_eligible_sources", 91);
    let tracker = TargetFailureTracker::with_thresholds(3, 40);
    let inputs = DroughtInputs {
        consecutive_failures: 7,
        rolling_success_rate: 0.0,
        discovery_mode: DiscoveryMode::Conservative,
        target_tracker: Some(&tracker),
        current_epoch: 0,
        target_cooldown_skipped: 3,
        rejection_breakdown: &breakdown,
        candidates_returned: 0,
    };
    let diagnostic =
        emit_drought_diagnostic(&inputs, 5).expect("a streak past the threshold must emit");
    let wire = serde_json::to_value(&diagnostic).expect("DroughtDiagnostic must serialise");
    let object = wire
        .as_object()
        .expect("droughtDiagnostic serialises as a JSON object")
        .clone();
    object.keys().cloned().collect()
}

/// The first backticked token on each row of a markdown table, skipping the
/// header and separator rows.
fn first_column_codes(table_section: &str) -> Vec<&str> {
    table_section
        .lines()
        .filter(|line| line.trim_start().starts_with("| `"))
        .filter_map(|line| {
            let after = line.trim_start().trim_start_matches("| ");
            after
                .strip_prefix('`')
                .and_then(|rest| rest.split('`').next())
        })
        .collect()
}

/// Item 1 — `DroughtDiagnostic` emits exactly nine keys, so the playbook may
/// document exactly those nine and none of the six #1274 phantoms.
#[test]
fn the_playbook_documents_only_the_nine_emitted_diagnostic_fields() {
    let keys = emitted_wire_keys();
    assert_eq!(
        keys.len(),
        9,
        "droughtDiagnostic must carry exactly nine fields, got: {keys:?}"
    );

    for phantom in PHANTOM_FIELDS {
        assert!(
            !keys.iter().any(|k| k == phantom),
            "{phantom} must not be on the wire — the emitted keys are {keys:?}"
        );
        assert!(
            !PLAYBOOK.contains(phantom),
            "the playbook must not document {phantom}: no such field is ever emitted"
        );
    }

    for key in &keys {
        assert!(
            PLAYBOOK.contains(key.as_str()),
            "the playbook must still document the emitted field {key}"
        );
    }
}

/// Item 1 (continued) — the Diagnostic Walkthrough lever table has one row per
/// emitted field, so an operator reading it top-to-bottom sees the real schema.
#[test]
fn the_walkthrough_lever_table_has_one_row_per_emitted_field() {
    let keys = emitted_wire_keys();
    let walkthrough = section(PLAYBOOK, "## Diagnostic Walkthrough");
    let lever_table: Vec<&str> = walkthrough
        .lines()
        .take_while(|line| !line.contains("Common `dominantRejectionReason`"))
        .collect();
    let lever_rows = lever_table.join("\n");
    let documented = first_column_codes(&lever_rows);

    for key in &keys {
        assert!(
            documented.contains(&key.as_str()),
            "the lever table must carry a row for {key}, has: {documented:?}"
        );
    }
    assert_eq!(
        documented.len(),
        keys.len(),
        "the lever table must not carry rows for fields that are never emitted: {documented:?}"
    );
}

/// Item 2 — every rejection-reason name the playbook lists must be a stable
/// name the code can actually put in `dominantRejectionReason`.
#[test]
fn every_documented_rejection_reason_is_a_stable_reason_name() {
    let walkthrough = section(PLAYBOOK, "## Diagnostic Walkthrough");
    let start = walkthrough
        .find("Common `dominantRejectionReason`")
        .expect("the walkthrough must carry the common-reasons table");
    let documented = first_column_codes(&walkthrough[start..]);
    assert!(
        !documented.is_empty(),
        "the common-reasons table must list at least one reason"
    );

    for reason in &documented {
        assert!(
            ALL_REJECTION_REASONS.contains(reason),
            "`{reason}` is not in ALL_REJECTION_REASONS — the playbook names a reason \
             the code cannot emit"
        );
    }
}

/// Item 2 (continued) — the suppression-layer diagram names reasons too, and a
/// diagram is the first thing an operator greps against.
#[test]
fn the_suppression_layer_diagram_names_only_stable_reasons() {
    let layers = section(PLAYBOOK, "## Suppression Layers");
    let node = layers
        .lines()
        .find(|line| line.contains("Post -."))
        .expect("the diagram must carry the post-processing rejection node");

    for token in node.split(&['"', '<', '>', '/', ' '][..]) {
        // Reason names are the lower_snake_case tokens in the node label.
        if token.len() > 3 && token.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
            assert!(
                ALL_REJECTION_REASONS.contains(&token),
                "the diagram names `{token}`, which is not a stable rejection reason"
            );
        }
    }
}

/// Item 2 (continued) — none of the three invented names may survive anywhere
/// in the playbook. `cooldown_skipped` is checked backticked so the legitimate
/// `target_cooldown_skipped` is not caught by the substring.
#[test]
fn the_invented_reason_names_appear_nowhere_in_the_playbook() {
    for invented in ["budget_exceeded", "`cooldown_skipped`", "`duplicate`"] {
        assert!(
            !PLAYBOOK.contains(invented),
            "the playbook must not name {invented} — no such rejection reason exists"
        );
    }
    assert!(
        !PLAYBOOK.contains("redundant_path"),
        "`redundant_path` is a discovery module, not a rejection reason"
    );
}

/// Item 3 — the drought reset's only clearable input is the cooldown tracker,
/// so the CONFIGURATION.md row must not promise a candidate-cache flush.
#[test]
fn the_configuration_reset_row_describes_only_the_cooldown_clearing() {
    let mut tracker = TargetFailureTracker::with_thresholds(1, 10);
    tracker.record_failure("target-1", 0);
    let outcome = maybe_perform_drought_reset(Some(&mut tracker), 50, 50, 0)
        .expect("the reset must fire once the streak crosses the threshold");
    assert_eq!(
        outcome.target_cooldown_cleared, 1,
        "the reset clears cooldown entries and nothing else"
    );

    let row = CONFIGURATION
        .lines()
        .find(|line| line.contains("NEAT_AI_DISCOVERY_DROUGHT_RESET_AFTER_EPOCHS"))
        .expect("CONFIGURATION.md must document the reset lever");
    assert!(
        !row.contains("failed-candidate cache"),
        "the candidate cache went with CandidateOutcomeCache (#1792): {row}"
    );
    assert!(
        row.contains("active target cooldowns"),
        "the row must still describe the cooldown reset it does perform: {row}"
    );
}

/// Item 4 — the extended-drought cooldown relaxation compares the streak
/// against the **compiled** default with `>=`, while the risk-bias revert
/// compares against the env-resolved value with `>`. The two boundaries differ
/// by one epoch, and the playbook must say so.
#[test]
fn the_cooldown_escalation_boundary_is_pinned_to_the_compiled_default() {
    let tracker = TargetFailureTracker::with_thresholds(3, 40);
    let boundary = DEFAULT_CONSERVATIVE_MODE_MAX_EPOCHS;

    // At the boundary streak the cooldown window has already escalated …
    let at_boundary = tracker.effective_cooldown_epochs(DiscoveryMode::Conservative, boundary);
    let below = tracker.effective_cooldown_epochs(DiscoveryMode::Conservative, boundary - 1);
    assert!(
        at_boundary < below,
        "the extended divisor must engage at streak == {boundary} (>=): \
         below={below} at_boundary={at_boundary}"
    );

    // … while the risk bias is still Conservative at that same streak (>).
    let log = DiscoveryOutcomeLog::from_outcomes(vec![false; boundary as usize]);
    assert_eq!(
        decide_mode(&log, DEFAULT_LOW_SUCCESS_RATE_THRESHOLD, boundary),
        DiscoveryMode::Conservative,
        "the bias revert uses `>`, so streak == {boundary} is still Conservative"
    );

    let notes = section(PLAYBOOK, "## Adaptive Responses");
    assert!(
        notes.contains("compiled default"),
        "the playbook must say the cooldown escalation is pinned to the compiled default"
    );
    assert!(
        notes.contains("≥"),
        "the playbook must use ≥ for the cooldown-escalation boundary"
    );
}

/// Item 4 (continued) — a 30-epoch drought never reaches the default 50-epoch
/// reset, so the worked example must not claim the escape hatch fires.
#[test]
fn the_worked_example_does_not_claim_the_default_reset_fires() {
    let mut tracker = TargetFailureTracker::with_thresholds(1, 10);
    tracker.record_failure("target-1", 0);
    assert_eq!(
        DEFAULT_DROUGHT_RESET_AFTER_EPOCHS, 50,
        "the worked example is written against the default reset threshold"
    );
    assert!(
        maybe_perform_drought_reset(
            Some(&mut tracker),
            30,
            DEFAULT_DROUGHT_RESET_AFTER_EPOCHS,
            0
        )
        .is_none(),
        "a 30-epoch streak cannot reach the default 50-epoch reset"
    );

    let example = section(PLAYBOOK, "## Worked Example");
    assert!(
        !example.contains("the one-shot operator reset fires here"),
        "the example must not claim a reset that cannot fire at the stated defaults"
    );
    assert!(
        !example.contains("decide whether to enable"),
        "the escape hatch is armed by default (#1422) — the example must not say to enable it"
    );
    assert!(
        example.contains("50"),
        "the example must name the 50-epoch threshold it does not reach"
    );
}
