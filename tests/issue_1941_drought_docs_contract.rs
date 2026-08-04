//! Issue #1941 — the #1777–#1818 drought campaign's own documents must not
//! state superseded facts in the present tense.
//!
//! `docs/DROUGHT_PLAYBOOK.md` is an operator playbook read mid-incident, and
//! `docs/analysis/candidate-rate-diagnosis-1777.md` is the root-cause reference
//! five campaign PR summaries cite by name. Each test below first proves the
//! *current* behaviour with the real code, then asserts the prose agrees with
//! what that behaviour just demonstrated.

use neat_ai_discovery::analysis::candidate_starvation::{
    StarvationClass, StarvationConfig, classify, signals_from_breakdown,
};
use neat_ai_discovery::analysis::diagnostics::RejectionBreakdown;
use neat_ai_discovery::analysis::diagnostics::rejection_reasons::{
    REJECTION_BELOW_THRESHOLD, REJECTION_DUPLICATE_OF_FAILURE_CACHE,
};
use neat_ai_discovery::analysis::discovery_mode::DiscoveryMode;
use neat_ai_discovery::analysis::drought_reset::maybe_perform_drought_reset;
use neat_ai_discovery::analysis::target_failure_tracker::TargetFailureTracker;
use neat_ai_discovery::config::DroughtMitigationConfig;

const PLAYBOOK: &str = include_str!("../docs/DROUGHT_PLAYBOOK.md");
const DIAGNOSIS_1777: &str = include_str!("../docs/analysis/candidate-rate-diagnosis-1777.md");
const GATING_1739: &str = include_str!("../docs/analysis/candidate-generation-gating-1739.md");
const ARCHIVE_README: &str = include_str!("../docs/archive/README.md");

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

/// Everything before the first `##` heading — the playbook's opening claim.
fn intro(doc: &str) -> &str {
    doc.find("\n## ").map_or(doc, |idx| &doc[..idx])
}

/// A tracker with one target parked in cooldown at epoch 0.
fn tracker_with_one_cooldown() -> TargetFailureTracker {
    let mut tracker = TargetFailureTracker::with_thresholds(1, 10);
    tracker.record_failure("target-1", 0);
    assert!(
        tracker.is_in_cooldown("target-1", 0),
        "fixture must park the target in cooldown"
    );
    tracker
}

/// Item 1 — the candidate outcome cache was deleted by #1792, so the intro must
/// not open with four suppression layers including it. Proved by the reset's own
/// signature: the only clearable input left is the cooldown tracker.
#[test]
fn the_intro_counts_only_the_three_live_suppression_layers() {
    let mut tracker = tracker_with_one_cooldown();
    let outcome = maybe_perform_drought_reset(Some(&mut tracker), 50, 50, 0)
        .expect("reset must fire once the streak crosses the threshold");
    assert_eq!(
        outcome.target_cooldown_cleared, 1,
        "the reset's one clearable input is the cooldown tracker"
    );

    let intro = intro(PLAYBOOK);
    assert!(
        !intro.contains("four"),
        "the intro must not claim four suppression layers — the cache was deleted by #1792: {intro}"
    );
    assert!(
        !intro.to_lowercase().contains("candidate outcome cache"),
        "the intro must not list the deleted candidate outcome cache as a live layer: {intro}"
    );
    assert!(
        intro.contains("three"),
        "the intro must state three suppression layers: {intro}"
    );

    // The body already got this right — the two must not contradict.
    assert!(
        section(PLAYBOOK, "## Suppression Layers").contains("Three mechanisms"),
        "the Suppression Layers section must still say three mechanisms"
    );
}

/// Item 2 — `clear_failed_entries` no longer exists, so no prose may promise the
/// escape hatch clears cache entries. Pinned to the outcome type, which reports
/// cleared cooldowns and nothing else.
#[test]
fn the_escape_hatch_clears_cooldowns_only_and_the_prose_says_so() {
    let mut tracker = tracker_with_one_cooldown();
    let outcome = maybe_perform_drought_reset(Some(&mut tracker), 50, 50, 0)
        .expect("reset must fire once the streak crosses the threshold");
    assert!(!outcome.is_noop(), "the reset cleared a real cooldown");
    assert!(
        !tracker.is_in_cooldown("target-1", 0),
        "the target must leave cooldown after the reset"
    );

    let lower = PLAYBOOK.to_lowercase();
    assert!(
        !lower.contains("cache entries and active cooldowns"),
        "the playbook must not claim the reset clears failed cache entries"
    );
    assert!(
        !lower.contains("cache + cooldown reset"),
        "the Operator Levers table must not describe a cache + cooldown reset"
    );
}

/// Item 3 — adaptive target-cooldown relaxation (Issue #1204) has shipped: the
/// effective window and trigger both move with mode and drought state.
#[test]
fn adaptive_cooldown_relaxation_has_shipped_and_the_playbook_agrees() {
    let tracker = TargetFailureTracker::with_thresholds(3, 40);
    let base = tracker.effective_cooldown_epochs(DiscoveryMode::Normal, 0);
    let conservative = tracker.effective_cooldown_epochs(DiscoveryMode::Conservative, 0);
    let extended = tracker.effective_cooldown_epochs(DiscoveryMode::Normal, 1_000);
    assert!(
        conservative < base && extended < conservative,
        "the window must relax through Conservative and Extended Drought: \
         base={base} conservative={conservative} extended={extended}"
    );

    let base_trigger = tracker.effective_consecutive_failures(DiscoveryMode::Normal, 0);
    let relaxed_trigger = tracker.effective_consecutive_failures(DiscoveryMode::Conservative, 0);
    assert!(
        relaxed_trigger > base_trigger,
        "the consecutive-failure trigger must relax in Conservative mode"
    );

    let lower = PLAYBOOK.to_lowercase();
    assert!(
        !lower.contains("not yet shipped"),
        "Issue #1204 has shipped — the playbook must not say otherwise"
    );
    assert!(
        !lower.contains("thresholds remain\n  static")
            && !lower.contains("thresholds remain static"),
        "the playbook must not claim the cooldown thresholds are static"
    );
}

/// Item 3 (continued) — the two divisor knobs are operator levers, so they
/// belong in the Operator Levers table.
#[test]
fn the_operator_levers_table_lists_the_relaxation_divisors() {
    let levers = section(PLAYBOOK, "## Operator Levers");
    for var in [
        "NEAT_AI_DISCOVERY_COOLDOWN_CONSERVATIVE_DIVISOR",
        "NEAT_AI_DISCOVERY_COOLDOWN_EXTENDED_DROUGHT_DIVISOR",
    ] {
        assert!(
            levers.contains(var),
            "the Operator Levers table must carry {var}"
        );
    }
}

/// Item 4 — every field the startup line emits must appear in the quoted block.
#[test]
fn the_quoted_startup_line_lists_every_emitted_lever() {
    let cfg = DroughtMitigationConfig::from_env();
    // Reading the field proves it exists and is emitted alongside the rest.
    assert!(
        cfg.remove_neuron_drought_factor.is_finite(),
        "the drought-mitigation config must carry a finite remove-neuron factor"
    );

    let block = section(PLAYBOOK, "## Effective config at startup");
    for field in [
        "drought_reset_after_epochs",
        "drought_log_threshold",
        "low_success_rate_threshold",
        "conservative_mode_max_epochs",
        "conservative_gain_multiplier",
        "target_cooldown_failures",
        "target_cooldown_epochs",
        "drought_alarm_epochs",
        "remove_neuron_drought_factor",
    ] {
        assert!(
            block.contains(field),
            "the quoted startup config block must include {field}"
        );
    }
}

/// The 1777 diagnosis is a historical study, so its superseded claims must carry
/// the same "resolved / fixed by" markers it already uses elsewhere — starting
/// with the headline claim the drought reset disproves.
#[test]
fn the_1777_diagnosis_annotates_its_superseded_root_cause_b() {
    let mut tracker = tracker_with_one_cooldown();
    let outcome = maybe_perform_drought_reset(Some(&mut tracker), 50, 50, 0)
        .expect("reset must fire once the streak crosses the threshold");
    assert_eq!(
        outcome.target_cooldown_cleared, 1,
        "the drought reset clears real cooldowns — the B1 claim is false today"
    );

    for (heading, issue) in [
        ("### B1 — ", "#1792"),
        ("### B2 — ", "#1781"),
        ("### B3 — ", "#1800"),
    ] {
        let body = section(DIAGNOSIS_1777, heading);
        assert!(
            body.contains("Superseded"),
            "{heading} must carry a Superseded annotation"
        );
        assert!(
            body.contains(issue),
            "{heading} must name the issue that superseded it ({issue})"
        );
    }
}

/// The 1739 gating study's "HOLD" verdict was superseded by #1800: folding
/// failure-cache suppression into the classifier input flips the same pass to
/// `CandidateStarved`.
#[test]
fn the_1739_gating_study_is_dated_and_marked_superseded() {
    let cfg = StarvationConfig::default();

    let mut gate_only = RejectionBreakdown::default();
    gate_only.record_many(REJECTION_BELOW_THRESHOLD, 4);
    let before = classify(&signals_from_breakdown(&gate_only, 0), &cfg);
    assert_eq!(
        before,
        StarvationClass::ProposalRichOverRejected,
        "without the #1800 fold the pass reads as over-rejected"
    );

    let mut folded = gate_only.clone();
    folded.record_many(REJECTION_DUPLICATE_OF_FAILURE_CACHE, 9);
    let after = classify(&signals_from_breakdown(&folded, 0), &cfg);
    assert_eq!(
        after,
        StarvationClass::CandidateStarved,
        "with suppression folded in (#1800) the same pass is candidate-starved"
    );
    // Sanity: the fold is what moved the verdict, not the accepted count.
    assert_eq!(
        signals_from_breakdown(&folded, 0).upstream_rejections,
        9,
        "failure-cache suppression counts as an upstream drop"
    );
    assert_ne!(before, after, "the #1800 fold changed the verdict");

    assert!(
        GATING_1739.contains("as at"),
        "the study must carry a dated 'as at' header"
    );
    assert!(
        GATING_1739.contains("Superseded") && GATING_1739.contains("#1800"),
        "the study must carry a status note pointing at #1800"
    );
}

/// Prevention — `docs/archive/README.md` must document what goes where, naming
/// every documentation tier that actually exists on disk.
#[test]
fn the_archive_readme_documents_every_documentation_tier() {
    for dir in ["docs/analysis", "docs/archive/pr-summaries"] {
        assert!(
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/"))
                .join(dir)
                .is_dir(),
            "{dir} must exist for the README to describe it"
        );
    }

    for tier in ["docs/analysis/", "docs/archive/pr-summaries/", "docs/*.md"] {
        assert!(
            ARCHIVE_README.contains(tier),
            "the archive README must say what goes in {tier}"
        );
    }

    // The transient tier's obligation must agree with the canonical
    // fold-then-delete retention rule, not contradict it (Issue #1991).
    assert!(
        !ARCHIVE_README.contains("leave the summary alone"),
        "the transient tier must not tell a reader to keep a folded summary — the retention \
         rule is fold, then delete (Issue #1991)"
    );
}

/// Guard rail — no live section of the playbook may present a deleted component
/// as a current mechanism. Historical notes are explicitly framed as such.
#[test]
fn the_playbook_never_presents_deleted_components_as_live() {
    let disabled = section(
        PLAYBOOK,
        "## Environmentally-disabled passes vs genuine exhaustion (Issue #1421)",
    );
    assert!(
        !disabled.contains("ModuleStarvationTracker"),
        "ModuleStarvationTracker was deleted by #1793 — it cannot be cited as a live helper"
    );

    // `CandidateOutcomeCache` survives only inside the #1792 historical note.
    let mentions = PLAYBOOK.matches("CandidateOutcomeCache").count();
    let historical = section(PLAYBOOK, "## Suppression Layers")
        .matches("CandidateOutcomeCache")
        .count();
    assert_eq!(
        mentions, historical,
        "CandidateOutcomeCache may only appear in the #1792 historical note"
    );

    // The walkthrough must not send an operator to a cache that no longer exists.
    let walkthrough = section(PLAYBOOK, "## Diagnostic Walkthrough");
    assert!(
        !walkthrough.contains("the cache and cooldown ate them"),
        "the walkthrough must not blame a deleted cache"
    );
}
