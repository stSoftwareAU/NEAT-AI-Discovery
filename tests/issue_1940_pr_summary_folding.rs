//! Issue #1940 — fold five unabsorbed PR-summary learnings into the live docs,
//! then delete the folded summaries.
//!
//! Continues the #1682 fold-then-delete retention rule for the ≥ 1780 archive.
//! Five durable learnings were cited by later PRs while being written down
//! nowhere a reader would look. These tests lock each one into its target doc so
//! it cannot be lost again, and assert the folded summaries were deleted only
//! after their learnings landed (capture is the precondition for deletion):
//!
//!   1. Dead-wiring test doctrine (#1795/#1806/#1815) → `CONTRIBUTING.md`.
//!   2. Vacuously-passing fixtures (#1799) → `CONTRIBUTING.md`.
//!   3. The "dead levers" deletion rule (#1792/#1793/#1818) → `AGENTS.md`.
//!   4. The Mermaid `;` statement-separator trap (#1817) → `AGENTS.md`.
//!   5. Poisoned-mutex recovery for counter-only state (#1875) →
//!      `docs/DROUGHT_PLAYBOOK.md`.

use std::path::Path;

const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");
const AGENTS: &str = include_str!("../AGENTS.md");
const PLAYBOOK: &str = include_str!("../docs/DROUGHT_PLAYBOOK.md");

/// Collapse the doc's hard line wraps so a wrapped sentence still matches.
fn unwrapped(doc: &str) -> String {
    doc.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ============================================================================
// 1. Dead-wiring test doctrine → CONTRIBUTING.md § Testing Guidelines.
// ============================================================================

#[test]
fn contributing_records_the_dead_wiring_diagnosis() {
    assert!(
        unwrapped(CONTRIBUTING).contains("cannot detect a missing production caller"),
        "CONTRIBUTING.md must carry the root diagnosis of the #1780 bug class (Issue #1940)"
    );
    for issue in ["#1795", "#1806", "#1815"] {
        assert!(
            CONTRIBUTING.contains(issue),
            "CONTRIBUTING.md must attribute the dead-wiring doctrine to {issue} (Issue #1940)"
        );
    }
}

#[test]
fn contributing_names_the_1806_convention() {
    // Cited by pr-summary-1810 as "the #1806 convention" while documented nowhere.
    assert!(
        unwrapped(CONTRIBUTING).contains("#1806 convention"),
        "CONTRIBUTING.md must define the #1806 convention by name (Issue #1940)"
    );
    assert!(
        unwrapped(CONTRIBUTING).contains("shipped entry point")
            || CONTRIBUTING.contains("Shipped Entry Point"),
        "CONTRIBUTING.md must require guards to drive the shipped entry point (Issue #1940)"
    );
}

#[test]
fn contributing_requires_assertions_on_the_ffi_response_shape() {
    assert!(
        unwrapped(CONTRIBUTING).contains("FFI response shape"),
        "CONTRIBUTING.md must require wiring guards to assert on the FFI response shape \
         (Issue #1940)"
    );
    assert!(
        CONTRIBUTING.contains("rejectionBreakdown"),
        "CONTRIBUTING.md must name a serialised response key as the assertion target \
         (Issue #1940)"
    );
}

// ============================================================================
// 2. Vacuously-passing fixtures → CONTRIBUTING.md § Testing Guidelines.
// ============================================================================

#[test]
fn contributing_records_the_vacuous_assertion_trap() {
    assert!(
        CONTRIBUTING.contains("#1799"),
        "CONTRIBUTING.md must attribute the vacuous-fixture trap to #1799 (Issue #1940)"
    );
    let lower = CONTRIBUTING.to_lowercase();
    assert!(
        lower.contains("vacuous"),
        "CONTRIBUTING.md must name vacuous passing as the failure mode (Issue #1940)"
    );
    assert!(
        unwrapped(&lower).contains("whether or not the path"),
        "CONTRIBUTING.md must state that an either-way assertion is not coverage (Issue #1940)"
    );
}

#[test]
fn contributing_prescribes_a_positive_precondition() {
    assert!(
        CONTRIBUTING.contains("considered > 0"),
        "CONTRIBUTING.md must give the `considered > 0` positive-precondition remedy \
         (Issue #1940)"
    );
    assert!(
        CONTRIBUTING.contains("#1271"),
        "CONTRIBUTING.md must name the per-target cap (#1271) that shrank the fixture \
         (Issue #1940)"
    );
}

// ============================================================================
// 3. The "dead levers" deletion rule → AGENTS.md.
// ============================================================================

#[test]
fn agents_records_the_dead_lever_rule() {
    assert!(
        unwrapped(AGENTS).contains("silently does nothing is worse than no lever"),
        "AGENTS.md must carry the dead-lever one-liner from #1793 (Issue #1940)"
    );
    for issue in ["#1792", "#1793", "#1818"] {
        assert!(
            AGENTS.contains(issue),
            "AGENTS.md must attribute the dead-lever rule to {issue} (Issue #1940)"
        );
    }
}

#[test]
fn agents_requires_the_config_surface_to_die_with_the_component() {
    assert!(
        unwrapped(AGENTS).contains("concrete writer can be"),
        "AGENTS.md must require a concrete writer before keeping a component (Issue #1940)"
    );
    assert!(
        unwrapped(AGENTS).contains("config surface in the same change"),
        "AGENTS.md must require deleting the config surface in the same change (Issue #1940)"
    );
}

#[test]
fn agents_records_why_no_writer_can_exist() {
    // The negative result: the only inbound per-candidate history is failures-only.
    assert!(
        AGENTS.contains("failureCache"),
        "AGENTS.md must name failureCache as the only inbound per-candidate history \
         (Issue #1940)"
    );
    assert!(
        AGENTS.contains("source_uuid"),
        "AGENTS.md must record that failureCache carries no source_uuid (Issue #1940)"
    );
    assert!(
        AGENTS.contains("candidate_starvation.rs"),
        "AGENTS.md must exempt the live candidate_starvation component (Issue #1940)"
    );
}

// ============================================================================
// 4. The Mermaid `;` trap → AGENTS.md.
// ============================================================================

#[test]
fn agents_records_the_mermaid_semicolon_trap() {
    assert!(
        AGENTS.contains("#1817"),
        "AGENTS.md must attribute the Mermaid `;` trap to #1817 (Issue #1940)"
    );
    assert!(
        unwrapped(AGENTS).contains("statement separator"),
        "AGENTS.md must explain that Mermaid parses `;` as a statement separator (Issue #1940)"
    );
    assert!(
        AGENTS.contains("mermaid_validator.ts"),
        "AGENTS.md must record that the enforcing gate lives outside this repo (Issue #1940)"
    );
}

// ============================================================================
// 5. Poisoned-mutex recovery → docs/DROUGHT_PLAYBOOK.md.
// ============================================================================

#[test]
fn playbook_records_the_poisoned_mutex_convention() {
    assert!(
        PLAYBOOK.contains("#1875"),
        "DROUGHT_PLAYBOOK.md must attribute the poison-recovery convention to #1875 \
         (Issue #1940)"
    );
    assert!(
        PLAYBOOK.contains("PoisonError::into_inner"),
        "DROUGHT_PLAYBOOK.md must prescribe PoisonError::into_inner recovery (Issue #1940)"
    );
    assert!(
        unwrapped(PLAYBOOK).contains("if let Ok(guard)"),
        "DROUGHT_PLAYBOOK.md must name the `if let Ok(guard)` no-else-arm failure mode \
         (Issue #1940)"
    );
}

#[test]
fn playbook_bounds_the_poison_recovery_convention() {
    // Recovery is safe only because the guarded state is counters mutated by
    // infallible operations — it must not be presented as a blanket rule.
    assert!(
        PLAYBOOK.contains("infallible"),
        "DROUGHT_PLAYBOOK.md must justify recovery by the infallible mutations (Issue #1940)"
    );
    assert!(
        PLAYBOOK.contains("half-updated"),
        "DROUGHT_PLAYBOOK.md must exclude state a panic can leave half-updated (Issue #1940)"
    );
}

// ============================================================================
// Capture is the precondition for deletion — folded summaries are gone.
// ============================================================================

#[test]
fn folded_summaries_were_deleted_after_capture() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/docs/archive/pr-summaries");
    for n in [1792, 1793, 1795, 1799, 1802, 1806, 1815, 1817, 1818, 1875] {
        let path = format!("{dir}/pr-summary-{n}.md");
        assert!(
            !Path::new(&path).exists(),
            "pr-summary-{n}.md must be deleted once its learning is folded into the live docs \
             (Issue #1940)"
        );
    }
    // #1802's fail-loud reconciliation invariant already lives in its own live
    // analysis doc, which is the capture that permits deleting the summary.
    assert!(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/docs/analysis/candidate-reconciliation-1802.md"
        ))
        .exists(),
        "the #1802 reconciliation analysis doc must remain as the live capture (Issue #1940)"
    );
}
