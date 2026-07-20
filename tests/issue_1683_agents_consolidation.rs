//! Issue #1683 — `AGENTS.md` had grown into a 685-line parallel content store
//! whose largest section (a hand-maintained mirror of the source tree) had
//! materially drifted, while the human docs deferred *into* it — the inverse of
//! the intended hierarchy.
//!
//! These tests lock in the consolidation:
//!   1. The drift-prone source-tree mirror and its stale counts are gone.
//!   2. `AGENTS.md` is now a thin, agent-only pointer file.
//!   3. The canonical coding conventions, testing philosophy, and quality gate
//!      live in `CONTRIBUTING.md`, which no longer defers back into `AGENTS.md`.
//!   4. The quality-gate description matches `quality.sh` (shellcheck + PR-summary
//!      location) and the CI trigger matches `ci.yml` (`milestone/*`).
//!   5. The agent-only invariants (`validate_forward_only_synapses`, the
//!      do-not-touch-`ci.yml` rule) are preserved, and the broken emoji anchors
//!      are gone.

const AGENTS: &str = include_str!("../AGENTS.md");
const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");
const QUALITY_SH: &str = include_str!("../quality.sh");
const CI_YML: &str = include_str!("../.github/workflows/ci.yml");

// ============================================================================
// 1. The drifted source-tree mirror is gone.
// ============================================================================

#[test]
fn agents_no_longer_mirrors_the_source_tree() {
    // The phantom path the mirror listed (only src/parquet_format/shared_records.rs
    // exists) must be gone along with the whole "Source Layout" tree.
    assert!(
        !AGENTS.contains("src/analysis/cache/shared_records.rs"),
        "AGENTS.md must not list the phantom src/analysis/cache/shared_records.rs (Issue #1683)"
    );
    assert!(
        !AGENTS.contains("### Source Layout"),
        "AGENTS.md must not carry the hand-maintained source-tree mirror (Issue #1683)"
    );
}

#[test]
fn agents_drops_the_stale_directory_counts() {
    // The mirror's counts had drifted (tests/ said ~286, benches/ said 31 suites).
    assert!(
        !AGENTS.contains("~286 files"),
        "AGENTS.md must not carry the stale tests/ file count (Issue #1683)"
    );
    assert!(
        !AGENTS.contains("31 suites"),
        "AGENTS.md must not carry the stale benches/ suite count (Issue #1683)"
    );
}

#[test]
fn agents_is_thin() {
    // A pointer file, not a 685-line parallel content store.
    let lines = AGENTS.lines().count();
    assert!(
        lines < 200,
        "AGENTS.md should be a thin pointer file, but has {lines} lines (Issue #1683)"
    );
}

// ============================================================================
// 2. The human docs own the moved content; AGENTS points to them.
// ============================================================================

#[test]
fn agents_points_at_the_human_docs() {
    assert!(
        AGENTS.contains("CONTRIBUTING.md"),
        "AGENTS.md must point at CONTRIBUTING.md for the moved conventions (Issue #1683)"
    );
    assert!(
        AGENTS.contains("README.md"),
        "AGENTS.md must point at README.md for the user-facing overview (Issue #1683)"
    );
}

#[test]
fn contributing_no_longer_defers_into_agents() {
    // The inbound pointers that made AGENTS.md canonical are flipped.
    for stub in [
        "AGENTS.md#2-architecture",
        "AGENTS.md#3-coding-conventions",
        "AGENTS.md#4-testing-philosophy",
        "AGENTS.md#5-quality-gate",
    ] {
        assert!(
            !CONTRIBUTING.contains(stub),
            "CONTRIBUTING.md must not defer to {stub}; it now owns that content (Issue #1683)"
        );
    }
}

#[test]
fn contributing_owns_the_coding_conventions_and_testing_doctrine() {
    assert!(
        CONTRIBUTING.contains("Australian English"),
        "CONTRIBUTING.md must document the Australian English requirement (Issue #1683)"
    );
    // The "what" vs "how" testing doctrine moved here from AGENTS.md.
    assert!(
        CONTRIBUTING.contains("Benchmarks disguised as tests"),
        "CONTRIBUTING.md must document the testing doctrine (Issue #1683)"
    );
}

// ============================================================================
// 3. The quality-gate description matches reality (quality.sh + ci.yml).
// ============================================================================

#[test]
fn contributing_quality_gate_lists_the_real_steps() {
    // quality.sh really runs shellcheck and the PR-summary location check; the
    // stale AGENTS.md list omitted both.
    assert!(
        QUALITY_SH.contains("shellcheck"),
        "quality.sh runs shellcheck"
    );
    assert!(
        QUALITY_SH.contains("check-pr-summary-location.sh"),
        "quality.sh runs the PR-summary location check"
    );
    assert!(
        CONTRIBUTING.contains("shellcheck"),
        "CONTRIBUTING.md quality gate must list the shellcheck step (Issue #1683)"
    );
    assert!(
        CONTRIBUTING.contains("check-pr-summary-location.sh"),
        "CONTRIBUTING.md quality gate must list the PR-summary location step (Issue #1683)"
    );
}

#[test]
fn contributing_ci_trigger_matches_ci_yml() {
    // ci.yml triggers on Develop AND milestone/* (Issue #1651); the doc must say so.
    assert!(
        CI_YML.contains("milestone/*"),
        "ci.yml triggers on milestone/*"
    );
    assert!(
        CONTRIBUTING.contains("milestone/"),
        "CONTRIBUTING.md must document the milestone/* CI trigger (Issue #1683)"
    );
}

// ============================================================================
// 4. Agent-only invariants and rules are preserved.
// ============================================================================

#[test]
fn agents_keeps_the_agent_only_invariants() {
    assert!(
        AGENTS.contains("validate_forward_only_synapses"),
        "AGENTS.md must keep the forward-only FFI validation invariant (Issue #1683)"
    );
    assert!(
        AGENTS.contains("ci.yml"),
        "AGENTS.md must keep the do-not-modify-ci.yml rule (Issue #1683)"
    );
    assert!(
        AGENTS.contains("free_discovery_result"),
        "AGENTS.md must keep the FFI memory-free invariant (Issue #1683)"
    );
}

// ============================================================================
// 5. Duplication and broken anchors are gone.
// ============================================================================

#[test]
fn agents_drops_the_duplicated_blocks() {
    // The runlib.sh description (triplicated in README/CONTRIBUTING) is gone.
    assert!(
        !AGENTS.contains("Installs Rust and Cargo if missing"),
        "AGENTS.md must not duplicate the runlib.sh description (Issue #1683)"
    );
    // The candidate-type table (duplicated in README/DISCOVERY_TYPES) is gone.
    assert!(
        !AGENTS.contains("| `addNeuron` |"),
        "AGENTS.md must not duplicate the candidate-type table (Issue #1683)"
    );
}

#[test]
fn agents_broken_emoji_anchors_are_gone() {
    // README's headings carry emoji, so #gpu-requirement / #additional-documentation
    // never resolved. They must be retargeted or dropped.
    assert!(
        !AGENTS.contains("README.md#gpu-requirement"),
        "AGENTS.md must not link the broken README.md#gpu-requirement anchor (Issue #1683)"
    );
    assert!(
        !AGENTS.contains("README.md#additional-documentation"),
        "AGENTS.md must not link the broken README.md#additional-documentation anchor (Issue #1683)"
    );
}
