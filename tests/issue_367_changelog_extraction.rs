//! Issue #367: Verify that version history has been extracted from README.md
//! into CHANGELOG.md following Keep a Changelog conventions.
//!
//! These tests ensure:
//! - CHANGELOG.md exists and follows the expected format
//! - README.md no longer contains version-specific entries
//! - README.md references CHANGELOG.md for historical changes
//! - Australian English spelling is used throughout

const README: &str = include_str!("../README.md");
const CHANGELOG: &str = include_str!("../CHANGELOG.md");

// ---------------------------------------------------------------------------
// 1. CHANGELOG.md exists and follows Keep a Changelog conventions
// ---------------------------------------------------------------------------

#[test]
fn changelog_exists_and_has_title() {
    assert!(
        CHANGELOG.contains("# Changelog"),
        "CHANGELOG.md must contain a '# Changelog' title"
    );
}

#[test]
fn changelog_references_keep_a_changelog() {
    assert!(
        CHANGELOG.contains("keepachangelog.com"),
        "CHANGELOG.md must reference the Keep a Changelog convention"
    );
}

#[test]
fn changelog_contains_version_entries() {
    // Must contain at least some of the version entries extracted from README
    assert!(
        CHANGELOG.contains("v0.1.115"),
        "CHANGELOG.md must contain v0.1.115 entry (Bias-aware weight calculation)"
    );
    assert!(
        CHANGELOG.contains("v0.1.138"),
        "CHANGELOG.md must contain v0.1.138 entry (Tighter outgoing weight clamp)"
    );
    assert!(
        CHANGELOG.contains("v0.2.1"),
        "CHANGELOG.md must contain v0.2.1 entry (Normalised impact calculation)"
    );
    assert!(
        CHANGELOG.contains("v0.2.17"),
        "CHANGELOG.md must contain v0.2.17 entry (Synapse-friendly discovery)"
    );
}

#[test]
fn changelog_contains_version_descriptions() {
    // Verify actual content was moved, not just version numbers
    assert!(
        CHANGELOG.contains("Bias-aware weight calculation"),
        "CHANGELOG.md must contain the bias-aware weight calculation description"
    );
    assert!(
        CHANGELOG.contains("Tighter outgoing weight clamp"),
        "CHANGELOG.md must contain the tighter outgoing weight clamp description"
    );
    assert!(
        CHANGELOG.contains("Synapse-friendly discovery"),
        "CHANGELOG.md must contain the synapse-friendly discovery description"
    );
}

// ---------------------------------------------------------------------------
// 2. README.md no longer contains version-specific entries
// ---------------------------------------------------------------------------

#[test]
fn readme_does_not_contain_versioned_section_headings() {
    // These version-specific #### headings should have been moved to CHANGELOG.md
    let versioned_headings = [
        "#### Bias-aware weight calculation (v0.1.115)",
        "#### Tighter outgoing weight clamp (v0.1.138)",
        "#### Expanded activation functions and discrete weight fix (v0.1.139)",
        "#### Prediction validation (v0.1.140)",
        "#### GPU shader activation function fix (v0.1.141)",
        "#### Root cause identified: Sample representativeness (v0.1.142)",
        "#### Impact calculation fix: Absolute not normalised (v0.1.145)",
        "#### VALUE domain error interpretation (v0.1.117)",
        "#### ACTIVATION domain consistency (v0.1.120)",
        "#### Simplified candidate filtering (v0.1.134)",
        "#### Split-error evaluation for all activations (v0.1.135)",
        "#### Split-error fallback candidate fix (v0.1.136)",
        "#### Hidden neuron impact discounting (v0.1.123)",
        "#### Impact calculation fix (v0.1.126)",
        "#### Dynamic removal threshold based on synapse counts (v0.1.127)",
        "#### Squash-aware impact calculation (v0.1.132)",
        "#### Removal candidate expected error reduction fix (v0.1.162)",
        "#### Saturation detection for add-neuron candidates (v0.1.167, Issue #123)",
        "#### Creature-level metrics (v0.1.169, Issue #128)",
        "#### Normalised impact calculation (v0.2.1, Issue #130)",
        "#### Source variance discounting (v0.2.2, Issue #130)",
        "#### Configurable costOfGrowth (v0.2.3, Issue #132)",
        "#### Synapse-friendly discovery (v0.2.17)",
        "#### Synapse analysis for all squash types (v0.2.18)",
    ];

    for heading in &versioned_headings {
        assert!(
            !README.contains(heading),
            "README.md must not contain version-specific heading: {heading}"
        );
    }
}

// ---------------------------------------------------------------------------
// 3. README.md references CHANGELOG.md for historical changes
// ---------------------------------------------------------------------------

#[test]
fn readme_references_changelog() {
    assert!(
        README.contains("CHANGELOG.md"),
        "README.md must reference CHANGELOG.md for version history"
    );
}

// ---------------------------------------------------------------------------
// 4. README.md retains essential non-versioned documentation
// ---------------------------------------------------------------------------

#[test]
fn readme_retains_project_mission() {
    assert!(
        README.contains("## Project Mission"),
        "README.md must still contain the Project Mission section"
    );
}

#[test]
fn readme_retains_coordinated_structural_discovery() {
    assert!(
        README.contains("### Coordinated Structural Discovery"),
        "README.md must still contain the Coordinated Structural Discovery section"
    );
}

#[test]
fn readme_retains_discrete_activation_handling() {
    assert!(
        README.contains("### Discrete activation function handling"),
        "README.md must still contain the Discrete activation function handling section"
    );
}

// ---------------------------------------------------------------------------
// 5. CHANGELOG.md uses Australian English spelling
// ---------------------------------------------------------------------------

#[test]
fn changelog_uses_australian_english_normalised() {
    // The changelog contains "normalised" (Australian) not "normalized" (American)
    // from the entries that were moved
    if CHANGELOG.contains("normalis") || CHANGELOG.contains("Normalis") {
        assert!(
            !CHANGELOG.contains("normalized") || CHANGELOG.contains("normalised"),
            "CHANGELOG.md must use Australian English spelling: 'normalised' not 'normalized'"
        );
    }
}
