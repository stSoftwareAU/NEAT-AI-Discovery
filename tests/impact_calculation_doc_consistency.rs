//! Verifies that impact calculation documentation is consolidated (DRY) (Issue #371).
//!
//! Ensures `docs/IMPACT_CALCULATION.md` is the single source of truth for impact
//! calculation, and that README.md only contains a brief summary with a link.

use std::fs;
use std::path::Path;

/// README.md must link to docs/IMPACT_CALCULATION.md for impact details.
#[test]
fn readme_links_to_impact_calculation_md() {
    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md");
    let content = fs::read_to_string(&readme_path).expect("Failed to read README.md");

    assert!(
        content.contains("docs/IMPACT_CALCULATION.md"),
        "README.md must link to docs/IMPACT_CALCULATION.md"
    );
}

/// README.md must NOT contain impact calculation formula details (DRY).
/// The formulas belong in docs/IMPACT_CALCULATION.md only.
#[test]
fn readme_does_not_contain_impact_formula_details() {
    let readme_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("README.md");
    let content = fs::read_to_string(&readme_path).expect("Failed to read README.md");

    // Must not contain mathematical formula notation for impact
    assert!(
        !content.contains("impact(n) ="),
        "README.md must not contain impact formula notation — \
         these belong in docs/IMPACT_CALCULATION.md"
    );

    // Must not contain the normalised path weight formula
    assert!(
        !content.contains("|w| / T") && !content.contains("|w|/T"),
        "README.md must not contain normalised path weight formula — \
         these belong in docs/IMPACT_CALCULATION.md"
    );

    // Must not contain activation_weighted_impact formula
    assert!(
        !content.contains("structural_impact × mean_absolute_activation")
            && !content.contains("structural_impact * mean_absolute_activation"),
        "README.md must not contain activation_weighted_impact formula — \
         these belong in docs/IMPACT_CALCULATION.md"
    );

    // Must not contain squash category implementation details
    assert!(
        !content.contains("SquashCategory"),
        "README.md must not contain SquashCategory implementation details — \
         these belong in docs/IMPACT_CALCULATION.md"
    );
}

/// docs/IMPACT_CALCULATION.md must be the single source of truth.
/// It must contain all key impact formula components.
#[test]
fn impact_calculation_md_is_single_source_of_truth() {
    let doc_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/IMPACT_CALCULATION.md");
    let content = fs::read_to_string(&doc_path).expect("Failed to read docs/IMPACT_CALCULATION.md");

    // Must contain the basic impact formula
    assert!(
        content.contains("impact"),
        "docs/IMPACT_CALCULATION.md must describe the impact formula"
    );

    // Must contain squash-aware handling
    assert!(
        content.contains("Squash") || content.contains("squash"),
        "docs/IMPACT_CALCULATION.md must document squash-aware impact handling"
    );

    // Must document the three squash categories
    assert!(
        content.contains("Linear")
            && content.contains("Threshold")
            && content.contains("Selection"),
        "docs/IMPACT_CALCULATION.md must document all three squash categories \
         (Linear, Threshold, Selection)"
    );

    // Must contain activation-weighted impact
    assert!(
        content.contains("activation_weighted_impact")
            || content.contains("Activation-Weighted Impact")
            || content.contains("activation-weighted impact"),
        "docs/IMPACT_CALCULATION.md must document activation-weighted impact"
    );

    // Must contain mathematical formulas
    assert!(
        content.contains("$$") || content.contains("\\text{impact}"),
        "docs/IMPACT_CALCULATION.md must contain mathematical formulas"
    );

    // Must contain the related code reference table
    assert!(
        content.contains("compute_impacts") || content.contains("compute_impact"),
        "docs/IMPACT_CALCULATION.md must reference the implementation functions"
    );

    // Must document the removal threshold (costOfGrowth)
    assert!(
        content.contains("costOfGrowth"),
        "docs/IMPACT_CALCULATION.md must document the costOfGrowth removal threshold"
    );

    // Must document removal savings formula
    assert!(
        content.contains("removalSavings") || content.contains("removal_savings"),
        "docs/IMPACT_CALCULATION.md must document removal savings calculation"
    );
}

/// CHANGELOG.md should reference IMPACT_CALCULATION.md for detailed explanation
/// rather than duplicating formula details.
#[test]
fn changelog_links_to_impact_calculation_md_for_details() {
    let changelog_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("CHANGELOG.md");
    let content = fs::read_to_string(&changelog_path).expect("Failed to read CHANGELOG.md");

    // CHANGELOG should reference the detailed doc for squash-aware impact
    assert!(
        content.contains("docs/IMPACT_CALCULATION.md") || content.contains("IMPACT_CALCULATION.md"),
        "CHANGELOG.md should reference docs/IMPACT_CALCULATION.md for detailed \
         impact calculation explanation"
    );
}

/// AGENTS.md should reference IMPACT_CALCULATION.md (not duplicate formulas).
#[test]
fn agents_md_links_to_impact_calculation_md() {
    let agents_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("AGENTS.md");
    let content = fs::read_to_string(&agents_path).expect("Failed to read AGENTS.md");

    assert!(
        content.contains("IMPACT_CALCULATION.md"),
        "AGENTS.md must link to docs/IMPACT_CALCULATION.md"
    );

    // AGENTS.md must not contain impact formula details
    assert!(
        !content.contains("impact(n) ="),
        "AGENTS.md must not contain impact formula notation — \
         these belong in docs/IMPACT_CALCULATION.md"
    );
}
