//! Issue #373: Verify GitHub issue templates exist and follow conventions.
//!
//! These tests ensure:
//! - Bug report, feature request, and cleanup templates exist
//! - Template chooser config exists
//! - Templates use Australian English spelling
//! - Templates contain required sections

const BUG_REPORT: &str = include_str!("../.github/ISSUE_TEMPLATE/bug_report.md");
const FEATURE_REQUEST: &str = include_str!("../.github/ISSUE_TEMPLATE/feature_request.md");
const CLEANUP: &str = include_str!("../.github/ISSUE_TEMPLATE/cleanup.md");
const CONFIG: &str = include_str!("../.github/ISSUE_TEMPLATE/config.yml");

// ---------------------------------------------------------------------------
// 1. Bug report template
// ---------------------------------------------------------------------------

#[test]
fn bug_report_has_frontmatter_name() {
    assert!(
        BUG_REPORT.contains("name:"),
        "bug_report.md must contain a 'name:' field in frontmatter"
    );
}

#[test]
fn bug_report_has_description() {
    assert!(
        BUG_REPORT.contains("description:"),
        "bug_report.md must contain a 'description:' field in frontmatter"
    );
}

#[test]
fn bug_report_has_steps_to_reproduce() {
    let lower = BUG_REPORT.to_lowercase();
    assert!(
        lower.contains("steps to reproduce") || lower.contains("reproduce"),
        "bug_report.md must contain a 'Steps to reproduce' section"
    );
}

#[test]
fn bug_report_has_expected_behaviour() {
    assert!(
        BUG_REPORT.contains("Expected behaviour") || BUG_REPORT.contains("expected behaviour"),
        "bug_report.md must use Australian English 'behaviour' (not 'behavior')"
    );
}

#[test]
fn bug_report_has_actual_behaviour() {
    assert!(
        BUG_REPORT.contains("Actual behaviour") || BUG_REPORT.contains("actual behaviour"),
        "bug_report.md must use Australian English 'behaviour' (not 'behavior')"
    );
}

#[test]
fn bug_report_has_environment_section() {
    let lower = BUG_REPORT.to_lowercase();
    assert!(
        lower.contains("environment"),
        "bug_report.md must contain an 'Environment' section"
    );
}

#[test]
fn bug_report_does_not_use_american_spelling() {
    assert!(
        !BUG_REPORT.contains("behavior"),
        "bug_report.md must not use American spelling 'behavior'"
    );
}

// ---------------------------------------------------------------------------
// 2. Feature request template
// ---------------------------------------------------------------------------

#[test]
fn feature_request_has_frontmatter_name() {
    assert!(
        FEATURE_REQUEST.contains("name:"),
        "feature_request.md must contain a 'name:' field in frontmatter"
    );
}

#[test]
fn feature_request_has_description() {
    assert!(
        FEATURE_REQUEST.contains("description:"),
        "feature_request.md must contain a 'description:' field in frontmatter"
    );
}

#[test]
fn feature_request_has_problem_statement() {
    let lower = FEATURE_REQUEST.to_lowercase();
    assert!(
        lower.contains("problem"),
        "feature_request.md must contain a problem statement section"
    );
}

#[test]
fn feature_request_has_proposed_solution() {
    let lower = FEATURE_REQUEST.to_lowercase();
    assert!(
        lower.contains("proposed solution"),
        "feature_request.md must contain a 'Proposed solution' section"
    );
}

#[test]
fn feature_request_has_acceptance_criteria() {
    let lower = FEATURE_REQUEST.to_lowercase();
    assert!(
        lower.contains("acceptance criteria"),
        "feature_request.md must contain an 'Acceptance criteria' section"
    );
}

// ---------------------------------------------------------------------------
// 3. Cleanup template
// ---------------------------------------------------------------------------

#[test]
fn cleanup_has_frontmatter_name() {
    assert!(
        CLEANUP.contains("name:"),
        "cleanup.md must contain a 'name:' field in frontmatter"
    );
}

#[test]
fn cleanup_has_description() {
    assert!(
        CLEANUP.contains("description:"),
        "cleanup.md must contain a 'description:' field in frontmatter"
    );
}

#[test]
fn cleanup_has_what_to_clean_up() {
    let lower = CLEANUP.to_lowercase();
    assert!(
        lower.contains("clean up") || lower.contains("cleanup") || lower.contains("refactor"),
        "cleanup.md must describe what to clean up or refactor"
    );
}

#[test]
fn cleanup_has_dry_violations() {
    assert!(
        CLEANUP.contains("DRY"),
        "cleanup.md must mention DRY violations"
    );
}

#[test]
fn cleanup_has_acceptance_criteria() {
    let lower = CLEANUP.to_lowercase();
    assert!(
        lower.contains("acceptance criteria"),
        "cleanup.md must contain an 'Acceptance criteria' section"
    );
}

// ---------------------------------------------------------------------------
// 4. Config file
// ---------------------------------------------------------------------------

#[test]
fn config_is_valid_yaml_with_blank_issues() {
    assert!(
        CONFIG.contains("blank_issues_enabled"),
        "config.yml must configure blank_issues_enabled"
    );
}

// ---------------------------------------------------------------------------
// 5. Templates use labels
// ---------------------------------------------------------------------------

#[test]
fn bug_report_has_labels() {
    assert!(
        BUG_REPORT.contains("labels:"),
        "bug_report.md must specify labels in frontmatter"
    );
}

#[test]
fn feature_request_has_labels() {
    assert!(
        FEATURE_REQUEST.contains("labels:"),
        "feature_request.md must specify labels in frontmatter"
    );
}

#[test]
fn cleanup_has_labels() {
    assert!(
        CLEANUP.contains("labels:"),
        "cleanup.md must specify labels in frontmatter"
    );
}
