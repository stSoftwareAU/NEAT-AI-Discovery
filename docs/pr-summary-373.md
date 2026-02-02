## Summary

Added GitHub issue templates for bug reports, feature requests, and cleanup/refactoring tasks. Also added a template chooser configuration file. All templates use Australian English spelling and include appropriate frontmatter with labels.

### Templates created

- `.github/ISSUE_TEMPLATE/bug_report.md` — includes steps to reproduce, expected/actual behaviour, environment
- `.github/ISSUE_TEMPLATE/feature_request.md` — includes problem statement, proposed solution, acceptance criteria
- `.github/ISSUE_TEMPLATE/cleanup.md` — includes what to clean up, DRY violations, acceptance criteria
- `.github/ISSUE_TEMPLATE/config.yml` — enables blank issues alongside templates

Closes #373

## Evidence

Unable to generate screenshot: these are Markdown templates with no visual interface.

## Test Plan

- Added `tests/issue_373_github_issue_templates.rs` with 21 tests verifying:
  - All four template files exist and can be loaded
  - Bug report template has steps to reproduce, expected/actual behaviour, environment section
  - Bug report template uses Australian English (no American "behavior" spelling)
  - Feature request template has problem statement, proposed solution, acceptance criteria
  - Cleanup template has what to clean up, DRY violations, acceptance criteria
  - All templates have frontmatter with name, description, and labels
  - Config file has `blank_issues_enabled` setting
