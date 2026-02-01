## Summary

Extracted version history from README.md into a new CHANGELOG.md file (Issue #367).

The README.md was approximately 3,089 lines with roughly half being detailed
version-by-version change history. This made the README difficult to navigate.

**Changes made:**
- Created `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/)
  conventions with all 25 version-specific entries extracted from README.md
- Removed all version-specific `####` sections (v0.1.x, v0.2.x) from the README.md
  details block while preserving general documentation sections
- Added a reference to CHANGELOG.md in the README.md details block and in the
  "Additional documentation" section
- Australian English spelling preserved throughout (normalised, behaviour, etc.)

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.
The change is a documentation restructuring — no code behaviour was altered.

## Test Plan

- Added `tests/issue_367_changelog_extraction.rs` with 10 tests verifying:
  - `changelog_exists_and_has_title` — CHANGELOG.md has a `# Changelog` title
  - `changelog_references_keep_a_changelog` — References keepachangelog.com
  - `changelog_contains_version_entries` — Contains key version numbers (v0.1.115, v0.1.138, v0.2.1, v0.2.17)
  - `changelog_contains_version_descriptions` — Contains actual descriptive content
  - `readme_does_not_contain_versioned_section_headings` — All 24 versioned `####` headings removed from README
  - `readme_references_changelog` — README.md references CHANGELOG.md
  - `readme_retains_project_mission` — Project Mission section preserved
  - `readme_retains_coordinated_structural_discovery` — Coordinated Structural Discovery preserved
  - `readme_retains_discrete_activation_handling` — Discrete activation handling preserved
  - `changelog_uses_australian_english_normalised` — Australian English spelling used
- `./quality.sh` passes cleanly (all 434 unit tests + 96 integration tests)
