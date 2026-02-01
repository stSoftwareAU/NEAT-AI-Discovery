## Summary

Created 11 GitHub issues for cleaning up, refactoring, and rationalising the documentation
and source code after large changes over recent weeks. Follows the DRY principle and TDD
approach as requested.

### Documentation Cleanup Issues

| Issue | Title | Purpose |
|-------|-------|---------|
| #367 | Extract version history from README.md into CHANGELOG.md | Move ~1,500 lines of version-specific entries out of the 3,089-line README |
| #368 | Create AGENTS.md with coding guidelines for AI agents | Single source of truth for agent-relevant coding information |
| #369 | Slim down README.md to be human-readable | Target under 500 lines; link to CHANGELOG.md, AGENTS.md, and docs/ |
| #370 | Consolidate discovery type documentation (DRY) | Make docs/DISCOVERY_TYPES.md the single source of truth |
| #371 | Consolidate impact calculation documentation (DRY) | Make docs/IMPACT_CALCULATION.md the single source of truth |
| #372 | Add CONTRIBUTING.md with development guidelines | Standard Rust community practice for contributor onboarding |
| #373 | Add GitHub issue templates for bugs, features, and cleanup | Standardise issue creation |
| #374 | Archive old PR summary files from docs/ | Move 57 pr-summary-*.md files to docs/archive/ |

### Source Code Cleanup Issues

| Issue | Title | Purpose |
|-------|-------|---------|
| #375 | Extract generic discovery module dispatch pattern (DRY) | Reduce ~500 lines of repeated boilerplate in analyze_all() |
| #376 | Add unit tests for discovery detection modules | 9 detection modules currently have zero unit tests |
| #377 | Extract shared test utilities module | DRY for test fixture construction and GPU check macros |

### Key Findings

- **README.md** is 3,089 lines — roughly half is version-by-version change history
- **9 discovery modules** share an identical dispatch pattern repeated in `src/analysis/mod.rs`
- **9 detection modules** have zero unit tests (saturation, bottleneck, dead neuron, etc.)
- **No AGENTS.md**, CONTRIBUTING.md, or CHANGELOG.md exists
- **57 PR summary files** clutter the docs/ directory
- Discovery type and impact calculation information is duplicated between README.md and docs/

## Evidence

Unable to generate screenshot: This is a Rust FFI library with no visual interface. The
deliverable is a set of GitHub issues, not code changes.

## Test Plan

No code changes were made in this PR — only GitHub issues were created for future work.
The issues themselves specify TDD acceptance criteria where applicable.
