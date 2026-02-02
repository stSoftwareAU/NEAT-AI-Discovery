## Summary

Consolidates impact calculation documentation so `docs/IMPACT_CALCULATION.md` is
the single source of truth (DRY principle), per issue #371.

### Changes

- **`docs/IMPACT_CALCULATION.md`**: Added removal threshold (`costOfGrowth`)
  section documenting the configurable threshold, common values, removal savings
  formula, and non-finite value filtering. This information was previously only
  available in CHANGELOG.md version entries.

- **`tests/impact_calculation_doc_consistency.rs`** (new): Added 5 integration
  tests enforcing DRY constraints:
  - `readme_links_to_impact_calculation_md` — README must link to the doc
  - `readme_does_not_contain_impact_formula_details` — README must not duplicate
    formula notation, normalised path weight formula, activation-weighted impact
    formula, or SquashCategory details
  - `impact_calculation_md_is_single_source_of_truth` — doc must contain all key
    components (formulas, squash categories, activation-weighted impact,
    costOfGrowth, removal savings)
  - `changelog_links_to_impact_calculation_md_for_details` — CHANGELOG must
    reference the detailed doc
  - `agents_md_links_to_impact_calculation_md` — AGENTS.md must link to doc
    without duplicating formulas

### What was already in place

The prior cleanup work (issue #370) had already removed impact formula details
from README.md and established the link to `docs/IMPACT_CALCULATION.md`. This PR
adds the automated test enforcement and fills in the missing `costOfGrowth`
removal threshold documentation.

## Evidence

Unable to generate screenshot: This is a Rust library with no visual interface.

## Test Plan

- Added `tests/impact_calculation_doc_consistency.rs` with 5 tests verifying:
  - README.md links to IMPACT_CALCULATION.md
  - README.md does not contain impact formula details
  - IMPACT_CALCULATION.md is the single source of truth
  - CHANGELOG.md references IMPACT_CALCULATION.md
  - AGENTS.md links to IMPACT_CALCULATION.md without duplicating formulas
- `./quality.sh` passes cleanly
