# PR Summary — Issue #2346

## Summary

Bound correlated-error analysis with output and input ceilings, and propagate
the shared analysis deadline through its matrix and predictive-input scans.
An interrupted matrix is discarded; cancellation during group analysis keeps
only completed groups. Version bump: `VERSION` updated for this change; the
merged commit records the final version. Closes #2346.

## Spec

### Intent and Rationale

- Untrusted creatures could drive an unbounded dense output-correlation matrix
  and long predictive-input scans. Fixed ceilings and in-loop cancellation
  bound those operations.

### Essential Design Decisions

- Eligible output counts above 1,000 skip before matrix allocation; input counts
  above 10,000 skip only predictive-input searches, retaining completed output
  groups.
- A matrix interrupted while incomplete is never clustered. A predictive-input
  interruption stops later groups while preserving groups already completed.
- The legacy API remains available and delegates to the deadline-aware API with
  no explicit deadline; dispatch uses the deadline-aware API.

### Undiscoverable Facts

- `deadline_passed` also observes global cancellation, so these checks respond
  to both the wall-clock deadline and the global cancellation signal.

## Evidence

Backend-only change; no visual surface is affected. The original trigger is
closed: more than 1,000 eligible outputs return `OutputCeilingExceeded` before
the dense matrix allocation, and the pair and predictive-input scans now
observe cancellation. The security regression test failed against the base
production implementation with the assertion `1,001 eligible outputs must be
rejected by the 1,000-output ceiling`, then passed against this branch.

## Test Plan

- `tests/issue_2346_correlated_error_output_ceiling_test.rs::rejects_output_count_above_ceiling_without_building_the_matrix` — failed against base production code (expected regression; exit 101), passed at head.
- `tests/issue_2346_correlated_error_matrix_guard_test.rs` — 7 passed, including above-limit rejection and exact output/input ceiling boundaries.
- `src/analysis/detection/correlated_error.rs::deadline_during_pair_scan_returns_no_groups` — passed; removing the pair-scan guard made it fail because the incomplete matrix was clustered.
- `src/analysis/detection/correlated_error.rs::deadline_during_group_search_keeps_completed_groups` — passed; the cancellation case asserts later groups are not processed and completed groups survive.
- `src/analysis/module_dispatch_specs/structural_specs_issue_2346_tests.rs::correlated_error_spec_stops_at_elapsed_deadline` — passed; restoring the caller to the legacy wrapper makes the assertion fail.
- `cargo fmt --all -- --check` — passed.
- `cargo clippy --all-targets -- -D warnings` — passed.
- `./quality.sh` — not completed: timed out after 900 seconds (exit 124) while building the release library. The complete test phase passed (1,659 library tests and all integration test targets); documentation generation passed. The release build did not finish.

**Docs sweep** — grep: `detect_correlated_error_patterns`, `find_predictive_inputs`, `CorrelatedError`, `MAX_CORRELATED_ERROR_OUTPUTS`, `MAX_CORRELATED_ERROR_INPUTS`, and deadline claims; section: `docs/discoveries/correlated-error.md#-bounds-issue-2346`, `docs/DISCOVERY_TYPES.md#correlated-error-pattern-detection`, and `docs/ANALYSIS_DEEP_DIVE.md#-correlated-error-pattern-detection-issue-344`; updated: `docs/discoveries/correlated-error.md`, `docs/DISCOVERY_TYPES.md`, `docs/ANALYSIS_DEEP_DIVE.md`, and the `correlated_error.rs` module comment. Remaining hits: `docs/COST_FUNCTION_NOTES.md:124,125,331` describe the unchanged error-field reads; `docs/ANALYSIS_DEEP_DIVE.md:283-284` describe the new ceilings; these remain accurate. Archive audit entries were excluded from the manual sweep.

The tested entry points are the public detector (both the deadline-aware API
and legacy wrapper), the structural-spec closure, and the dispatch deadline
forwarding path. No assertions were removed from existing tests.

## Security self-check

- [x] Oversized output and input counts are bounded before their expensive work.
- [x] No secrets staged; no new shell, filesystem, or network surface.
- [x] No new dependencies.
