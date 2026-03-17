## Summary

Audit and remove or convert "how" tests to behavioural "what" tests. Closes #813.

Reviewed all 10 test files identified in the audit. Each was either removed, converted,
or kept with justification:

### Removed (6 files) — tested documentation structure/file existence, not behaviour:
- `tests/discovery_types_doc_consistency.rs` — read source files and docs, checked for cross-references
- `tests/impact_calculation_doc_consistency.rs` — read README/CHANGELOG, checked for formula patterns
- `tests/issue_367_changelog_extraction.rs` — embedded README/CHANGELOG, checked headings and version strings
- `tests/issue_346_readme_project_goal.rs` — embedded README, checked for specific phrases
- `tests/issue_373_github_issue_templates.rs` — embedded issue templates, checked YAML/markdown structure
- `tests/issue_410_discovery_scenario_docs.rs` — read docs/discoveries/ files, checked section headings

### Converted (2 files) — removed file-reading, kept behavioural tests:
- `tests/activation_coverage_neat_ai_registry.rs` — removed `fs::read_to_string()` of external
  TypeScript files; now tests `apply_scalar_squash`, `is_known_squash_name`, `is_aggregate_squash`
  directly with hardcoded activation names and concrete input/output assertions
- `tests/issue_576_benchmark_regression_tracking.rs` — removed file existence checks for benchmark
  `.rs` files; retained behavioural tests that run `benchmark_compare.sh` and verify its outputs
  (help, list, error handling)

### Kept (2 files) — genuinely test "what":
- `tests/issue_477_rust_edition_2024.rs` — tests that Rust 2024 edition features compile correctly
  (let-chains, reserved keywords, unsafe attributes); compilation itself is the behaviour being tested
- `tests/issue_804_detection_helpers.rs` — calls `build_record_map()` with test data and asserts on
  results; tests the public API output, not implementation details

## Evidence
- No remaining tests use `fs::read_to_string()` or `include_str!()` to inspect source code or
  documentation files for keywords
- `quality.sh` passes cleanly after all changes

## Test Plan
- Converted `tests/activation_coverage_neat_ai_registry.rs` — 7 behavioural tests covering scalar
  and aggregate activation recognition, finite output, identity correctness, ReLU behaviour, alias
  consistency, case-insensitive recognition, and unknown name handling
- Converted `tests/issue_576_benchmark_regression_tracking.rs` — 5 behavioural tests covering
  script syntax validation, suite discovery, help output, unknown bench rejection, and missing
  baseline handling
- Kept `tests/issue_477_rust_edition_2024.rs` (3 tests) and `tests/issue_804_detection_helpers.rs`
  (4 tests) unchanged
