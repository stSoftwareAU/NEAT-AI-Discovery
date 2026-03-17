## Summary

Stricter Cargo.toml validation in CI to exclude comment lines and require exact field matches. Closes #691.

The previous `grep -q "$field = "` pattern had two false-positive risks:
1. Commented-out fields (e.g. `# name = "old"`) would still match
2. Substring matches (e.g. `rust-version` matching `version`)

Updated to use a two-stage grep:
- `grep -vE "^[[:space:]]*#"` strips comment lines
- `grep -qE "^[[:space:]]*${field}[[:space:]]*="` requires the field at line start

## Evidence

This is a CI/shell change with no visual output. Verified by:
- Shell test script `tests/test_cargo_toml_validation.sh` covering 16 test cases
- `quality.sh` passes cleanly

## Test Plan

- Added `tests/test_cargo_toml_validation.sh` with 16 test cases:
  - Uncommented fields are correctly detected
  - Commented-out fields are excluded
  - Substring fields (e.g. `rust-version`) do not match `version`
  - Varying whitespace is handled
  - Real `Cargo.toml` passes validation
  - Regression tests for old pattern false positives
