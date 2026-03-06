## Summary

Remove redundant `.to_uppercase()` calls on squash strings that are already normalised to uppercase at load time (Issue #753). Closes #771.

**Changes:**

- `input_sensitivity.rs`: Replaced two `.to_uppercase().as_str()` calls with direct `&str` matching, since squash strings from `orchestration.rs` are guaranteed uppercase.
- `activation_properties.rs`: Replaced `.to_uppercase().as_str()` in `is_saturating_squash()` with `eq_ignore_ascii_case()` to remain case-insensitive (public API) while avoiding String allocations.

## Evidence

This is a backend-only change with no visual output. All existing tests pass, and `quality.sh` passes cleanly. No `.to_uppercase()` calls remain in detection modules.

## Test Plan

- All existing tests continue to pass (including `test_is_saturating_squash` in `weight_coherence.rs` which tests both uppercase and lowercase inputs)
- Doc tests for `is_saturating_squash` verify case-insensitive behaviour is preserved
- `quality.sh` passes cleanly (fmt, clippy, check, test, doc, release build)
