## Summary

`cap_deadline_to_wall_clock` multiplied the caller-supplied minutes by 60 with an unchecked `u64` multiply, so a large `max_discovery_wall_clock_minutes` wrapped to a short cap in release builds and panicked in debug builds. The multiply is now checked (an unrepresentable cap yields no wall-clock deadline instead of a wrapped one), and `analyze_all` clamps the FFI value to 1–120 minutes through a new `config::effective_wall_clock_minutes`, reached via a private `resolve_wall_clock_deadline` helper, logging a `WARN` when it changes the value. Version bump: `Cargo.toml` updated for this change; the merged commit records the final version. Closes #2460.

## Spec

### Intent and Rationale

- Two layers, as the issue suggests: a checked multiply makes the helper safe for any caller, and the clamp keeps the FFI value in the documented 1–120 range already enforced for the env value.

### Essential Design Decisions

- The FFI value is clamped (not rejected) to match the env accessor; an FFI `0` therefore becomes 1 minute rather than an immediate deadline.
- An unrepresentable cap returns the analysis deadline unchanged (no cap), rather than an error, keeping the function's signature.

### Undiscoverable Facts

- None.

## Evidence

- Regression tests in `tests/issue_2460_wall_clock_cap_overflow.rs` and `tests/issue_2460_analyze_all_wall_clock_clamp.rs`, and a unit test in `src/config/mod.rs`.
- `tests/issue_2460_analyze_all_wall_clock_clamp.rs::analyze_all_reports_clamped_cap_for_oversized_values` drives `analyze_all` with both overflow triggers and checks the WARN reports the requested value and the clamped 120-minute value on GPU hosts. `::clamp_warns_only_when_value_changes` checks WARN fields for out-of-range values and no WARN for in-range or absent values without a GPU.
- Against the base `src/analysis/utils/deadline.rs` (all else unchanged), `tests/issue_2460_wall_clock_cap_overflow.rs` fails 3 of 4 with `attempt to multiply with overflow` (`deadline.rs::cap_deadline_to_wall_clock`); with the fix all 4 pass (the file compiles on the base: it only uses `cap_deadline_to_wall_clock`, which exists there). The one passing on base is the in-range boundary test.
- Original trigger closed: `307445734561825861` and `u64::MAX` are clamped to 120 by `effective_wall_clock_minutes` in `analyze_all` (the only reader of the field), and even if passed directly to `cap_deadline_to_wall_clock` the `checked_mul` yields `None` for the cap, so there is no wrap and no panic. No equivalent bypass exists: `cap_deadline_to_wall_clock`'s only production caller is `analyze_all`.
- Docs updated: `docs/CONFIGURATION.md`, `docs/FFI_API.md`, doc comments in `src/ffi_types/requests.rs`, `src/config/user_facing.rs`, `src/analysis/utils/deadline.rs`. Historical records in `docs/audits/security-sweep-chunk-13-config-env.md` left as written.

## Test Plan

- `tests/issue_2460_wall_clock_cap_overflow.rs::wrapping_multiplier_does_not_produce_short_cap` — red on base, green at head.
- `tests/issue_2460_wall_clock_cap_overflow.rs::u64_max_minutes_yields_no_wall_clock_deadline` — red on base, green at head.
- `tests/issue_2460_wall_clock_cap_overflow.rs::unrepresentable_cap_preserves_analysis_deadline` — red on base, green at head.
- `tests/issue_2460_wall_clock_cap_overflow.rs::boundary_minutes_still_cap_normally` — green on base and head (in-range guard).
- `tests/issue_2460_analyze_all_wall_clock_clamp.rs::analyze_all_reports_clamped_cap_for_oversized_values` — checks the clamp WARN at the production entry point when a GPU is available.
- `tests/issue_2460_analyze_all_wall_clock_clamp.rs::clamp_warns_only_when_value_changes` — verifies warning fields and absence of warnings for unchanged values.
- Review correction: the obsolete INFO assertion could not observe a 120-minute cap because the analysis deadline is shorter. The remaining entry-point WARN assertion checks the requested and clamped values in `tests/issue_2460_analyze_all_wall_clock_clamp.rs::analyze_all_reports_clamped_cap_for_oversized_values`.
- Review correction: the obsolete publicly exported, helper-only test path could not detect a bypass in `analyze_all`. The helper remains private and the shipped entry point is covered by `tests/issue_2460_analyze_all_wall_clock_clamp.rs::analyze_all_reports_clamped_cap_for_oversized_values`.
- `src/config/mod.rs::effective_wall_clock_minutes_clamps_ffi_value` — unit test for the clamp bounds and the `None` fallback (new function, so it lives in the crate and is not cited as a base/head discriminator).
- Existing `cap_deadline` unit tests and `issue_1098_wall_clock_cap` (8 tests) pass unchanged.
- Callers checked: `analyze_all` (src/analysis/orchestration.rs) is the only production reader of `max_discovery_wall_clock_minutes` and now passes it through the clamp via `resolve_wall_clock_deadline`.

## Quality gate

See the latest validation results on the PR head.

The Develop merge retains version 0.74.330 in both the manifest and lockfile,
above Develop's 0.74.329.
