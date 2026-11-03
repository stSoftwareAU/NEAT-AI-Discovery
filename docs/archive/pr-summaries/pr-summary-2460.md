## Summary

`cap_deadline_to_wall_clock` multiplied the caller-supplied minutes by 60 with an unchecked `u64` multiply, so a large `max_discovery_wall_clock_minutes` wrapped to a short cap in release builds and panicked in debug builds. The multiply is now checked (an unrepresentable cap yields no wall-clock deadline instead of a wrapped one), and `analyze_all` clamps the FFI value to 1–120 minutes through a new `config::effective_wall_clock_minutes`, reached via a small `resolve_wall_clock_deadline` helper (extracted from `analyze_all` so the wiring is testable without a GPU), logging a `WARN` when it changes the value. Version bump: `Cargo.toml` updated for this change; the merged commit records the final version. Closes #2460.

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
- `analyze_all` wiring: `tests/issue_2460_analyze_all_wall_clock_clamp.rs` drives `u64::MAX` and `307445734561825861` through `resolve_wall_clock_deadline` (the exact resolve-and-cap step `analyze_all` calls) and asserts the effective cap is 120 minutes. Red-checked: replacing the clamp in the helper with the raw `unwrap_or_else(max_wall_clock_minutes)` read makes all 3 tests fail.
- Against the base `src/analysis/utils/deadline.rs` (all else unchanged), `tests/issue_2460_wall_clock_cap_overflow.rs` fails 3 of 4 with `attempt to multiply with overflow` (`deadline.rs:215`); with the fix all 4 pass (the file compiles on the base: it only uses `cap_deadline_to_wall_clock`, which exists there). The one passing on base is the in-range boundary test.
- Original trigger closed: `307445734561825861` and `u64::MAX` are clamped to 120 by `effective_wall_clock_minutes` in `analyze_all` (the only reader of the field), and even if passed directly to `cap_deadline_to_wall_clock` the `checked_mul` yields `None` for the cap, so there is no wrap and no panic. No equivalent bypass exists: `cap_deadline_to_wall_clock`'s only production caller is `analyze_all`.
- Docs updated: `docs/CONFIGURATION.md`, `docs/FFI_API.md`, doc comments in `src/ffi_types/requests.rs`, `src/config/user_facing.rs`, `src/analysis/utils/deadline.rs`. Historical records in `docs/audits/security-sweep-chunk-13-config-env.md` left as written.

## Test Plan

- `tests/issue_2460_wall_clock_cap_overflow.rs::wrapping_multiplier_does_not_produce_short_cap` — red on base, green at head.
- `tests/issue_2460_wall_clock_cap_overflow.rs::u64_max_minutes_yields_no_wall_clock_deadline` — red on base, green at head.
- `tests/issue_2460_wall_clock_cap_overflow.rs::unrepresentable_cap_preserves_analysis_deadline` — red on base, green at head.
- `tests/issue_2460_wall_clock_cap_overflow.rs::boundary_minutes_still_cap_normally` — green on base and head (in-range guard).
- `tests/issue_2460_analyze_all_wall_clock_clamp.rs::oversized_ffi_minutes_are_clamped_to_max_cap` — `u64::MAX` and `307445734561825861` resolve to a 120 minute cap and deadline; fails if the clamp is removed from the `analyze_all` path.
- `tests/issue_2460_analyze_all_wall_clock_clamp.rs::oversized_ffi_minutes_still_respect_tighter_analysis_deadline` and `::zero_ffi_minutes_clamp_up_to_one_minute` — clamp interplay with a tighter analysis deadline, and the FFI `0` → 1 minute decision.
- `src/config/mod.rs::effective_wall_clock_minutes_clamps_ffi_value` — unit test for the clamp bounds and the `None` fallback (new function, so it lives in the crate and is not cited as a base/head discriminator).
- Existing `cap_deadline` unit tests and `issue_1098_wall_clock_cap` (8 tests) pass unchanged.
- Callers checked: `analyze_all` (src/analysis/orchestration.rs) is the only production reader of `max_discovery_wall_clock_minutes` and now passes it through the clamp via `resolve_wall_clock_deadline`.

## Quality gate

`./quality.sh < /dev/null` was started on the final code head but cut off by the 595 s foreground cap before it reported; no failure was seen. Targeted checks passed (rustfmt, clippy `-D warnings`, the new tests, `cap_deadline` unit tests, `issue_1098_wall_clock_cap`). CI runs the full gate.

<!-- vibe-quality-gate-skipped reason="full gate exceeds the 600 s foreground command cap; left to CI" -->
