## Summary

Detection-threshold overrides now accept only finite positive floats and use
the caller's detector default for all other parsed values. Regression coverage
proves that NaN no longer emits a noisy-neuron candidate below the default
threshold. Closes #2438.

Version bump: `Cargo.toml` updated for this change; the merged commit records the
final version.

## Spec

### Intent and Rationale

- Reject malformed threshold values at the three accessors that own these
  settings, keeping the shared `parse_env` behaviour unchanged for other
  environment readers.

### Essential Design Decisions

- Preserve each accessor's signature and caller-supplied default. An override
  is used only when it is finite and greater than zero.

### Undiscoverable Facts

- The failing-first run against unchanged production code returned NaN from all
  three accessors and emitted a noisy-neuron candidate under a NaN override.

## Evidence

Backend-only change; no visual surface changed. The new tests call the public
accessors and `detect_noisy_neurons` with a deterministic 100-record fixture.
The original trigger is closed: every parsed non-finite or non-positive value
fails the same finite-positive predicate before the caller fallback is used, so
equivalent spellings and overflow to infinity cannot bypass the validation.

## Test Plan

- Added `tests/issue_2438_detection_threshold_hostile_values.rs::noise_signal_threshold_rejects_non_finite_and_non_positive_overrides`.
- Added `tests/issue_2438_detection_threshold_hostile_values.rs::dominance_threshold_rejects_non_finite_and_non_positive_overrides`.
- Added `tests/issue_2438_detection_threshold_hostile_values.rs::gradient_threshold_rejects_non_finite_and_non_positive_overrides`.
- Added `tests/issue_2438_detection_threshold_hostile_values.rs::noisy_neuron_detection_uses_default_when_threshold_is_nan`.
- Added caller-fallback checks for NaN, positive and negative infinity, zero,
  negative zero, negative values, overflow, empty and malformed values, and
  unset variables. The three accessor checks passed on the final implementation.
- The detector regression failed before the fix: the NaN override produced a
  candidate. The positive control at threshold `0.1` emitted one candidate;
  after the fix the NaN case returned no candidates.
- Guard-removal checks for each accessor: removing `is_finite()` made the
  corresponding hostile-value test fail on `inf`; removing `> 0.0` made it
  fail on `0`.
- Edited-expectation red checks: with only the named accessor's validation
  removed, each selector `noise_signal_threshold_`, `dominance_threshold_` and
  `gradient_threshold_` ran all three updated #2272 tests and all three failed
  on NaN, infinity and `-1`, respectively.
- `tests/issue_2272_config_small_modules.rs`: all nine changed assertions now
  require the caller fallback, and the 31-test binary passed.
- `tests/issue_2006_numeric_env_override_trim.rs`: the 14-test binary passed,
  including whitespace and malformed-value fallback checks.
- `cargo fmt --all -- --check`: passed. `cargo clippy --all-targets
  --all-features -- -D warnings`: passed.
- `timeout 900 ./quality.sh < /dev/null`: timed out with exit 124 during the
  1,753-test library unit suite. Dependency checks, debug build, formatting,
  Clippy, type checking and test-target compilation completed; the full test
  suite did not complete.

Assertions replaced in `tests/issue_2272_config_small_modules.rs`:

- Removed from `tests/issue_2272_config_small_modules.rs`:
  `value.is_nan()` in `noise_signal_threshold_nan_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `f32::INFINITY` in `noise_signal_threshold_infinity_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `value, -1.0` in `noise_signal_threshold_negative_one_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `value.is_nan()` in `dominance_threshold_nan_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `f32::INFINITY` in `dominance_threshold_infinity_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `value, -1.0` in `dominance_threshold_negative_one_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `value.is_nan()` in `gradient_threshold_nan_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `f32::INFINITY` in `gradient_threshold_infinity_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.
- Removed from `tests/issue_2272_config_small_modules.rs`:
  `value, -1.0` in `gradient_threshold_negative_one_uses_fallback`.
  Required by: "NaN, inf, -inf, zero or a negative number included".
  Replaced by: `assert_eq!(value, 0.5, ...)`.

Documentation sweep: read and updated the `Detection thresholds` section in
`docs/CONFIGURATION.md` and all three accessor comments in
`src/config/detection.rs`. Updated the stale Issue #2438 commentary in
`tests/issue_2272_config_small_modules.rs`. A final search of those updated files
found no remaining claims that these overrides pass hostile values through.
The historical audit in `docs/audits/security-sweep-chunk-13-config-env.md`
still records the pre-fix pass-through behaviour and the replaced test names
in its three detection-threshold rows; those rows describe the audited baseline,
not the current implementation.
