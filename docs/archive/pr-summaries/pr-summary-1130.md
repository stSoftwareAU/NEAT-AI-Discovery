## Summary

Added a target-neuron cooldown so discovery stops wasting budget probing a
target that has just failed three times in a row. In the GRQ-sampler commit
`4c4fbdc560ad6b3070c5c48613ea1393aee2f225`, 17 of 18 `add-neurons` failure
cache entries all pointed at the same target (`neuron-1063112866`) — this
change parks such targets for a window of epochs while their neighbours get a
chance to surface. Closes #1130.

### What changed

- New module `src/analysis/target_failure_tracker.rs` with `TargetFailureTracker`,
  `filter_cooldown_targets`, and a process-global `global_tracker()`
  singleton. Tracks per-target consecutive failures, last-failure epoch, and
  last-improvement epoch; supports both explicit-epoch and internal-tick APIs.
- New constants `TARGET_COOLDOWN_CONSECUTIVE_FAILURES = 3` and
  `TARGET_COOLDOWN_EPOCHS = 10` in `src/analysis/constants/detection_thresholds.rs`.
- Env-var overrides `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_FAILURES` and
  `NEAT_AI_DISCOVERY_TARGET_COOLDOWN_EPOCHS` via `src/config/detection.rs`,
  documented in `src/config/mod.rs` following the existing table style.
- Neuron preparation (`src/analysis/neuron/preparation.rs`) and synapse
  orchestration (`src/analysis/synapse/orchestration.rs`) now apply
  `filter_cooldown_targets` after focus filtering and emit a
  `cooldown_skipped` diagnostic count (reason-name matching Issue #1129's
  convention). A success recorded against a target resets the streak and
  immediately clears cooldown.

### Evidence

CLI/library change — no UI to screenshot. Verification:

- `cargo test --test issue_1130_target_cooldown`: 6 integration tests pass
  (three-failure cooldown, below-threshold no-op, window release, success
  reset, default-threshold sanity check, GRQ-sampler budget pattern).
- `cargo test --lib target_failure_tracker`: 8 unit tests pass covering the
  three transitions called out in the acceptance criteria.
- `./quality.sh` runs green (fmt, clippy, check, full test suite, doc build,
  release build).

### Test Plan

- [x] Unit tests for `TargetFailureTracker` in
      `src/analysis/target_failure_tracker.rs` covering failure increment,
      threshold-triggered cooldown, and success-clears-cooldown transitions.
- [x] Unit tests for `filter_cooldown_targets` covering selective skipping
      and epoch-window release.
- [x] Integration tests in `tests/issue_1130_target_cooldown.rs` exercising
      the public API, including a replay of the GRQ-sampler 18-attempt
      pattern that confirms cooldown saves budget after the third failure.
- [x] `./quality.sh < /dev/null` passes cleanly.
