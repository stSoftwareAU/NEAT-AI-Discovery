## Summary

Restored the `Develop` build by adding the adaptive target-cooldown API that
PR #1209 referenced but never committed. The call sites in
`src/analysis/neuron/preparation.rs` and `src/analysis/synapse/orchestration.rs`
imported `filter_cooldown_targets_adaptive` from `target_failure_tracker`, but
the function — together with `is_in_cooldown_adaptive`,
`effective_cooldown_epochs`, and `effective_consecutive_failures` — was missing
from the module, leaving `cargo check` failing with E0432 fleet-wide.

This change adds the four missing public symbols with the exact signatures the
existing call sites and the Issue #1204 test suite already expect, and wires
the previously-orphaned integration tests
(`tests/analysis/issue_1204_adaptive_target_cooldown.rs`) into the
`tests/analysis/main.rs` module tree so they actually run. No call sites or
other behaviour changed.

Closes #1212.

## Evidence

CLI/library change — no UI surface. Verified by:

- `cargo check --lib --tests --all-features` now succeeds (previously failed
  with the two E0432 errors from the issue).
- `./quality.sh < /dev/null` passes end-to-end (fmt, clippy `-D warnings`,
  check, full test suite, doc build, release build).
- All 15 Issue #1204 integration tests in
  `tests/analysis/issue_1204_adaptive_target_cooldown.rs` now compile, run,
  and pass — including the acceptance criterion that a target with 3 failures
  at epoch 0 is in cooldown at epoch 15 under `Normal` mode but cleared under
  `Conservative` mode.
- The pre-existing 13 unit tests in `target_failure_tracker.rs` still pass.

### Adaptive cooldown decision flow

```mermaid
flowchart TD
    A[focus_order, mode, drought_failures] --> B{drought_failures >=<br/>CONSERVATIVE_MODE_MAX_EPOCHS?}
    B -- yes --> C[Extended drought:<br/>÷ extended divisor (default 4)<br/>trigger + 2]
    B -- no --> D{mode == Conservative?}
    D -- yes --> E[Conservative:<br/>÷ conservative divisor (default 2)<br/>trigger + 1]
    D -- no --> F[Normal:<br/>÷ 1, trigger unchanged]
    C --> G[Clamp effective window to<br/>>= COOLDOWN_EPOCHS_FLOOR (2)]
    E --> G
    F --> G
    G --> H[is_in_cooldown_adaptive uses<br/>effective trigger + window]
    H --> I[filter_cooldown_targets_adaptive<br/>drops only effective-cooldown targets]
```

## Test Plan

- `cargo test --test analysis --all-features issue_1204 -- --test-threads=2`
  — 15/15 tests pass, covering: three regimes for `effective_cooldown_epochs`,
  three regimes for `effective_consecutive_failures`, env-var overrides for
  both divisors (including unparsable and `0` clamped to floor),
  saturating-add on `u32::MAX`, and integration via
  `filter_cooldown_targets_adaptive` / `is_in_cooldown_adaptive`.
- `cargo test --lib --all-features target_failure_tracker -- --test-threads=2`
  — 13/13 existing tracker unit tests still pass.
- `./quality.sh < /dev/null` — full quality gate passes.
