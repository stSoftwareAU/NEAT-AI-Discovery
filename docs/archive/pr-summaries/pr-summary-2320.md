# PR Summary — Issue #2320

## Summary

Closes #2320

The two scaling tests in `issue_2169_structural_patterns_cancellation_test.rs`
asserted `t_large <= t_small * 3` on wall-clock timings. `#[serial]` only
serialises against other `#[serial]` tests, so the ratio could fail under load
from the rest of the suite. Both now assert deterministic work counts using the
#2296 `*_observed` pattern:

- `detect_noisy_vs_trusted` delegates to a private
  `detect_noisy_vs_trusted_observed(..., on_pair: impl FnMut())`, which ticks
  once per pair visited in the `'pairs` scan.
- `detect_collapsible_hidden_neurons` delegates to a private
  `detect_collapsible_hidden_neurons_observed(..., on_shared_map_build: impl FnMut())`,
  which ticks each time a shared `a`/`b` observation map is built.
- The public functions pass `|| {}`, so production behaviour is unchanged.

The tests now check:

- **Noisy scan:** at 16 inputs, a positive precondition (Issue #1799) expects
  `16*15/2` pairs. Above `MAX_INCOMING_INPUTS_FOR_NOISY_SCAN`, and at double
  that, 0 pairs are scanned.
- **Collapse detector:** at (8 hidden, 64 records) and (16, 128), the shared
  maps are built exactly twice, so the count does not grow with the fixture.
  Every chain must also still collapse.

Version bumped `0.74.271` → `0.74.272`.

## Evidence

This change touches tests only, so the tests are the evidence.

- **Failing-first check:** removing the `!a_maps.contains_key(a)`
  memoisation guard makes the collapse test fail with
  `left: (9, 17) right: (2, 2)`. The old ratio test could only catch this if
  the timing happened to cross the threshold. The guard was then restored.
- `cargo test --lib issue_2169`: 7 passed. `cargo test --lib issue_2161`:
  2 passed.
- `./quality.sh`: all checks passed (library suite 1625 passed, 0 failed).

## Test Plan

- [x] Replace both wall-clock ratio tests with work counters and positive
      preconditions
- [x] Failing-first check: the regression test reproduces a de-memoised
      collapse detector
- [x] `cargo fmt`, `cargo clippy --all-targets --all-features -- -D warnings`
- [x] `timeout 900 ./quality.sh < /dev/null` passes
