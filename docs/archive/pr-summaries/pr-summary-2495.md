## Summary

Tests in the `tests/infrastructure` binary that call detectors observing the process-global cancellation flag are now `#[serial]`, so they can no longer run while a serial cancellation test holds the flag set. Closes #2495.

Version bump: `Cargo.toml` and `Cargo.lock` updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- `deadline_passed` (`src/analysis/utils/deadline.rs`, via `crate::cancellation::is_cancelled()`) returns `true` whenever the global flag is set. `#[serial]` only excludes other `#[serial]` tests, so an unmarked detector test could see the flag mid-run and return fewer candidates.
- The issue asks to mark `test_dead_neuron_cache_vs_no_cache` and audit the rest of the binary; the audit found the same gap in five more tests.

### Essential Design Decisions

- Marked every non-serial test in the binary that calls a detector reading `deadline_passed`: `detect_dead_neurons`, `detect_bottleneck_neurons`, `detect_restricted_range_neurons`. `detect_bounded_range_neurons` has no cancellation check (grep of `bounded_range.rs`), so `bounded_range_detection_preserves_correctness` is left parallel.
- Assumption: the audit covers detectors only, as the issue words it. GPU pipeline tests that call `analyze_synapses` (for example `issue_196`, `issue_832`) also reach cancellation-aware code; they skip without a GPU and are not changed here.

### Undiscoverable Facts

- None.

## Evidence

Test-only change; no UI. Marked tests:

- `tests/infrastructure/issue_754_topology_cache.rs`: `test_dead_neuron_cache_vs_no_cache`, `test_bottleneck_cache_vs_no_cache`, `test_bottleneck_conversion_cache_vs_no_cache`.
- `tests/infrastructure/issue_487_reduce_clone_allocations.rs`: `bottleneck_detection_with_borrowed_lists`, `restricted_range_detection_preserves_correctness`, `bottleneck_multiple_candidates_correct_uuids`.

## Test Plan

- `cargo test --test infrastructure -- issue_754 issue_487`: 13 passed, 0 failed.
- `cargo fmt --check` and `cargo check --tests --test infrastructure`: clean.
- No test was added: the change is a scheduling marker, and the race is timing-dependent, so a deterministic regression test would only grep source. No assertion was removed or edited. Not red-checked against the base for the same reason (the failure is an intermittent race, not reproduced).

<!-- vibe-quality-gate-skipped reason="full ./quality.sh exceeded the 590 s foreground cap while compiling (exit 124); fmt, check and the targeted tests above were run; CI runs the full gate" -->

Not done / risk: the full `./quality.sh` did not complete; the race itself was not reproduced; GPU pipeline tests are not audited.
