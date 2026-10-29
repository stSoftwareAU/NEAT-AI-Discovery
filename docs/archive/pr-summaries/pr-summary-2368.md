## Summary

`detect_skip_connection_candidates` no longer does uncancellable O(H·(I+H)) pair work, and its depth pass no longer re-pushes neurons or loops forever on a cyclic graph. Closes #2368.

- **Depth pass** (`compute_depths_from_inputs`): replaced the re-pushing longest-path BFS with a Kahn topological pass, O(V + E). A cycle leaves nodes unpopped, so the scan returns no candidates and reports `SkipConnectionStop::CycleDetected` instead of hanging.
- **Pair scan**: shallow sources are split once into depth-0 and depth-1 lists. Each deep target takes the first unconnected depth-0 source, else the first unconnected depth-1 source. Work per target is its existing in-edges plus one. A depth-0 source always gives a larger gap than a depth-1 source for a deep target (depth ≥ 3), and the first-in-creature-order tie-break is kept, so the chosen source equals the old scan's.
- **Deadline**: new `detect_skip_connection_candidates_with_deadline` checks `deadline_passed` before the depth pass, once per neuron popped, and once per deep target. `detect_skip_connection_candidates` stays as an unbounded wrapper. `detect_skip_connection_candidates_observed` adds an `on_step` work counter for deterministic tests.
- **Dispatch**: the skip-connection spec in `structural_specs.rs` now passes the analysis deadline and logs a `tracing::warn!` with the stop reason.
- **Docs**: `docs/DISCOVERY_TYPES.md` gains a "Cost and cancellation" paragraph. `docs/ANALYSIS_DEEP_DIVE.md` skip-connection algorithm steps are rewritten to match the code; they described a reverse BFS to outputs that the code never did.
- Version bump: `Cargo.toml` version updated for this change; the merged commit records the final version.

## Spec

### Intent and Rationale

- The issue's suggested fix: one topological pass (cycle reported, not looped on), an indexed shallow-source search, and the deadline threaded into the pair loop. This PR does those three.
- A work-count seam (`on_step`) is used instead of a timing ratio, because CONTRIBUTING forbids timing assertions in unit tests.

### Essential Design Decisions

- A stop during the depth pass returns no candidates (depths are incomplete); a stop in the target loop keeps the candidates already gathered. Both are surfaced via `SkipConnectionScan::stop`, never silent.
- A cycle returns an empty result with `CycleDetected` rather than best-effort depths, because no topological depth exists.
- Unreachable-from-input neurons stay absent from the depth map (the old code defaulted them to depth 0 at lookup, which is kept).

### Undiscoverable Facts

- None.

## Evidence

Backend only; no UI files touched. The cyclic-graph hang and the pair-scan cost are covered by deterministic tests (no wall-clock assertions).

Security-fix evidence:

- Regression tests: `tests/issue_2368_skip_connection_termination.rs::cyclic_graph_terminates_with_no_candidates`, `tests/issue_2368_skip_connection_termination.rs::long_chain_with_input_fan_out_terminates`.
- Both call only `detect_skip_connection_candidates`, which exists unchanged on the base, so they compile against base's non-test files. Run against the base they FAIL behaviourally: the old re-pushing depth BFS never returns on the cyclic graph, and needs about n²/2 pops on the 30,000-link chain with an input edge to every link, so the 20 s hang guard expires (observed: both failed after 20.07 s). At head both pass in 0.13 s.
- The guard is a hang guard on a worker thread, not a timing assertion: the budget is orders of magnitude above the new cost.
- The deadline and per-target work-count checks need the new API (`detect_skip_connection_candidates_with_deadline`, `detect_skip_connection_candidates_observed`), so they cannot run on the base and are not cited as discriminating. They remain in `tests/issue_2368_skip_connection_pair_scan.rs` and `src/analysis/skip_connection_deadline_tests.rs`.
- The old algorithm's pair-scan cost is covered by `acyclic_result_matches_legacy_algorithm` (`src/analysis/skip_connection_deadline_tests.rs`), which embeds the old depth BFS and pair scan and asserts the new result is identical on a DAG.
- ORIGINAL TRIGGER is closed with no trivial bypass: a cyclic creature handed to the public `analysis::detection::skip_connection` API is now detected by the Kahn pass (unpopped nodes) and returns immediately; no code path re-pushes a neuron, so no input makes the depth pass unbounded. The pair scan is bounded by in-edges + 1 per target, and every loop (depth pop, per target) checks the deadline and global cancellation.

## Test Plan

- Added `tests/issue_2368_skip_connection_termination.rs` (2 base-runnable hang-guard tests above) and `tests/issue_2368_skip_connection_pair_scan.rs` (3 tests, new API): step count at H=200 vs 800 is within 5× (linear, the old scan would be ~16×); cycle → `CycleDetected` with a no-back-edge control giving `stop None`; `UNIX_EPOCH` deadline → `DeadlinePassed` with an `&None` control yielding candidates.
- Added `src/analysis/skip_connection_deadline_tests.rs` (6 tests: deadline on first popped neuron, deadline at the per-target check keeps candidates so far, 2- and 3-node cycles, cycle downstream of a deep chain, DAG equivalence with the legacy algorithm) and `src/analysis/module_dispatch_specs/structural_specs_issue_2368_tests.rs::skip_connection_spec_stops_at_elapsed_deadline`.
- Audit record `docs/audits/security-sweep-chunk-08a-detection-neuron.md` updated for the new capacity sites and the Kahn traversal row; `issue_2154`, `issue_2280`, `issue_2282`, `issue_2294`, `issue_2088`, `issue_2281`, `issue_2284`, `issue_2298` and `issue_2300` audit tests pass. Known pre-existing mismatch left alone: the record's capacity grep counts 88 hits vs 82 rows, the gap being `correlated_error.rs`, unrelated to #2368.
- Existing `tests/recommendation/issue_570_skip_connection_discovery.rs` tests are unedited and exercise the unchanged wrapper.
- `cargo test --lib skip_connection`: 7 passed. `cargo test --test issue_2368_skip_connection_pair_scan`: 3 passed. `cargo clippy --all-targets -- -D warnings`: clean (run by the executor).
- No existing test expectation was edited; no assertion removed.
- `./quality.sh < /dev/null` on the final head: passed ("All quality checks passed!", exit 0; every `test result` line reports 0 failed).
- Not done: break-the-guard red checks per branch were not run individually.

## Merge verification

- `./quality.sh` result: passed on the final head (see Test Plan). CI and code review are pending.
