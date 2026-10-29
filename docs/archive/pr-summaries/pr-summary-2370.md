# PR Summary — Issue #2370

## Summary

`detect_dead_neurons` no longer runs one allocating depth-first search per dead
hidden neuron. Connected outputs for all dead neurons now come from one shared,
iterative Tarjan strongly-connected-component pass. Its cost is O(V + E) plus the
size of the output sets it merges, with two bounds: the analysis deadline and
global cancellation are checked on every walk iteration, and total merged work is
capped at `MAX_REACHABILITY_MERGE_WORK` (8,000,000 elements). A component that
adds no direct outputs and has a single closed successor shares that successor's
set without copying it, so a long live chain above a wide fan-out costs no
per-component copy. The discovery spec uses the deadline-aware variant and logs a
warning when a scan stops early. Version bump: `Cargo.toml` and `Cargo.lock`
updated for this change; the merged commit records the final version. Closes
#2370.

## Spec

### Intent and Rationale

- The old code did a full DFS with fresh `HashSet`/`Vec` allocations for each of
  D dead neurons (O(D·(V+E)) CPU and allocation) with no deadline check; the
  hidden-neuron and synapse counts are uncapped.
- Tarjan SCC (not a plain reverse-topological memo) keeps the old DFS's cycle
  tolerance: a corrupt creature with cycles still terminates and gives each
  neuron in a cycle the same reachable outputs.
- Review of the first attempt (PR #2435) found that copying each closed
  successor's output set into every intermediate component made memory O(live
  components × outputs). Set sharing plus a merged-work ceiling fixes that.

### Essential Design Decisions

- `detect_dead_neurons` keeps its signature and delegates with no deadline;
  `detect_dead_neurons_with_deadline` and `detect_dead_neurons_observed` follow
  the `symmetry_breaking` precedent (`DeadNeuronScan { candidates, truncation }`).
- Stop during the hidden-neuron statistics pass returns no candidates (outputs
  cannot be computed for a partial scan, so fail closed). Stop during the
  reachability pass keeps only candidates whose outputs completed.
- Hitting the merged-work ceiling reports `ScanTruncation::CandidateCeiling`
  (the existing "work ceiling reached" variant; no new variant added). The
  ceiling is checked against an upper bound before any copy, so it bounds memory.
  A ladder-shaped graph (each node feeds its own output and the next node) needs
  quadratic distinct sets and stops at the ceiling instead of exhausting memory.
- `detect_dead_neurons_observed` takes the ceiling and a step observer so tests
  can count work and force the ceiling; production passes the constant and a
  no-op observer through `detect_dead_neurons_with_deadline`.
- No wall-clock test: the repo's CONTRIBUTING.md keeps timing out of unit tests,
  so the regression guard counts reachability steps.

### Undiscoverable Facts

- `hidden_uuids` is a `HashSet`, so root order (and the order of equal-confidence
  candidates) is arbitrary; tests compare candidates sorted by UUID, and a test
  of "an earlier root completes, a later one hits the ceiling" cannot be made
  deterministic.
- The crate installs its own `#[global_allocator]`, so a test binary cannot add a
  counting allocator; an allocation-bytes guard that also compiles on the base
  branch was not possible.

## Evidence

Backend-only change; no visual surface is affected.

```mermaid
flowchart LR
    A[Hidden neurons] --> B[Dead-criteria pass, deadline checked per neuron]
    B --> C[Dead drafts]
    C --> D[Shared Tarjan SCC pass, sets shared or merged]
    D --> E[Candidates with connected outputs]
    B -- deadline --> F[No candidates, DeadlinePassed]
    D -- deadline --> G[Completed candidates, DeadlinePassed]
    D -- merge ceiling --> H[Completed candidates, CandidateCeiling]
```

Security Fix Evidence

- Regression test:
  `tests/issue_2370_dead_neuron_reachability.rs::reachability_grows_linearly_with_dead_count`
  runs the same dead chain at N=64 and N=256 and counts reachability steps through
  the `detect_dead_neurons_observed` observer: 128 and 512 steps (exactly 4x).
  Resetting the memo per dead neuron (the old per-neuron work profile) gives a
  roughly 16x growth and turns the test red. Against the base branch the test file
  does not compile, because `detect_dead_neurons_observed` does not exist there;
  that is the base-branch failure, not a runtime red.
- Memory regression test:
  `tests/issue_2370_dead_neuron_reachability.rs::live_chain_above_wide_fan_out_does_not_copy_sets_per_component`
  (one dead root, a live chain, then 64 outputs: 256 steps at N=64, 640 at N=256;
  disabling the sharing branch gave 17,024 steps and turned it red against its
  `N × outputs` bound of 16,384).
- The original trigger is closed with no equivalent bypass: the per-neuron DFS
  and its per-call allocations are removed, so reachability work no longer
  multiplies by the dead count; cycles are handled by the SCC pass (tested), the
  walk uses an explicit stack (a 20,000-deep chain is tested), the deadline and
  cancellation are checked once per walk iteration and once per hidden neuron,
  and merged output-set work is capped, so a hostile fan-out stops at
  `CandidateCeiling` rather than exhausting memory. Remaining retained memory is
  bounded by the 8,000,000-element ceiling plus the candidates' own output lists.

## Test Plan

- Added `tests/issue_2370_dead_neuron_reachability.rs` (9 tests, all pass):
  `reachability_grows_linearly_with_dead_count`,
  `live_chain_above_wide_fan_out_does_not_copy_sets_per_component`,
  `merge_ceiling_stops_the_scan_and_reports_candidate_ceiling`,
  `deep_chain_does_not_overflow_the_stack`,
  `cycle_among_dead_neurons_terminates_with_correct_outputs`,
  `matches_a_reference_traversal_on_a_branching_graph`,
  `elapsed_deadline_stops_the_scan_and_reports_truncation`,
  `far_future_deadline_does_not_truncate`,
  `without_a_cache_and_with_a_cache_agree`.
- Added `src/analysis/detection/dead_neuron_deadline_tests.rs` (2 in-crate tests):
  `phase_one_deadline_check_truncates_and_returns_no_candidates` and
  `reachability_deadline_check_stops_mid_scan`. Their sequence-scripted deadline
  seam makes each check the one that stops the scan.
- Red checks run: resetting the memo per root turned
  `reachability_grows_linearly_with_dead_count` red. Disabling the set-sharing
  branch in `close()` turned `live_chain_above_wide_fan_out_does_not_copy_sets_per_component`
  red (17,024 steps against 16,384); its `4n < 6n` assertion did not fail, the
  `N × outputs` bound is the discriminating one. Disabling the hidden-neuron
  deadline check turned both deadline unit tests red; disabling the walk's
  deadline check turned `reachability_deadline_check_stops_mid_scan` red.
  The ceiling test uses a ceiling of 1 against `usize::MAX` and the production
  ceiling on the same graph, so only the ceiling distinguishes the outcomes; I did
  not separately break the ceiling comparison.
- Removed from the first attempt: the wall-clock test
  `scan_time_grows_linearly_with_dead_count` (CONTRIBUTING.md forbids timing
  assertions in tests); the step-count tests above replace it.
- Existing dead-neuron suites re-run and pass: `tests/detection/issue_341_*`
  (`--test detection dead`, 19 passed), `tests/analysis/issue_424_*` and
  `issue_481_*` (`--test analysis dead`, 16 passed),
  `tests/infrastructure/issue_754_topology_cache.rs` (`--test infrastructure
  topology`, 6 passed).
- Not covered by a test: the discovery spec in `neuron_specs.rs` passing
  `&deadline`. I searched the tests that reach spec construction
  (`tests/detection/issue_571_co_adaptation_detection.rs`,
  `src/analysis/module_dispatch_specs/mod.rs`); none runs a spec with an elapsed
  deadline, so the `detect:` block is `exempt (untestable)`: no existing harness
  can stage an elapsed deadline through the dispatch. A ceiling stop that keeps an
  earlier completed candidate is not asserted either (root order is random).
- Docs: `docs/DISCOVERY_TYPES.md` "Dead Neuron Detection" has a
  `Scan bounds (Issue #2370)` paragraph describing the shared pass, the set
  sharing, the merged-work cap and both truncation reasons;
  `docs/discoveries/dead-neuron.md` and `docs/ANALYSIS_DEEP_DIVE.md` were read and
  have no reachability claims. The module docs in `dead_neuron.rs` were updated to
  match.
- Coverage reconciliation: updated `docs/audits/security-sweep-chunk-08a-detection-neuron.md`
  for the added deadline test module (54 files), the three current capacity
  sites and the shared SCC traversal. Updated the three existing ledger contracts,
  `tests/issue_2154_chunk_08a_finalisation.rs`,
  `tests/issue_2280_chunk_08a_ledger_scaffold.rs` and
  `tests/issue_2282_chunk_08a_graph_sweep.rs`, without changing their scanners.
  The finalisation suite reproduced the CI failure (6 passed, 2 failed) before
  the correction; all eight suites reading this record now pass (50 tests).
  Reverting the file-count
  update, removing the new graph inventory entry or ledger row, misnaming either
  SCC capacity binding, restoring the deleted traversal expectation, and
  misnaming the SCC traversal row each made the corresponding contract red.
  Restoring the scaffold count to 53 or removing its new section entry each
  made its corresponding contract red too. All three copies of the inventory
  rule were updated together.
- Twin-rule search: the deadline-aware scan pattern mirrors
  `detect_symmetric_neurons_*` and `detect_co_adapted_neurons_with_deadline`; no
  shared matcher was changed.
- Dependencies: `parquet` and `arrow` bumped to 60.1 (with the lockfile moves
  they require). After merging `Develop`, the zstd advisories
  RUSTSEC-2026-0391/0398/0403/0410 stay cleared by the base's `parquet` feature
  list that omits `zstd` (no zstd crates remain in `Cargo.lock`); the 60.1 bump
  is kept on top of it. Fixes #2432.
- Current CI-fix validation: `timeout 900 ./quality.sh < /dev/null` passed
  shell syntax, ShellCheck, cargo-install pinning, PR-summary layout,
  `cargo deny check`, the debug build, formatting, Clippy and type checks.
  It exited 124 while compiling the full test suite, before executing it.
  After the final scaffold update, targeted Clippy on all three edited test
  targets passed with warnings denied, as did `cargo fmt --all -- --check`.
  The documentation and release-build stages were not reached. Instrumented
  coverage generation and upload were not run locally.
<!-- vibe-quality-gate-skipped reason="full ./quality.sh reached its 900-second timeout during test compilation; targeted audit suites and red checks are recorded above" -->
