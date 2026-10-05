# PR Summary — Issue #2347

## Summary

`weight_coherence.rs::detect_symmetric_cancellation` ran an `i < j` pair loop
over every target's fan-in. For each opposite-sign, similar-magnitude pair it
called `calculate_correlation`, which rebuilt two `obs_index → activation`
`HashMap`s from both sources' full record slices. The cost was
`O(Σ_targets k² · S)`. Nothing capped the fan-in `k` or the number of emitted
candidates, and the scan never checked the deadline. A hostile creature with
one wide-fan-in target and alternating `+w` / `-w` weights could pin a core for
as long as it liked (CWE-834).

The scan is now bounded in three ways, and the per-pair rebuild is gone:

- **Fan-in cap.** A target whose weighted fan-in exceeds
  `MAX_FANIN_FOR_CANCELLATION_SCAN` (256) is skipped before any pair work and
  counted in `skipped_high_fanin_targets`.
- **Candidate ceiling.** The scan stops after
  `MAX_SYMMETRIC_CANCELLATION_CANDIDATES` (1024) candidates with
  `ScanTruncation::CandidateCeiling`.
- **Deadline.** `deadline_passed` (which also reports host cancellation) is
  checked before each target. An elapsed deadline stops the scan with
  `ScanTruncation::DeadlinePassed`.
- **Map cache.** Each source's activation map is built at most once per scan
  and cached across pairs and targets.

The new `detect_symmetric_cancellation_with_deadline` returns a
`SymmetricCancellationScan` holding the candidates, the truncation reason and
deterministic work counters. The old `detect_symmetric_cancellation` is now a
thin no-deadline wrapper. Production dispatch (`synapse_specs.rs`) forwards the
discovery deadline and logs a `warn!` when the scan was truncated or skipped
targets, so a partial result is never shown as complete.

Closes #2347

## Spec

### Intent and Rationale

- The pair scan must finish in bounded time and memory whatever the creature
  looks like, and must stop when the analysis deadline passes or the host
  cancels.
- The fix follows the earlier pairwise fixes (#2161 / #2169 / #2190) and reuses
  the existing `ScanTruncation` enum and `deadline_passed` helper.

### Essential Design Decisions

- Wide targets are skipped outright, not truncated to their first 256 sources.
  A partial pair set would depend on fan-in order and look complete when it is
  not.
- Targets are visited in sorted UUID order, so candidate-ceiling truncation is
  deterministic.
- Work is measured with counters (`pairs_correlated`, `activation_maps_built`),
  not wall-clock time, so the tests are deterministic (precedent: #2320).
- `correlation_from_maps` keeps the exact semantics of the old
  `calculate_correlation`. Non-finite activations are still filtered, and the
  shared-sample minimum is still enforced.

### Undiscoverable Facts

- Before this change, cancellation was module-level only.
  `discovery_dispatch.rs` checks the deadline before a module's closure starts,
  and `max_candidates` truncates its output afterwards. Neither bounds the work
  while the closure is running.

## Evidence

This is a backend-only change, with no visual surface and so no screenshots.

**Security fix evidence.** I added the regression test
`tests/issue_2347_symmetric_cancellation_growth_test.rs::symmetric_cancellation_work_grows_no_faster_than_the_cap_allows`.
It reproduces the original trigger: one target fed by N and then 4N sources
(N = 128) with alternating `+5.0` / `-5.0` weights and perfectly correlated
activations, so every cross-parity pair reaches the correlation step. The test
asserts that the over-cap target is skipped and that `pairs_correlated` grows
no more than 4× when fan-in grows 4×. It fails against the unfixed code and
passes after the fix:

- Against the unfixed code it does not compile, because the bounded scan entry
  point does not exist there.
- With the fan-in cap disabled on the fixed code, it fails with
  `the over-cap target must be skipped entirely, not partially scanned`
  (`left: 0, right: 1`), because the work grows quadratically.

The production entry point is covered too:

- `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::symmetric_cancellation_spec_returns_none_when_deadline_already_elapsed`
  proves that dispatch forwards the discovery deadline.
- `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::scan_is_partial_when_high_fanin_targets_skipped`,
  together with its siblings, proves that a partial scan is reported and not
  shown as complete.

These tests cover the other bounds:

- `tests/issue_2347_symmetric_cancellation_growth_test.rs::activation_maps_are_cached_across_pairs_not_rebuilt_per_pair`
  reproduces the per-pair map rebuild. With the rebuild still in place it
  failed with `200 maps for 20 sources`, and it passes after the fix.
- `tests/issue_2347_symmetric_cancellation_growth_test.rs::fanin_at_the_cap_is_scanned_fanin_over_the_cap_is_skipped`
  pins the cap boundary.
- `tests/issue_2347_symmetric_cancellation_growth_test.rs::elapsed_deadline_stops_the_scan_before_any_pair_work`
  pins the deadline check.
- `tests/issue_2347_symmetric_cancellation_growth_test.rs::symmetric_cancellation_stops_at_the_candidate_ceiling_and_reports_it`
  pins the candidate ceiling.

**The original trigger is closed, with no trivial bypass.** There is one pair
loop. Every way into it passes the per-target deadline check and then the
fan-in cap, which runs before any pair work or map build. So per-target pair
work is at most `256²/2` correlations, and map builds are at most one per
distinct source. The candidate ceiling bounds the emitted `Vec`. The legacy
wrapper calls the same bounded function, so it applies the same fan-in and
candidate ceilings. Spreading fan-in over many targets just under the cap
leaves every target bounded, the total bounded by the deadline, and the output
bounded by the ceiling.

```mermaid
flowchart TD
    T[next target, sorted UUID] --> D{deadline passed or cancelled}
    D -- yes --> X[stop, DeadlinePassed]
    D -- no --> F{weighted fan-in above 256}
    F -- yes --> S[skip, count it] --> T
    F -- no --> P[pair scan, cached source maps]
    P --> C{candidates at 1024}
    C -- yes --> Y[stop, CandidateCeiling]
    C -- no --> T
```

**Docs sweep** — grep: `detect_symmetric_cancellation`, `MAX_FANIN_FOR_CANCELLATION_SCAN`, `MAX_SYMMETRIC_CANCELLATION_CANDIDATES`, `calculate_correlation`, `SymmetricCancellationScan`; section: `docs/discoveries/weight-coherence.md#-3-symmetric-cancellation`, `docs/DISCOVERY_TYPES.md` (Detection criteria — Symmetric cancellation), `docs/ANALYSIS_DEEP_DIVE.md` (Symmetric cancellation); updated: `docs/discoveries/weight-coherence.md`, `docs/DISCOVERY_TYPES.md`, `docs/ANALYSIS_DEEP_DIVE.md`

Every remaining hit on the head:

- `docs/DISCOVERY_TYPES.md:990`, `docs/DISCOVERY_TYPES.md:992` — still true because this diff wrote these lines to describe the fan-in cap, the candidate ceiling and the deadline.
- `docs/ANALYSIS_DEEP_DIVE.md:607`, `docs/ANALYSIS_DEEP_DIVE.md:608` — still true because this diff wrote these lines to describe the bounded scan.
- `docs/discoveries/weight-coherence.md:60`, `docs/discoveries/weight-coherence.md:62` — still true because this diff wrote these lines to describe the bounds and the skip count.
- `src/analysis/detection/weight_coherence.rs:27`, `:31`, `:33` — still true because they are the new module doc for the scan bounds.
- `src/analysis/detection/weight_coherence.rs:74`, `:78` — still true because they define the two constants.
- `src/analysis/detection/weight_coherence.rs:178`, `:185` — still true because they are the new result struct and its `skipped_high_fanin_targets` doc.
- `src/analysis/detection/weight_coherence.rs:406`, `:409`, `:419`, `:425` — still true because the legacy wrapper still exists, still calls the bounded scan with no deadline, and its doc says production uses the deadline entry point.
- `src/analysis/detection/weight_coherence.rs:434`, `:446`, `:448`, `:454`, `:455` — still true because they are the new bounded function's doc and signature.
- `src/analysis/detection/weight_coherence.rs:507`, `:577` — still true because they are the cap and ceiling checks.
- `src/analysis/detection/weight_coherence.rs:729` — still true because the `correlation_from_maps` doc states, as history, that the removed `calculate_correlation` rebuilt both maps on every call.
- `src/analysis/module_dispatch_specs/synapse_specs.rs:31`, `:49`, `:54`, `:167`, `:291`, `:292` — still true because they are the production call into the bounded scan, the partial-scan helper and its doc, and the test fixtures for it.
- `tests/detection/issue_437_weight_coherence_validation.rs:22`, `:289`, `:365` — still true because they call the legacy wrapper, which keeps its signature and results.
- `tests/detection/issue_770_weight_coherence_topology_cache.rs:11`, `:158`, `:184`, `:185` — still true because the wrapper still honours the topology cache.
- `tests/infrastructure/issue_776_hashset_hashmap_iteration.rs:9`, `:98`, `:125`, `:163` — still true because the wrapper is kept and still produces correct results.
- `tests/issue_2347_symmetric_cancellation_growth_test.rs` (lines 2, 14, 15, 80, 84, 97, 120, 121, 137, 138, 166, 191, 204, 215, 237, 240, 243, 252, 255, 268, 270) — still true because this diff adds the file, and its hits are the module doc, imports and calls to the bounded scan.

## Test Plan

- **Tests added:**
  - `tests/issue_2347_symmetric_cancellation_growth_test.rs`, with 9 tests.
  - In `src/analysis/module_dispatch_specs/synapse_specs.rs::tests`:
    - `symmetric_cancellation_spec_returns_none_when_deadline_already_elapsed`
    - `symmetric_cancellation_spec_returns_candidates_without_deadline`
    - `scan_is_not_partial_when_complete`
    - `scan_is_partial_when_deadline_passed`
    - `scan_is_partial_when_candidate_ceiling_hit`
    - `scan_is_partial_when_high_fanin_targets_skipped`
- **Tests modified:** the three unit tests in `src/analysis/detection/weight_coherence.rs` that called the removed `calculate_correlation` now call `correlation_from_maps(&build_activation_map(..), &build_activation_map(..), 10)`. Only the call site changed.
- **Removed assertions:** none. Every existing assertion is kept unchanged.
- **Results:**
  - `cargo test --test issue_2347_symmetric_cancellation_growth_test` → `test result: ok. 9 passed; 0 failed`
  - `cargo test --lib module_dispatch_specs` → `test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 1626 filtered out`
  - `./quality.sh < /dev/null` → QUALITY_RESULT

**Branch outcomes:**
- `src/analysis/detection/weight_coherence.rs:489` — deadline elapsed (stop, `DeadlinePassed`) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::elapsed_deadline_stops_the_scan_before_any_pair_work` — flipped to never stop, test went red
- `src/analysis/detection/weight_coherence.rs:489` — no deadline, or deadline not reached (scan continues, no truncation) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::no_deadline_does_not_truncate_and_finds_candidates`, `tests/issue_2347_symmetric_cancellation_growth_test.rs::far_future_deadline_behaves_like_no_deadline` — flipped to always stop, both tests went red
- `src/analysis/detection/weight_coherence.rs:507` — fan-in over the cap (target skipped and counted) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::fanin_at_the_cap_is_scanned_fanin_over_the_cap_is_skipped`, `tests/issue_2347_symmetric_cancellation_growth_test.rs::symmetric_cancellation_work_grows_no_faster_than_the_cap_allows` — flipped to never skip, both tests went red
- `src/analysis/detection/weight_coherence.rs:507` — fan-in exactly at the cap (target scanned) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::fanin_at_the_cap_is_scanned_fanin_over_the_cap_is_skipped` — flipped `>` to `>=`, test went red
- `src/analysis/detection/weight_coherence.rs:543` — source map not yet cached (build it once and count it) vs. already cached (reuse it) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::activation_maps_are_cached_across_pairs_not_rebuilt_per_pair` — flipped to rebuild on every pair, test went red
- `src/analysis/detection/weight_coherence.rs:577` — candidate ceiling reached (stop, `CandidateCeiling`) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::symmetric_cancellation_stops_at_the_candidate_ceiling_and_reports_it` — flipped to never stop, test went red
- `src/analysis/detection/weight_coherence.rs:577` — below the ceiling (scan completes, no truncation) — `tests/issue_2347_symmetric_cancellation_growth_test.rs::candidate_ceiling_not_reported_when_the_scan_completes` — flipped to always stop, test went red
- `src/analysis/module_dispatch_specs/synapse_specs.rs:38` / `:56` — partial scan (truncated or skipped targets, so `warn!` is logged) — `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::scan_is_partial_when_deadline_passed`, `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::scan_is_partial_when_candidate_ceiling_hit`, `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::scan_is_partial_when_high_fanin_targets_skipped` — forced `symmetric_cancellation_scan_is_partial` to always return false, all three tests went red
- `src/analysis/module_dispatch_specs/synapse_specs.rs:38` / `:56` — complete scan (no log) — `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::scan_is_not_partial_when_complete` — forced it to always return true, test went red
- `src/analysis/module_dispatch_specs/synapse_specs.rs:118` — production dispatch forwards the discovery deadline — `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::symmetric_cancellation_spec_returns_none_when_deadline_already_elapsed` (paired with `src/analysis/module_dispatch_specs/synapse_specs.rs::tests::symmetric_cancellation_spec_returns_candidates_without_deadline`) — reverted `&deadline` to `&None`, the elapsed-deadline test went red

🤖 Generated with [Claude Code](https://claude.com/claude-code)
