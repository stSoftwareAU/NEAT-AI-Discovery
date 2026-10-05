# PR Summary — Issue #2349

## Summary

`detect_symmetric_neurons` (`src/analysis/detection/symmetry_breaking.rs`)
rebuilt both incoming-weight vectors for every pair of eligible hidden neurons.
Each rebuild did a linear `find` per source, and the hidden-neuron membership
test was an `.any(..)` over the whole list for every synapse. The pair scan had
no deadline, no global-cancellation check and no cap on candidates, so a large
creature could pin a dispatch worker (CWE-407).

The scan is now bounded:

- hidden-neuron membership uses a `HashSet`;
- each eligible neuron's weight vector is built once, before the pair loop;
- the outer loop checks `deadline_passed` (which honours global cancellation)
  on every iteration;
- the candidate list stops at `MAX_SYMMETRIC_PAIR_CANDIDATES` (256).

A new `detect_symmetric_neurons_with_deadline` returns a `SymmetricPairScan`
that records why a scan stopped early and how many otherwise-eligible hidden
neurons were skipped by the cap. The production dispatch path
(`neuron_specs.rs`) uses it, forwards the dispatch deadline and logs a warning
on truncation. The legacy `detect_symmetric_neurons` keeps its signature and is
still capped.

Closes #2349

## Spec

### Intent and Rationale

- The symmetry-breaking scan must not be an uncancellable quadratic-times-linear
  hot loop. It must stop at the dispatch deadline or global cancellation.
- Work per pair drops to one cosine similarity over prebuilt vectors, so the
  cost grows with pairs, not with pairs × synapses.

### Essential Design Decisions

- The scan reuses the existing `ScanTruncation` type from
  `recommendation::epistatic`, but returns a new `SymmetricPairScan` result
  type (carrying `eligible_skipped` alongside `candidates` and `truncation`),
  rather than the generic `BoundedScan<T>` shape other bounded scans return —
  `BoundedScan<T>` has no field for a skipped-neuron count.
- `detect_symmetric_neurons_observed` takes an `FnMut()` hook that fires once
  per weight-vector build. This is the test seam that proves vectors are built
  once per neuron. Production passes a no-op.
- When a source has duplicate synapses into the same neuron, the first one in
  `creature.synapses` order still wins, as with the legacy `find`. The vector
  is filled in reverse so the first occurrence is written last.
- The deadline is checked once before the vector build, once per eligible
  neuron inside the build loop (bounding overshoot on a capped-but-still-large
  `A`), and once per outer iteration — never per pair. This bounds overshoot
  to one row of comparisons.

### Undiscoverable Facts

- `discovery_dispatch.rs` checks the deadline before it runs each spec closure.
  An already-elapsed deadline therefore never reaches the scan through dispatch,
  so the `neuron_specs.rs` truncation warning cannot be driven deterministically
  from a dispatch-level test.

## Evidence

This is a backend-only change, with no visual surface and so no screenshots.

**Security fix evidence.** The regression test
`tests/issue_2349_symmetry_breaking_growth.rs::symmetry_breaking_weight_vectors_are_built_once_per_neuron`
builds identical creatures with E and 4E eligible hidden neurons. It counts
weight-vector builds through the `_observed` hook and asserts that the count at
4E is at most 4 × the count at E.

- **Fails on the unfixed code.** On the unfixed code, vectors are rebuilt per
  pair, so the counts are 12 → 240. Reproducing that per-pair rebuild (two hook
  calls per pair) turned the test red (`16` vs `4`). The `_observed` hook does
  not exist on the base branch, so the test does not compile there either.
- **Passes after the fix.** The counts are 4 → 16, linear in neurons.

The original trigger is closed and has no trivial bypass:

- the production dispatch path calls the deadline variant with the dispatch
  deadline;
- the legacy deadline-free entry point is still capped at 256 candidates;
- vectors are built once each;
- membership is O(1).

Companion tests cover the elapsed-deadline and candidate-ceiling stops.

```mermaid
flowchart TD
    A[hidden neurons, HashSet membership] --> B{eligible >= 2}
    B -- no --> Z[empty, no truncation]
    B -- yes --> C{deadline passed}
    C -- yes --> D[empty, DeadlinePassed]
    C -- no --> E[build each weight vector once]
    E --> F[outer row i]
    F --> G{deadline passed}
    G -- yes --> H[partial, DeadlinePassed]
    G -- no --> I[compare i with j, push candidate]
    I --> J{candidates >= 256}
    J -- yes --> K[partial, CandidateCeiling]
    J -- no --> F
```

**Docs sweep** — grep: `symmetric_neurons`, `symmetry.breaking`, `MAX_SYMMETRIC`; section: `docs/discoveries/symmetry-breaking.md` (flowchart and "Bounded scan (Issue #2349)"); updated: `docs/discoveries/symmetry-breaking.md`, module doc in `src/analysis/detection/symmetry_breaking.rs`, `src/analysis/module_dispatch_specs/mod.rs` doc comment

- The grep scope was `*.md` and `*.rs` across the repository.
- Hits outside the diff:
  - `README.md:273` — still true because it is a module list entry.
  - `docs/ANALYSIS_DEEP_DIVE.md:677`, `:684` — still true because the scan
    still compares every pair of hidden neurons. It is now bounded, and the
    section makes no claim about cost or bounds.
  - `docs/DISCOVERY_TYPES.md:49`, `:198`, `:1296`, `:1298`, `:1331` — still true
    because they give detection criteria, operations and the source path, all
    unchanged.
  - `docs/PRIOR_ART.md:116`, `:243` — still true because they are prior-art
    citations.
  - `docs/discoveries/README.md:129` — still true because it is the index row.
  - `docs/discoveries/co-adaptation.md:144` — still true because it is a
    cross-link.
  - `src/analysis/detection/mod.rs:45` — still true because it is the module
    declaration.
  - `src/analysis/module_dispatch_specs/neuron_specs.rs:6`, `:15` — still true
    because they are a module doc and imports.
  - `src/analysis/detection/symmetry_breaking.rs:370`, `:399` — still true
    (shifted from `:313`/`:343` by the insertions above) because the candidate
    converter is unchanged.
  - `tests/detection/issue_569_symmetry_breaking.rs:16-17`, `:101`, `:167`,
    `:221` — still true because they call the legacy `detect_symmetric_neurons`,
    whose signature is unchanged.
  - `tests/integration.rs:645-652` — still true because it is a local variable
    name.
  - `docs/archive/pr-summaries/*` — still true because they are historical
    records.

## Test Plan

- `cargo test --test issue_2349_symmetry_breaking_growth < /dev/null`: 7 passed
  (two added for the eligible-neuron cap:
  `eligible_neurons_beyond_the_cap_are_skipped_and_recorded`,
  `under_cap_scan_records_no_skip`).
- `cargo test --test detection issue_569 < /dev/null`: 9 passed.
- `cargo clippy --all-targets --all-features -- -D warnings` and
  `cargo fmt --check`: clean.
- `timeout 900 ./quality.sh < /dev/null`: timed out (exit 124) while compiling
  the tests, after deny, build, fmt, clippy and check had passed.
- `timeout 1800 ./quality.sh < /dev/null`: GATE_RESULT_PENDING

Branch outcomes:

- `src/analysis/detection/symmetry_breaking.rs:173`, fewer than 2 hidden
  neurons gives empty and no truncation: an existing branch (refactored),
  reached by `tests/detection/issue_569_symmetry_breaking.rs` (the
  single-hidden-neuron case).
- `src/analysis/detection/symmetry_breaking.rs:194`, fewer than 2 eligible
  neurons gives empty and no truncation: an existing branch, reached by
  `tests/detection/issue_569_symmetry_breaking.rs` (the insufficient-samples
  case).
- `src/analysis/detection/symmetry_breaking.rs:205-208`, the eligible-neuron
  cap: otherwise-eligible hidden neurons beyond
  `MAX_SYMMETRY_ELIGIBLE_NEURONS` are counted into `eligible_skipped` and
  truncated out before the vector build:
  - skipped-and-recorded reached by
    `::eligible_neurons_beyond_the_cap_are_skipped_and_recorded`;
  - the under-cap (zero-skipped) outcome reached by
    `::under_cap_scan_records_no_skip`.
- `src/analysis/detection/symmetry_breaking.rs:210`, deadline passed before the
  build gives empty and `DeadlinePassed`:
  - reached by `tests/issue_2349_symmetry_breaking_growth.rs::symmetry_breaking_scan_stops_at_an_elapsed_deadline`;
  - removing the deadline checks went red (`None` vs `Some(DeadlinePassed)`).
- `src/analysis/detection/symmetry_breaking.rs:225`, membership hit or miss:
  - hit reached by every detection test;
  - miss (a synapse into a non-hidden neuron) reached by the growth-test
    creature's output synapses;
  - it is a pure lookup swap, with no behavioural flip.
- `src/analysis/detection/symmetry_breaking.rs:255`, the deadline check inside
  the per-neuron vector-build loop, gives empty and `DeadlinePassed`:
  - exercised the same way as the pre-build check at L210, since the test
    creatures build few enough vectors that both checks see the same
    deadline state;
  - the first duplicate-source synapse still wins at the write inside this
    loop (line 270): reached by
    `tests/issue_2349_symmetry_breaking_growth.rs::duplicate_source_synapse_keeps_first_weight`;
    making the last duplicate win went red (`0` vs `1`).
- `src/analysis/detection/symmetry_breaking.rs:284`, deadline passed in the
  outer loop gives partial and `DeadlinePassed`:
  - reached by `::symmetry_breaking_scan_stops_at_an_elapsed_deadline`;
  - removing the checks went red, as at L210;
  - the not-passed outcome is reached by
    `::symmetry_breaking_weight_vectors_are_built_once_per_neuron` with a
    far-future deadline.
- `src/analysis/detection/symmetry_breaking.rs:331`, ceiling reached gives
  partial and `CandidateCeiling`:
  - reached by `::symmetry_breaking_scan_stops_at_the_candidate_ceiling`;
  - removing the ceiling went red (`None` vs `Some(CandidateCeiling)`);
  - the under-ceiling outcome is reached by
    `::legacy_detect_symmetric_neurons_matches_deadline_free_scan`.
- Per-pair rebuild vs build-once:
  - reached by `::symmetry_breaking_weight_vectors_are_built_once_per_neuron`;
  - restoring a per-pair rebuild went red (`16` vs `4`).
- `src/analysis/module_dispatch_specs/neuron_specs.rs:200`, truncation gives a
  `warn!` (line 196 is now the doc comment above it, after the eligible-skipped
  warning block was added). There is no direct test: dispatch checks the
  deadline before running the closure, so an elapsed deadline cannot reach this
  branch deterministically. The truncation values it reads are covered by the
  scan tests above.

Entry points checked:

- `append_neuron_specs` and `module_dispatch_specs/mod.rs` now forward the
  dispatch deadline to the scan.
- Reverting to the deadline-free call is not observable through dispatch, for
  the same reason as the `neuron_specs.rs:200` branch above. This is noted, not
  hidden.

Removed assertions: none.

## Pre-PR Security Self-Check

- [x] Input validation: no new external input. Scan work is now bounded by a
      deadline and a candidate cap.
- [x] Secrets: none staged. Only source, tests, docs and this summary are
      staged.
- [x] Injection surface: there are no new SQL, shell, filesystem or HTTP calls.
- [x] Output encoding: not applicable.
- [x] Authentication and authorisation: not applicable.
- [x] Error handling: truncation is logged as a warning with its reason, and no
      internals are exposed.
- [x] Dependencies: none added.
- [x] Path confinement: not applicable.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
