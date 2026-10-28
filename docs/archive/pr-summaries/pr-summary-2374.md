## Summary

`detect_redundant_paths` ran an uncancellable O(P²·S) pair loop over a target's incoming paths and grew its candidate `Vec` to P(P−1)/2 entries (CWE-834). The scan is now bounded three ways: only the 64 strongest incoming paths by `|weight|` are compared (at most 2,016 pairs), at most 256 candidates are emitted, and the analysis deadline and global cancellation flag are checked before every pair. Partial results are returned and a warning is logged on early stop. Closes #2374.

Version bump: `Cargo.toml` updated for this change; the merged commit records the final version.

```mermaid
flowchart LR
    A[valid paths, >= 30 samples] --> B{more than 64?}
    B -- yes --> C[keep 64 strongest by abs weight]
    B -- no --> D[pair loop]
    C --> D
    D --> E{deadline or cancelled?}
    E -- yes --> F[stop: DeadlinePassed]
    E -- no --> G{256 candidates?}
    G -- yes --> H[stop: CandidateCeiling]
    G -- no --> I[evaluate pair]
    I --> D
```

## Spec

### Intent and Rationale

- The pair scan's cost and candidate count were bounded only by a target's fan-in, which `validate_creature` does not cap; a hostile model could make one target run millions of O(S) pair evaluations and materialise millions of candidates.
- The fix follows the repository's existing pair-scan precedent (`co_adaptation.rs`, `symmetry_breaking.rs`): a public `_with_deadline` variant, an `_observed` core taking a per-pair callback, a ceiling, and `ScanTruncation`.

### Essential Design Decisions

- `detect_redundant_paths` keeps its signature and delegates with no deadline, so existing callers and tests are unchanged; only `candidate_selection.rs` passes `ctx.deadline`.
- The fan-in cap keeps the strongest paths by `|existing_weight|` (the issue's suggested top-k), then restores original order so tie-breaking is unchanged for fan-ins at or under the cap. Under the cap, behaviour is identical to before.
- The deadline check precedes the ceiling check. A tie at one pair boundary reports `DeadlinePassed`; no test pins that order (see Test Plan).
- `paths_skipped` is reported separately from `truncation` (as `eligible_skipped` is in co-adaptation) and is logged by the caller.

### Undiscoverable Facts

- The production caller is `candidate_selection.rs::detect_redundant_path_candidates`; `NEAT_AI_DISCOVERY_MAX_SOURCES_PER_TARGET` is unset by default, so nothing else bounds P.

## Evidence

Backend-only change; no UI files touched.

Original trigger is closed with no trivial bypass: the only production caller of the pair loop is `detect_redundant_path_candidates`, which now goes through `detect_redundant_paths_observed`. In that function the pair loop runs over at most `MAX_REDUNDANT_PATH_FAN_IN` (64) paths, so P cannot exceed 64 whatever the fan-in; the candidate `Vec` cannot exceed `MAX_REDUNDANT_PATH_CANDIDATES` (256) because the check precedes every evaluation; and the deadline/cancel check precedes every pair. A large fan-in of near-identical sources, the original attack input, now costs at most 2,016 pair evaluations and 256 candidates. No other path reaches `evaluate_pair_for_redundancy`.

Regression test: `tests/issue_2374_redundant_path_pair_scan.rs::pair_scan_grows_linearly_with_fan_in`. It uses only the pre-existing `detect_redundant_paths` API, so it compiles and runs on the base branch. Against the unfixed `redundant_path.rs` and `candidate_selection.rs` it fails at runtime: `16 paths gave 120 candidates, 64 paths gave 2016 (limit 480)`. At the head it passes (120 and 256).

## Test Plan

Added (all pass at the head):

- `tests/issue_2374_redundant_path_pair_scan.rs::pair_scan_grows_linearly_with_fan_in`: candidates at 4P paths are at most 4× those at P, with a positive control that P=16 yields candidates.
- `tests/issue_2374_redundant_path_scan_bounds.rs` (7 tests): `candidate_ceiling_stops_the_scan`, `fan_in_cap_bounds_pair_evaluations`, `fan_in_cap_keeps_the_strongest_weights`, its positive control, `under_the_cap_nothing_is_skipped`, `far_future_deadline_matches_no_deadline`, `elapsed_deadline_returns_deadline_passed`.
- `src/analysis/detection/redundant_path_deadline_tests.rs` (2 `#[serial]` tests): `pair_loop_deadline_check_stops_a_scan_that_started_before_the_deadline`, `no_deadline_override_leaves_full_scan_untouched`.

No existing test assertion was edited or removed. The 11 existing `redundant_path` unit tests pass unchanged.

Red checks (one guard broken at a time, then restored):

- Whole change reverted (test A): red, 120 vs 2016.
- Deadline block deleted: `pair_loop_deadline_check_stops_a_scan_that_started_before_the_deadline` and `elapsed_deadline_returns_deadline_passed` red.
- Candidate-ceiling block deleted: `candidate_ceiling_stops_the_scan` (2016 vs 256) and `pair_scan_grows_linearly_with_fan_in` red.
- Weight sort removed (cap keeps first 64 by position): `fan_in_cap_keeps_the_strongest_weights` red.
- Fan-in cap disabled: `fan_in_cap_bounds_pair_evaluations` (19900 vs 2016) and `fan_in_cap_keeps_the_strongest_weights` red.
- Ceiling check moved before the deadline check: no test goes red. The order matters only when both fire at the same pair; this is pinned by no test.

Commands run on the final head: `cargo test --lib redundant_path` (13 passed), `cargo test --test issue_2374_redundant_path_pair_scan --test issue_2374_redundant_path_scan_bounds` (1 and 7 passed), `cargo fmt --check` (clean), `cargo clippy --all-targets --all-features -- -D warnings` (clean), `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --all-features` (clean).

Entry points: `candidate_selection.rs::detect_redundant_path_candidates` now passes `&ctx.deadline`. `candidate_selection.rs:detect_redundant_path_candidates` `&ctx.deadline` argument is `exempt (untestable)`: `TargetAnalysisContext` is built only in `src/analysis/synapse/orchestration.rs`, and a grep of `src` and `tests` for `TargetAnalysisContext` finds no test that constructs it, so no existing harness can drive the function with an elapsed deadline. The deadline behaviour itself is tested at `detect_redundant_paths_observed`.

Callers of the changed function: `detect_redundant_paths` (unchanged signature, unit and integration tests exercise it) and `detect_redundant_path_candidates` (switched to the deadline variant).

Docs updated: `docs/DISCOVERY_TYPES.md` (Redundant Path Pruning), `docs/ANALYSIS_DEEP_DIVE.md` (Redundant Path Pruning with Renormalisation), `docs/discoveries/redundant-path.md` (new "Scan Bounds" section). Existing sentences saying the scan covered "all" incoming sources were reworded.

<!-- vibe-quality-gate-skipped reason="the full ./quality.sh takes about 15 minutes, longer than the 10-minute cap on a single foreground command in this run; the targeted checks above were run instead, and CI runs the full gate" -->

Not run: the full `./quality.sh` and `cargo test --tests --all-features`.
